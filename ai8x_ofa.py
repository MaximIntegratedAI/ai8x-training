###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Contains the custom PyTorch modules for Once For All[1] training that take the AI84/AI85/AI87
implementations into account.

[1] Cai, Han, et al. "Once-for-all: Train one network and specialize it for efficient deployment."
arXiv preprint arXiv:1908.09791 (2019).
"""

import abc
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import ai8x
from ai8x import get_activation, quantize_clamp, quantize_clamp_pool


class OnceForAllModule(nn.Module):
    """
    AI8X - Common code for Once for All NAS layers
    """
    def __init__(self, pooling=None, activation=None, wide=False, pool=None, op=None, func=None,
                 bn=None, **_):
        super().__init__()

        self.pooling = pooling
        self.clamp = None
        self.clamp_pool = None
        self.activate = get_activation(activation)
        self.wide = wide
        self.pool = pool
        self.op = op
        self.func = func
        self.bn = bn

        self.quantize = None
        self.clamp = None
        self.quantize_pool = None
        self.clamp_pool = None

        if op is not None:
            self.in_channels = op.weight.shape[1]
            self.out_channels = op.weight.shape[0]
            self.max_kernel_size = op.weight.shape[2]  # kernel must be 1D or 2D square
            self.max_pad_size = op.padding[0]
            self.kernel_size = self.max_kernel_size
            self.pad = self.max_pad_size
            self.kernel_list = []
            self.padding_list = []
            self.ktm_list = torch.nn.ParameterList()
            self.in_ch_order = torch.arange(self.in_channels)
            self.out_ch_order = torch.arange(self.out_channels)

            if op.__class__.__name__.endswith('1d'):
                kernel_size = self.max_kernel_size - 2
                padding = self.max_pad_size - 1
                while (kernel_size > 0) and (padding >= 0):
                    self.kernel_list.append(kernel_size)
                    self.padding_list.append(padding)
                    ktm = torch.zeros(self.max_kernel_size, kernel_size)
                    j = (self.max_kernel_size-kernel_size)//2
                    for i in range(kernel_size):
                        ktm[j, i] = 1.
                        j += 1
                    self.ktm_list.append(nn.Parameter(data=ktm, requires_grad=True))
                    kernel_size -= 2
                    padding -= 1
            elif op.__class__.__name__.endswith('2d'):
                if (self.max_kernel_size == 3) and (self.max_pad_size >= 0):
                    self.kernel_list.append(1)
                    self.padding_list.append(0)
                    ktm = torch.zeros(self.max_kernel_size**2, 1)
                    ktm[self.max_kernel_size**2 // 2] = 1
                    self.ktm_list.append(nn.Parameter(data=ktm, requires_grad=True))
            else:
                assert False, f'Unknown operation for OFA module: {op}'

            # parameters to store in the checkpoint file
            self.max_kernel_size = nn.Parameter(data=torch.tensor(self.max_kernel_size),
                                                requires_grad=False)
            self.kernel_list = nn.Parameter(data=torch.Tensor(self.kernel_list),
                                            requires_grad=False)
            self.padding_list = nn.Parameter(data=torch.Tensor(self.padding_list),
                                             requires_grad=False)

        self.init_module()

    def init_module(self):
        """Initialize module paramaters"""
        self.set_functions()

    def set_functions(self):
        """Set functions wrt defined module parameters"""
        self.quantize, self.clamp = quantize_clamp(self.wide, False)
        self.quantize_pool, self.clamp_pool = quantize_clamp_pool(self.pooling, False)

    def set_channels(self, in_channels=None, out_channels=None):
        """Set channels"""
        if in_channels:
            self.in_channels = in_channels
        if out_channels:
            self.out_channels = out_channels

    def set_kernel_size(self, kernel_size):
        """Set kernel size"""
        self.kernel_size = kernel_size

    def sample_subnet_kernel(self, level):
        """OFA Elastic kernel search strategy"""
        kernel_opts = [int(self.max_kernel_size.detach().cpu().item())]
        kernel_list = self.kernel_list.detach().cpu().numpy()
        k_level = level if level >= 0 else kernel_list.size
        for i in range(k_level):
            kernel_opts.append(int(kernel_list[i]))
        with torch.no_grad():
            self.kernel_size = random.choice(kernel_opts)

    def reset_kernel_sampling(self):
        """Resets kernel to maximum widths"""
        with torch.no_grad():
            self.set_kernel_size(self.op.weight.shape[2])

    def set_out_ch_order(self, inds, reset_order=False):
        """Set order of the output channel of the operators"""
        if reset_order:
            self.reset_out_ch_order()
            self.out_ch_order = inds
        else:
            self.out_ch_order = self.out_ch_order[inds]

        self.op.weight.data = self.op.weight.data[inds]
        if self.op.bias is not None:
            self.op.bias.data = self.op.bias.data[inds]
        if self.bn is not None:
            self.bn.weight.data = self.bn.weight.data[inds]
            self.bn.bias.data = self.bn.bias.data[inds]
            self.bn.running_mean.data = self.bn.running_mean.data[inds]
            self.bn.running_var.data = self.bn.running_var.data[inds]

    def reset_out_ch_order(self):
        """Reset order of the output channel of the operators"""
        reset_ind = torch.argsort(self.out_ch_order)
        self.set_out_ch_order(reset_ind)

    def set_in_ch_order(self, inds, reset_order=False):
        """Set order of the input channel of the operators"""
        if reset_order:
            self.reset_in_ch_order()
            self.in_ch_order = inds
        else:
            self.in_ch_order = self.in_ch_order[inds]

        self.op.weight.data = self.op.weight.data[:, inds]

    def reset_in_ch_order(self):
        """Reset order of the input channel of the operators"""
        reset_ind = torch.argsort(self.in_ch_order)
        self.set_in_ch_order(reset_ind)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        if self.pool is not None:
            x = self.clamp_pool(self.quantize_pool(self.pool(x)))
        if self.op is not None:
            weight = self.op.weight[:self.out_channels, :self.in_channels]
            bias = self.op.bias
            if bias is not None:
                bias = bias[:self.out_channels]

            if self.kernel_size == int(self.max_kernel_size.detach().cpu().item()):
                x = self.func(x, weight, bias, self.op.stride, self.max_pad_size, self.op.dilation,
                              self.op.groups)
            else:
                for k_idx, k_size in enumerate(self.kernel_list):
                    if k_size == self.kernel_size:
                        break
                if weight.dim() == 4:
                    flattened_weight = weight.view(weight.size(0), weight.size(1), -1,
                                                   self.max_kernel_size**2)
                else:
                    flattened_weight = weight
                # pylint: disable=undefined-loop-variable
                weight = flattened_weight @ self.ktm_list[k_idx]
                # pylint: disable=undefined-loop-variable
                pad = int(self.padding_list[k_idx].detach().cpu().item())
                x = self.func(x, weight, bias, self.op.stride, pad, self.op.dilation,
                              self.op.groups)

            if self.bn is not None:
                x = F.batch_norm(x, self.bn.running_mean[:self.out_channels],
                                 self.bn.running_var[:self.out_channels],
                                 self.bn.weight[:self.out_channels],
                                 self.bn.bias[:self.out_channels],
                                 self.bn.training,
                                 self.bn.momentum,
                                 self.bn.eps)
                x /= 4.
            x = self.clamp(self.quantize(self.activate(x)))
        return x


class Conv2d(OnceForAllModule):
    """
    AI8X-OnceForAll - 2D pooling ('Avg', 'Max' or None) optionally followed by
    2D convolution/transposed 2D convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            op='Conv2d',
            pooling=None,
            pool_size=2,
            pool_stride=2,
            stride=1,
            padding=0,
            bias=True,
            activation=None,
            wide=False,
            batchnorm=None,
            **_
    ):
        assert not wide or activation is None

        if pooling is not None:
            if pool_stride is None:
                pool_stride = pool_size

            if isinstance(pool_size, int):
                assert ai8x.dev.device != 84 or pool_size & 1 == 0
                assert pool_size <= 16 \
                    and (ai8x.dev.device != 84 or pool_size <= 4 or pooling == 'Max')
            elif isinstance(pool_size, tuple):
                assert len(pool_size) == 2
                assert ai8x.dev.device != 84 or pool_size[0] & 1 == 0
                assert pool_size[0] <= 16 \
                    and (ai8x.dev.device != 84 or pool_size[0] <= 4 or pooling == 'Max')
                assert ai8x.dev.device != 84 or pool_size[1] & 1 == 0
                assert pool_size[1] <= 16 \
                    and (ai8x.dev.device != 84 or pool_size[1] <= 4 or pooling == 'Max')
            else:
                raise ValueError('pool_size must be int or tuple')

            if isinstance(pool_stride, int):
                assert pool_stride > 0
                assert pool_stride <= 16 \
                    and (ai8x.dev.device != 84 or pool_stride <= 4 or pooling == 'Max')
            elif isinstance(pool_stride, tuple):
                assert len(pool_stride) == 2
                assert ai8x.dev.device != 84 or pool_stride[0] == pool_stride[1]
                assert 0 < pool_stride[0] <= 16 \
                    and (ai8x.dev.device != 84 or pool_stride[0] <= 4 or pooling == 'Max')
                assert 0 < pool_stride[1] <= 16 \
                    and (ai8x.dev.device != 84 or pool_stride[1] <= 4 or pooling == 'Max')
            else:
                raise ValueError('pool_stride must be int or tuple')

            if op == 'ConvTranspose2d':
                assert stride == 2
            else:
                assert stride == 1
        else:
            if op == 'ConvTranspose2d':
                assert stride == 2
            else:
                assert 0 < stride <= 3

        assert 0 <= padding <= 2

        if pooling == 'Max':
            pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride, padding=0)
        elif pooling == 'Avg':
            pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride, padding=0)
        else:
            pool = None

        if batchnorm == 'Affine':
            bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.05, affine=True)
            assert bias, '`bias` must be set (enable --use-bias for models where bias is optional)'
        elif batchnorm == 'NoAffine':
            bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.05, affine=False)
            assert bias, '`bias` must be set (enable --use-bias for models where bias is optional)'
        else:
            bn = None

        if kernel_size is not None:
            if isinstance(kernel_size, tuple):
                assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
                kernel_size = kernel_size[0]

            assert kernel_size == 3 or ai8x.dev.device != 84 and kernel_size == 1

            if op == 'Conv2d':
                opn = nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias)
            elif op == 'ConvTranspose2d':
                assert ai8x.dev.device != 84
                opn = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         output_padding=1,
                                         padding=padding, bias=bias)
            else:
                raise ValueError('Unsupported operation')
        else:
            opn = None

        if op == 'ConvTranspose2d':
            func = nn.functional.conv_transpose2d
        else:
            func = nn.functional.conv2d

        super().__init__(
            pooling,
            activation,
            wide,
            pool,
            opn,
            func,
            bn,
        )


class FusedConv2dReLU(Conv2d):
    """
    AI8X-OnceForAll - Fused 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedConv2dBNReLU(FusedConv2dReLU):
    """
    AI8X-OnceForAll - Fused 2D Convolution and BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batchnorm='Affine', **kwargs)


class FusedMaxPoolConv2d(Conv2d):
    """
    AI8X-OnceForAll - Fused 2D Max Pool, 2D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConv2dBN(FusedMaxPoolConv2d):
    """
    AI8X-OnceForAll - Fused 2D Max Pool, 2D Convolution, BatchNorm and
    Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batchnorm='Affine', **kwargs)


class FusedMaxPoolConv2dReLU(FusedMaxPoolConv2d):
    """
    AI8X-OnceForAll - Fused 2D Max Pool, 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConv2dBNReLU(FusedMaxPoolConv2dReLU):
    """
    AI8X-OnceForAll - Fused 2D Max Pool, 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batchnorm='Affine', **kwargs)


class Conv1d(OnceForAllModule):
    """
    AI8X-OnceForAll - Fused 1D Pool ('Avg', 'Max' or None) followed by
    1D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            pooling=None,
            pool_size=2,
            pool_stride=2,
            stride=1,
            padding=0,
            bias=True,
            activation=None,
            wide=False,
            batchnorm=None,
            **_
    ):
        assert not wide or activation is None

        if pooling is not None:
            if pool_stride is None:
                pool_stride = pool_size

            assert ai8x.dev.device != 84 or pool_size & 1 == 0
            assert pool_size <= 16 \
                and (ai8x.dev.device != 84 or pool_size <= 4 or pooling == 'Max')

            assert 0 < pool_stride <= 16 \
                and (ai8x.dev.device != 84 or pool_stride <= 4 or pooling == 'Max')

            assert stride == 1
        else:
            assert ai8x.dev.device != 84 or stride == 3
            assert ai8x.dev.device == 84 or stride == 1

        if pooling == 'Max':
            pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride, padding=0)
        elif pooling == 'Avg':
            pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride, padding=0)
        else:
            pool = None

        if batchnorm == 'Affine':
            bn = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.05, affine=True)
            assert bias, '`bias` must be set (enable --use-bias for models where bias is optional)'
        elif batchnorm == 'NoAffine':
            bn = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.05, affine=False)
            assert bias, '`bias` must be set (enable --use-bias for models where bias is optional)'
        else:
            bn = None

        if kernel_size is not None:
            assert ai8x.dev.device != 84 or padding in [0, 3, 6]
            assert ai8x.dev.device == 84 or padding in [0, 1, 2]
            assert ai8x.dev.device != 84 or kernel_size == 9
            assert ai8x.dev.device == 84 or kernel_size in [1, 2, 3, 4, 5, 6, 7, 8, 9]

            opn = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, bias=bias)
        else:
            opn = None

        super().__init__(
            pooling,
            activation,
            wide,
            pool,
            opn,
            nn.functional.conv1d,
            bn,
        )


class FusedConv1dReLU(Conv1d):
    """
    AI8X-OnceForAll - Fused 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedConv1dBNReLU(FusedConv1dReLU):
    """
    AI8X-OnceForAll - Fused 1D Convolution and BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batchnorm='Affine', **kwargs)


class FusedMaxPoolConv1d(Conv1d):
    """
    AI8X-OnceForAll - Fused 1D Max Pool, 1D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConv1dBN(FusedMaxPoolConv1d):
    """
    AI8X-OnceForAll - Fused 1D Max Pool, 1D Convolution, BatchNorm and
    Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batchnorm='Affine', **kwargs)


class FusedMaxPoolConv1dReLU(FusedMaxPoolConv1d):
    """
    AI8X-OnceForAll - Fused 1D Max Pool, 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConv1dBNReLU(FusedMaxPoolConv1dReLU):
    """
    AI8X-OnceForAll - Fused 1D Max Pool, 1D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batchnorm='Affine', **kwargs)


class OnceForAllUnit(metaclass=abc.ABCMeta):
    """
    AI8X-OnceForAll - Interface for unit definition
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'sample_subnet_depth') and
                callable(subclass.sample_subnet_depth) and
                hasattr(subclass, 'reset_depth_sampling') and
                callable(subclass.reset_depth_sampling) and
                hasattr(subclass, 'get_max_elastic_depth_level') and
                callable(subclass.get_max_elastic_depth_level))


class OnceForAllModel(metaclass=abc.ABCMeta):
    """
    AI8X-OnceForAll - Interface for model definition
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'sample_subnet_width') and
                callable(subclass.sample_subnet_width) and
                hasattr(subclass, 'reset_width_sampling') and
                callable(subclass.reset_width_sampling) and
                hasattr(subclass, 'sample_subnet_depth') and
                callable(subclass.sample_subnet_depth) and
                hasattr(subclass, 'reset_depth_sampling') and
                callable(subclass.reset_depth_sampling) and
                hasattr(subclass, 'sample_subnet_kernel') and
                callable(subclass.sample_subnet_kernel) and
                hasattr(subclass, 'reset_kernel_sampling') and
                callable(subclass.reset_kernel_sampling) and
                hasattr(subclass, 'get_max_elastic_width_level') and
                callable(subclass.get_max_elastic_width_level) and
                hasattr(subclass, 'get_max_elastic_depth_level') and
                callable(subclass.get_max_elastic_depth_level) and
                hasattr(subclass, 'get_max_elastic_kernel_level') and
                callable(subclass.get_max_elastic_kernel_level))


def sample_subnet_kernel(ofa_model, level=0):
    """
    Sample kernels of the OnceForAll modules in the model
    """
    def _sample_subnet_kernel(m):
        if isinstance(m, OnceForAllModel):
            m.sample_subnet_kernel(level)

    ofa_model.apply(_sample_subnet_kernel)


def reset_kernel_sampling(ofa_model):
    """
    Reset kernel sampling for OnceForAll modules in the model
    """
    def _reset_kernel_sampling(m):
        if isinstance(m, OnceForAllModel):
            m.reset_kernel_sampling()

    ofa_model.apply(_reset_kernel_sampling)


def sample_subnet_depth(ofa_model, level=0, sample_kernel=True):
    """
    Sample depths of the OnceForAll units in the model
    """
    def _sample_subnet_depth(m):
        if isinstance(m, OnceForAllModel):
            if sample_kernel:
                m.sample_subnet_kernel(level=-1)
            m.sample_subnet_depth(level)

    ofa_model.apply(_sample_subnet_depth)


def reset_depth_sampling(ofa_model):
    """
    Reset depth sampling for OnceForAll modules in the model
    """
    def _reset_depth_sampling(m):
        if isinstance(m, OnceForAllModel):
            m.reset_kernel_sampling()
            m.reset_depth_sampling()

    ofa_model.apply(_reset_depth_sampling)


def sample_subnet_width(ofa_model, level=0, sample_depth=True):
    """
    Sample widths of the OnceForAll layers in the model
    """
    def _sample_subnet_width(m):
        if isinstance(m, OnceForAllModel):
            if sample_depth:
                with torch.no_grad():
                    sample_subnet_depth(m, level=-1)

            m.sample_subnet_width(level)

    ofa_model.apply(_sample_subnet_width)


def reset_width_sampling(ofa_model):
    """
    Reset width sampling for OnceForAll layers in the model
    """
    def _reset_width_sampling(m):
        if isinstance(m, OnceForAllModel):
            reset_depth_sampling(m)
            m.reset_width_sampling()

    ofa_model.apply(_reset_width_sampling)
