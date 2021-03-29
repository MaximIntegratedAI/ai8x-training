###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Contains the limits of the MAX78000 implementations and custom PyTorch modules that take
the limits into account.
"""
import torch
import torch.nn as nn
from torch.autograd import Function

import devices

dev = None


class normalize:
    """
    Normalize input to either [-128/128, +127/128] or [-128, +127]
    """
    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        if self.args.act_mode_8bit:
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127).div(128.)


class QuantizationFunction(Function):
    """
    Custom autograd function
    The forward pass divides by 2**(bits-1) (typically, 128) and rounds the result to the
    nearest integer.
    The backward pass is straight through.
    """
    @staticmethod
    def forward(_, x, bits=None, extra_bit_shift=None):  # pylint: disable=arguments-differ
        """Forward prop"""
        if dev.simulate:
            if bits > 1:
                return x.add(.5).div(2**(bits+extra_bit_shift-1)).add(.5).floor()
            if bits < 1:
                return x.mul(2**(1-bits-extra_bit_shift)).add(.5).floor()
            return x.add(.5).floor()

        factor1 = 2**(bits-extra_bit_shift-1)
        factor2 = 2**(bits-1)
        return x.mul(factor1).add(.5).floor().div(factor2)

    @staticmethod
    def backward(_, x):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x, None, None


class Quantize(nn.Module):
    """
    Post-activation integer quantization module
    Apply the custom autograd function
    """
    def __init__(self, num_bits=8, num_extra_bit_shift=0):
        super().__init__()
        self.num_bits = num_bits
        self.num_extra_bit_shift = num_extra_bit_shift

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return QuantizationFunction.apply(x, self.num_bits, self.num_extra_bit_shift)


class FloorFunction(Function):
    """
    Custom MAX78000 autograd function
    The forward pass returns the integer floor.
    The backward pass is straight through.
    """
    @staticmethod
    def forward(_, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x.floor()

    @staticmethod
    def backward(_, x):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x


class Floor(nn.Module):
    """
    Post-pooling integer quantization module
    Apply the custom autograd function
    """
    def forward(self, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        return FloorFunction.apply(x)


class RoundFunction(Function):
    """
    Custom MAX78000 autograd function
    The forward pass returns the integer rounded.
    The backward pass is straight through.
    """
    @staticmethod
    def forward(_, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x.round()

    @staticmethod
    def backward(_, x):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x


class Round(nn.Module):
    """
    Post-pooling integer quantization module
    Apply the custom autograd function
    """
    def forward(self, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        return RoundFunction.apply(x)


class Clamp(nn.Module):
    """
    Post-Activation Clamping Module
    Clamp the output to the given range (typically, [-128, +127])
    """
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x.clamp(min=self.min_val, max=self.max_val)


class Scaler(nn.Module):
    """
    Scaler module that considers integer quantization
    Apply the custom autograd function
    """
    def forward(self, x, s):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        if dev.simulate:
            return FloorFunction.apply(x*s)
        return x*s


class RoundQat(nn.Module):
    """
    Round function for AvgPool in QAT mode
    """
    def forward(self, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        factor = 2**(dev.ACTIVATION_BITS - 1)
        return RoundFunction.apply(x.mul(factor)).div(factor)


class FloorQat(nn.Module):
    """
    Floor function for AvgPool in QAT mode
    """
    def forward(self, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        factor = 2**(dev.ACTIVATION_BITS - 1)
        return FloorFunction.apply(x.mul(factor)).div(factor)


def quantize_clamp(wide, quantize_activation=False):
    """
    Return new Quantization and Clamp objects.
    """
    if dev.simulate:
        if not wide:
            quantize = Quantize(num_bits=dev.DATA_BITS)
            clamp = Clamp(
                min_val=-(2**(dev.ACTIVATION_BITS-1)),
                max_val=2**(dev.ACTIVATION_BITS-1)-1,
            )
        else:
            quantize = Quantize(num_bits=dev.DATA_BITS, num_extra_bit_shift=1)
            clamp = Clamp(
                min_val=-(2**(dev.FULL_ACC_BITS-1)),
                max_val=2**(dev.FULL_ACC_BITS-1)-1,
            )
    else:
        if quantize_activation:
            if not wide:
                quantize = Quantize(num_bits=dev.ACTIVATION_BITS)
            else:
                quantize = Quantize(num_bits=dev.DATA_BITS, num_extra_bit_shift=1)
        else:
            quantize = Empty()
        if not wide:
            clamp = Clamp(  # Do not combine with ReLU
                min_val=-1.,
                max_val=(2.**(dev.ACTIVATION_BITS-1)-1)/(2.**(dev.ACTIVATION_BITS-1)),
            )
        else:
            clamp = Clamp(
                min_val=-(2.**((dev.FULL_ACC_BITS-2*(dev.DATA_BITS-1))-1)),
                max_val=2.**((dev.FULL_ACC_BITS-2*(dev.DATA_BITS-1))-1),
            )

    return quantize, clamp


def quantize_clamp_pool(pooling, quantize_activation=False):
    """
    Return new Quantization and Clamp objects for pooling.
    """
    if dev.simulate:
        if pooling == 'Avg':
            quantize = Round() if dev.round_avg else Floor()
            clamp = Clamp(
                min_val=-(2**(dev.DATA_BITS-1)),
                max_val=2**(dev.DATA_BITS-1)-1,
            )
        else:  # Max, None
            quantize = Empty()
            clamp = Empty()
    else:
        quantize = Empty()
        if pooling == 'Avg':
            if quantize_activation:
                quantize = RoundQat() if dev.round_avg else FloorQat()
            clamp = Clamp(min_val=-1., max_val=127./128.)
        else:  # Max, None
            clamp = Empty()

    return quantize, clamp


def quantize_clamp_parameters(weight_bits, bias_bits):
    """
    Return new Quantization and Clamp objects for weight and bias parameters
    """
    if dev.simulate:
        quantize_weight = Quantize(num_bits=weight_bits-dev.DATA_BITS+1)
        quantize_bias = Quantize(num_bits=weight_bits-dev.DATA_BITS+1)
        clamp_weight = Empty()
        clamp_bias = Empty()
    else:
        if weight_bits == 0 and bias_bits == 0:
            quantize_weight = Empty()
            quantize_bias = Empty()
            clamp_weight = Empty()
            clamp_bias = Empty()
        else:
            quantize_weight = Quantize(num_bits=weight_bits)
            quantize_bias = Quantize(num_bits=bias_bits)
            clamp_weight = Clamp(min_val=-1.,
                                 max_val=(2.**(weight_bits-1)-1)/(2.**(weight_bits-1)))
            clamp_bias = Clamp(min_val=-1., max_val=(2.**(bias_bits-1)-1)/(2.**(bias_bits-1)))

    return quantize_weight, quantize_bias, clamp_weight, clamp_bias


class OutputShiftSqueeze(nn.Module):
    """
    Return output_shift when not using quantization-aware training.
    """
    def forward(self, _, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        return x.squeeze(0)


class OutputShift(nn.Module):
    """
    Calculate the clamped output shift when adjusting during quantization-aware training.
    """
    def __init__(self, shift_quantile=1.0):
        super().__init__()
        self.shift_quantile = shift_quantile

    def forward(self, x, _):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        limit = torch.quantile(x.abs(), self.shift_quantile)
        return -(1./limit).log2().floor().clamp(min=-15., max=15.)


class One(nn.Module):
    """
    Return 1.
    """
    def forward(self, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        return torch.ones(1).to(x.device)


class WeightScale(nn.Module):
    """
    Calculate the weight scale (square root of the output shift)
    """
    def forward(self, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        return 2.**(-x)


class OutputScale(nn.Module):
    """
    Calculate the output scale (square of the output shift)
    """
    def forward(self, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        return 2.**x


class Abs(nn.Module):
    """
    Return abs(x)
    """
    def forward(self, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        return torch.abs_(x)  # abs_() is the in-place version


class Empty(nn.Module):
    """
    Do nothing
    """
    def forward(self, x):  # pylint: disable=arguments-differ, no-self-use
        """Forward prop"""
        return x


def get_activation(activation=None):
    """
    Return the selected `activation` class ('ReLU', 'Abs', None)
    """
    if activation == 'ReLU':
        return nn.ReLU(inplace=True)
    if activation == 'Abs':
        assert dev.device != 84
        return Abs()
    return Empty()


class QuantizationAwareModule(nn.Module):
    """
    Common code for Quantization-Aware Training
    """
    def __init__(
            self,
            pooling=None,
            activation=None,
            wide=False,
            weight_bits=None,
            bias_bits=None,
            quantize_activation=False,
            pool=None,
            op=None,
            func=None,
            bn=None,
            shift_quantile=None
    ):
        super().__init__()

        assert weight_bits in [None, 1, 2, 4, 8], f'Weight bits cannot be {weight_bits}'
        assert bias_bits in [None, 8], f'Bias bits cannot be {bias_bits}'

        self.quantize = None
        self.clamp = None
        self.quantize_bias = None
        self.clamp_bias = None
        self.calc_out_shift = None
        self.scale = None
        self.calc_weight_scale = None
        self.calc_out_scale = None
        self.quantize_weight = None
        self.clamp_weight = None
        self.quantize_pool = None
        self.clamp_pool = None

        self.activate = get_activation(activation)
        self.wide = wide

        self.pool = pool
        self.op = op
        self.func = func
        self.bn = bn
        self.pooling = pooling

        self.output_shift = nn.Parameter(torch.Tensor([0.]), requires_grad=False)
        self.init_module(weight_bits, bias_bits, quantize_activation, shift_quantile)

    def init_module(self, weight_bits, bias_bits, quantize_activation, shift_quantile):
        """Initialize model parameters"""
        if weight_bits is None and bias_bits is None and not quantize_activation:
            self.weight_bits = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.bias_bits = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.quantize_activation = nn.Parameter(torch.Tensor([False]), requires_grad=False)
            self.adjust_output_shift = nn.Parameter(torch.Tensor([False]), requires_grad=False)
        elif weight_bits in [1, 2, 4, 8] and bias_bits == 8 and quantize_activation:
            self.weight_bits = nn.Parameter(torch.Tensor([weight_bits]), requires_grad=False)
            self.bias_bits = nn.Parameter(torch.Tensor([bias_bits]), requires_grad=False)
            self.quantize_activation = nn.Parameter(torch.Tensor([True]), requires_grad=False)
            self.adjust_output_shift = nn.Parameter(torch.Tensor([not dev.simulate]),
                                                    requires_grad=False)
        else:
            assert False, f'Undefined mode with weight_bits: {weight_bits}, ' \
                          f'bias_bits: {bias_bits}, ' \
                          f'quantize_activation: {quantize_activation}'

        self.shift_quantile = shift_quantile
        self.set_functions()

    def set_functions(self):
        """Set functions to be used wrt the model parameters"""
        if self.adjust_output_shift.detach():
            self.calc_out_shift = OutputShift(self.shift_quantile)
            self.calc_weight_scale = WeightScale()
        else:
            self.calc_out_shift = OutputShiftSqueeze()
            self.calc_weight_scale = One()

        self.scale = Scaler()
        self.calc_out_scale = OutputScale()

        self.quantize_weight, self.quantize_bias, self.clamp_weight, self.clamp_bias = \
            quantize_clamp_parameters(self.weight_bits.detach().item(),
                                      self.bias_bits.detach().item())
        self.quantize, self.clamp = \
            quantize_clamp(self.wide, self.quantize_activation.detach().item())
        self.quantize_pool, self.clamp_pool = \
            quantize_clamp_pool(self.pooling, self.quantize_activation.detach().item())

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        if self.pool is not None:
            x = self.clamp_pool(self.quantize_pool(self.pool(x)))
        if self.op is not None:
            if self.op.bias is not None:
                bias_r = torch.flatten(self.op.bias.detach())
                weight_r = torch.flatten(self.op.weight.detach())
                params_r = torch.cat((weight_r, bias_r))
            else:
                params_r = torch.flatten(self.op.weight.detach())
            out_shift = self.calc_out_shift(params_r, self.output_shift.detach())
            weight_scale = self.calc_weight_scale(out_shift)
            out_scale = self.calc_out_scale(out_shift)

            self.output_shift = nn.Parameter(out_shift.unsqueeze(0), requires_grad=False)

            weight = self.clamp_weight(self.quantize_weight(weight_scale * self.op.weight))
            bias = self.op.bias
            if bias is not None:
                bias = self.clamp_bias(self.quantize_bias(weight_scale * bias))

            x = self.func(x, weight, bias, self.op.stride, self.op.padding,
                          self.op.dilation, self.op.groups)
            if self.bn is not None:
                x = self.bn(x) / 4
            x = self.clamp(self.quantize(self.activate(self.scale(x, out_scale))))
        return x


class Conv2d(QuantizationAwareModule):
    """
    2D pooling ('Avg', 'Max' or None) optionally followed by
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
            weight_bits=None,
            bias_bits=None,
            quantize_activation=False,
    ):
        assert not wide or activation is None

        if pooling is not None:
            if pool_stride is None:
                pool_stride = pool_size

            if isinstance(pool_size, int):
                assert dev.device != 84 or pool_size & 1 == 0
                assert pool_size <= 16 \
                    and (dev.device != 84 or pool_size <= 4 or pooling == 'Max')
            elif isinstance(pool_size, tuple):
                assert len(pool_size) == 2
                assert dev.device != 84 or pool_size[0] & 1 == 0
                assert pool_size[0] <= 16 \
                    and (dev.device != 84 or pool_size[0] <= 4 or pooling == 'Max')
                assert dev.device != 84 or pool_size[1] & 1 == 0
                assert pool_size[1] <= 16 \
                    and (dev.device != 84 or pool_size[1] <= 4 or pooling == 'Max')
            else:
                raise ValueError('pool_size must be int or tuple')

            if isinstance(pool_stride, int):
                assert pool_stride > 0
                assert pool_stride <= 16 \
                    and (dev.device != 84 or pool_stride <= 4 or pooling == 'Max')
            elif isinstance(pool_stride, tuple):
                assert len(pool_stride) == 2
                assert dev.device != 84 or pool_stride[0] == pool_stride[1]
                assert 0 < pool_stride[0] <= 16 \
                    and (dev.device != 84 or pool_stride[0] <= 4 or pooling == 'Max')
                assert 0 < pool_stride[1] <= 16 \
                    and (dev.device != 84 or pool_stride[1] <= 4 or pooling == 'Max')
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

            assert kernel_size == 3 or dev.device != 84 and kernel_size == 1

            if op == 'Conv2d':
                opn = nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias)
            elif op == 'ConvTranspose2d':
                assert dev.device != 84
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
            weight_bits,
            bias_bits,
            quantize_activation,
            pool,
            opn,
            func,
            bn,
        )


class FusedMaxPoolConv2d(Conv2d):
    """
    Fused 2D Max Pool, 2D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConv2dBN(FusedMaxPoolConv2d):
    """
    Fused 2D Max Pool, 2D Convolution, BatchNorm and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConv2dReLU(FusedMaxPoolConv2d):
    """
    Fused 2D Max Pool, 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConv2dBNReLU(FusedMaxPoolConv2dReLU):
    """
    Fused 2D Max Pool, 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConv2dAbs(FusedMaxPoolConv2d):
    """
    Fused 2D Max Pool, 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedMaxPoolConv2dBNAbs(FusedMaxPoolConv2dAbs):
    """
    Fused 2D Max Pool, 2D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class MaxPool2d(FusedMaxPoolConv2d):
    """
    2D Max Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__(0, 0, None, pool_size=kernel_size, pool_stride=stride,
                         activation=None, **kwargs)


class FusedAvgPoolConv2d(Conv2d):
    """
    Fused 2D Avg Pool, 2D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Avg', **kwargs)


class FusedAvgPoolConv2dReLU(FusedAvgPoolConv2d):
    """
    Fused 2D Avg Pool, 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedAvgPoolConv2dBNReLU(FusedAvgPoolConv2dReLU):
    """
    Fused 2D Avg Pool, 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedAvgPoolConv2dAbs(FusedAvgPoolConv2d):
    """
    Fused 2D Avg Pool, 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedAvgPoolConv2dBNAbs(FusedAvgPoolConv2dAbs):
    """
    Fused 2D Avg Pool, 2D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class AvgPool2d(FusedAvgPoolConv2d):
    """
    2D Avg Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__(0, 0, None, pool_size=kernel_size, pool_stride=stride,
                         activation=None, **kwargs)


class FusedConv2dReLU(Conv2d):
    """
    Fused 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedConv2dBN(Conv2d):
    """
    Fused 2D Convolution and BatchNorm
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedConv2dBNReLU(FusedConv2dReLU):
    """
    Fused 2D Convolution and BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedConv2dAbs(Conv2d):
    """
    Fused 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class ConvTranspose2d(Conv2d):
    """
    2D pooling ('Avg', 'Max' or None) optionally followed by
    transposed 2D convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, op='ConvTranspose2d', **kwargs)


class FusedMaxPoolConvTranspose2d(ConvTranspose2d):
    """
    Fused 2D Max Pool, Transposed 2D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConvTranspose2dReLU(FusedMaxPoolConvTranspose2d):
    """
    Fused 2D Max Pool, Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConvTranspose2dAbs(FusedMaxPoolConvTranspose2d):
    """
    Fused 2D Max Pool, Transposed 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedAvgPoolConvTranspose2d(ConvTranspose2d):
    """
    Fused 2D Avg Pool, Transposed 2D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Avg', **kwargs)


class FusedAvgPoolConvTranspose2dReLU(FusedAvgPoolConvTranspose2d):
    """
    Fused 2D Avg Pool, Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedAvgPoolConvTranspose2dAbs(FusedAvgPoolConvTranspose2d):
    """
    Fused 2D Avg Pool, Transposed 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedConvTranspose2dReLU(ConvTranspose2d):
    """
    Fused Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedConvTranspose2dAbs(ConvTranspose2d):
    """
    Fused Transposed 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedSoftwareLinearReLU(nn.Module):
    """
    Fused Linear and ReLU using Software
    """
    def __init__(self, in_features, out_features, bias=None, relu=True):
        super().__init__()

        if dev.device != 84:
            print('WARNING: SoftwareLinear should be used on AI84 only')

        self.op = nn.Linear(in_features, out_features, bias)

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS)
            bits = dev.FC_ACTIVATION_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize = Empty()
            self.clamp = Clamp(min_val=-1., max_val=127./128.)  # Do not combine with ReLU

        if relu:
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = Empty()

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.op(x)
        x = self.clamp(self.quantize(self.activate(x)))
        return x


class SoftwareLinear(FusedSoftwareLinearReLU):
    """
    Linear using Software
    """
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, relu=False, **kwargs)


def func_linear(x, weight, bias, _stride, _padding, _dilation, _groups):
    """
    Wrapper for `nn.functional.linear` that takes the same number of arguments as Conv1d/Conv2d.
    """
    return nn.functional.linear(x, weight, bias)


class Linear(QuantizationAwareModule):
    """
    Fused Linear and activation ('ReLU', 'Abs', None)
    """
    def __init__(
            self,
            in_features,
            out_features,
            pooling=None,
            bias=None,
            activation=None,
            wide=False,
            batchnorm=None,  # pylint: disable=unused-argument
            weight_bits=None,
            bias_bits=None,
            quantize_activation=False,
    ):
        assert not wide or activation is None

        assert dev.device != 84
        assert in_features <= 1024
        assert out_features <= 1024
        assert pooling is None
        assert batchnorm is None

        super().__init__(
            pooling,
            activation,
            wide,
            weight_bits,
            bias_bits,
            quantize_activation,
            None,
            nn.Linear(in_features, out_features, bias),
            func_linear,
            None,
        )

        # Define dummy arguments to make Linear and Conv1d/Conv2d compatible.
        self.op.stride = None
        self.op.padding = None
        self.op.dilation = None
        self.op.groups = None


class FusedLinearReLU(Linear):
    """
    Fused Linear and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedLinearAbs(Linear):
    """
    Fused Linear and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class Conv1d(QuantizationAwareModule):
    """
    Fused 1D Pool ('Avg', 'Max' or None) followed by
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
            stride=3,
            padding=0,
            bias=True,
            activation=None,
            wide=False,
            batchnorm=None,
            weight_bits=None,
            bias_bits=None,
            quantize_activation=False,
    ):
        assert not wide or activation is None

        if pooling is not None:
            if pool_stride is None:
                pool_stride = pool_size

            assert dev.device != 84 or pool_size & 1 == 0
            assert pool_size <= 16 \
                and (dev.device != 84 or pool_size <= 4 or pooling == 'Max')

            assert 0 < pool_stride <= 16 \
                and (dev.device != 84 or pool_stride <= 4 or pooling == 'Max')

            assert stride == 1
        else:
            assert dev.device != 84 or stride == 3
            assert dev.device == 84 or stride == 1

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
            assert dev.device != 84 or padding in [0, 3, 6]
            assert dev.device == 84 or padding in [0, 1, 2]
            assert dev.device != 84 or kernel_size == 9
            assert dev.device == 84 or kernel_size in [1, 2, 3, 4, 5, 6, 7, 8, 9]

            opn = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, bias=bias)
        else:
            opn = None

        super().__init__(
            pooling,
            activation,
            wide,
            weight_bits,
            bias_bits,
            quantize_activation,
            pool,
            opn,
            nn.functional.conv1d,
            bn,
        )


class FusedMaxPoolConv1d(Conv1d):
    """
    Fused 1D Max Pool, 1D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConv1dBN(FusedMaxPoolConv1d):
    """
    Fused 1D Max Pool, 1D Convolution, BatchNorm and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConv1dReLU(FusedMaxPoolConv1d):
    """
    Fused 1D Max Pool, 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConv1dBNReLU(FusedMaxPoolConv1dReLU):
    """
    Fused 1D Max Pool, 1D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConv1dAbs(FusedMaxPoolConv1d):
    """
    Fused 1D Max Pool, 1D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedMaxPoolConv1dBNAbs(FusedMaxPoolConv1d):
    """
    Fused 1D Max Pool, 1D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class MaxPool1d(FusedMaxPoolConv1d):
    """
    1D Max Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__(0, 0, None, pool_size=kernel_size, pool_stride=stride,
                         activation=None, **kwargs)


class FusedAvgPoolConv1d(Conv1d):
    """
    Fused 1D Avg Pool, 1D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Avg', **kwargs)


class FusedAvgPoolConv1dReLU(FusedAvgPoolConv1d):
    """
    Fused 1D Avg Pool, 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedAvgPoolConv1dBNReLU(FusedAvgPoolConv1dReLU):
    """
    Fused 1D Avg Pool, 1D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedAvgPoolConv1dAbs(FusedAvgPoolConv1d):
    """
    Fused 1D Avg Pool, 1D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedAvgPoolConv1dBNAbs(FusedAvgPoolConv1d):
    """
    Fused 1D Avg Pool, 1D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class AvgPool1d(FusedAvgPoolConv1d):
    """
    1D Avg Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__(0, 0, None, pool_size=kernel_size, pool_stride=stride,
                         activation=None, **kwargs)


class FusedConv1dReLU(Conv1d):
    """
    Fused 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedConv1dBNReLU(FusedConv1dReLU):
    """
    Fused 1D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedConv1dAbs(Conv1d):
    """
    Fused 1D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedConv1dBNAbs(FusedConv1dAbs):
    """
    Fused 1D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class Eltwise(nn.Module):
    """
    Base Class for Elementwise Operation
    """
    def __init__(self, f):
        super().__init__()
        self.f = f
        if dev.simulate:
            bits = dev.ACTIVATION_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.clamp = Clamp(min_val=-1., max_val=127./128.)

    def forward(self, *x):
        """Forward prop"""
        y = x[0]
        for i in range(1, len(x)):
            y = self.f(y, x[i])

        x = self.clamp(y)
        return x


class Add(Eltwise):
    """
    Elementwise Add Operation
    """
    def __init__(self):
        super().__init__(torch.add)


class Sub(Eltwise):
    """
    Elementwise Subtract Operation
    """

    @staticmethod
    def sub(a, b):
        """
        Subtract Tensors
        """
        return torch.add(a, torch.neg(b))

    def __init__(self):
        super().__init__(self.sub)


class Xor(Eltwise):
    """
    Elementwise Bitwise Xor Operation
    """

    @staticmethod
    def bitwise_xor(a, b):
        """
        Bitwise XOR of Tensors via int intermediate
        """
        # Convert input from float to byte
        a = a.add(.5).mul(256.).round().int()
        b = b.add(.5).mul(256.).round().int()
        # Bitwise XOR on integers, convert back to float
        return torch.bitwise_or(a, b).div(256.).sub(.5)

    def __init__(self):
        super().__init__(self.bitwise_xor)


class Or(Eltwise):
    """
    Elementwise Bitwise Or Operation
    """

    @staticmethod
    def bitwise_or(a, b):
        """
        Bitwise OR of Tensors via int intermediate
        """
        a = a.add(.5).mul(256.).round().int()
        b = b.add(.5).mul(256.).round().int()
        # Bitwise OR on integers, convert back to float
        return torch.bitwise_xor(a, b).div(256.).sub(.5)

    def __init__(self):
        super().__init__(self.bitwise_or)


class Device:
    """
    Device base class
    """
    def __init__(self, device, simulate, round_avg):
        self.device = device
        self.simulate = simulate
        self.round_avg = round_avg

    def __str__(self):
        return self.__class__.__name__


class DevAI84(Device):
    """
    Implementation limits for AI84
    """
    def __init__(self, simulate, round_avg):
        assert not round_avg

        super().__init__(84, simulate, round_avg)

        self.WEIGHT_BITS = 8
        self.DATA_BITS = 8
        self.ACTIVATION_BITS = 8
        self.FULL_ACC_BITS = 8
        self.FC_ACTIVATION_BITS = 16

        self.WEIGHT_INPUTS = 64
        self.WEIGHT_DEPTH = 128

        self.MAX_AVG_POOL = 4

    def __str__(self):
        return self.__class__.__name__


class DevAI85(Device):
    """
    Implementation limits for MAX78000
    """
    def __init__(self, simulate, round_avg):
        super().__init__(85, simulate, round_avg)

        self.WEIGHT_BITS = 8
        self.DATA_BITS = 8
        self.ACTIVATION_BITS = 8
        self.FULL_ACC_BITS = 30
        self.FC_ACTIVATION_BITS = 16

        self.WEIGHT_INPUTS = 256
        self.WEIGHT_DEPTH = 768

        self.MAX_AVG_POOL = 16

    def __str__(self):
        return self.__class__.__name__


class DevAI87(DevAI85):
    """
    Implementation limits for AI87. For now, the same as MAX78000.
    """
    def __str__(self):
        return self.__class__.__name__


def set_device(
        device,
        simulate,
        round_avg,
        verbose=True,
):
    """
    Change implementation configuration to match the `device` input value and
    `simulate` bool. `round_avg` controls the average pooling rounding.
    """
    global dev  # pylint: disable=global-statement

    if verbose:
        print(f'Configuring device: {devices.partnum(device)}, simulate={simulate}.')

    if device == 84:
        dev = DevAI84(simulate, round_avg)
    elif device == 85:
        dev = DevAI85(simulate, round_avg)
    elif device == 87:
        dev = DevAI87(simulate, round_avg)
    else:
        raise ValueError(f'Unkown device {device}.')


class QuantizeONNX(nn.Module):
    """
    Post-activation integer quantization module
    Apply the custom autograd function
    """
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        factor = 2**(self.num_bits-1)
        return x.mul(factor).round().div(factor)


def initiate_qat(m, qat_policy):
    """
    Modify model `m` to start quantization aware training.
    """
    def _initiate_qat(m):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if isinstance(target_attr, QuantizationAwareModule):
                if 'shift_quantile' in qat_policy:
                    target_attr.init_module(qat_policy['weight_bits'], 8,
                                            True, qat_policy['shift_quantile'])
                else:
                    target_attr.init_module(qat_policy['weight_bits'], 8, True, 1.0)
                if 'overrides' in qat_policy:
                    if attr_str in qat_policy['overrides']:
                        weight_field = qat_policy['overrides'][attr_str]['weight_bits']
                        if 'shift_quantile' in qat_policy:
                            target_attr.init_module(weight_field, 8,
                                                    True, qat_policy['shift_quantile'])
                        else:
                            target_attr.init_module(weight_field,
                                                    8, True, 1.0)

                setattr(m, attr_str, target_attr)

    m.apply(_initiate_qat)


def update_model(m):
    """
    Update model `m` with the current parameters.
    It is used to update model functions after loading a checkpoint file.
    """
    def _update_model(m):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if isinstance(target_attr, QuantizationAwareModule):
                target_attr.set_functions()
                setattr(m, attr_str, target_attr)

    m.apply(_update_model)


def fuse_bn_layers(m):
    """
    Fuse the bn layers before the quantization aware training starts.
    """
    def _fuse_bn_layers(m):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if isinstance(target_attr, QuantizationAwareModule) \
               and target_attr.bn is not None:
                w = target_attr.op.weight.data
                b = target_attr.op.bias.data
                device = w.device

                r_mean = target_attr.bn.running_mean
                r_var = target_attr.bn.running_var
                r_std = torch.sqrt(r_var + 1e-20)
                beta = target_attr.bn.weight
                gamma = target_attr.bn.bias

                if beta is None:
                    beta = torch.ones(w.shape[0]).to(device)
                if gamma is None:
                    gamma = torch.zeros(w.shape[0]).to(device)

                beta = 0.25 * beta
                gamma = 0.25 * gamma

                w_new = w * (beta / r_std).reshape((w.shape[0],) + (1,) * (len(w.shape) - 1))
                b_new = (b - r_mean)/r_std * beta + gamma

                target_attr.op.weight.data = w_new
                target_attr.op.bias.data = b_new
                target_attr.bn = None
                setattr(m, attr_str, target_attr)

    m.apply(_fuse_bn_layers)


def onnx_export_prep(m, simplify=False):
    """
    Prepare model `m` for ONNX export. When `simplify` is True, remove several
    quantization related operators from the model graph.
    """
    def _onnx_export_prep(m):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if not simplify:
                if isinstance(target_attr, Quantize):
                    setattr(m, attr_str, QuantizeONNX(target_attr.num_bits))
            else:
                if isinstance(target_attr, (Quantize, Clamp, Round, Floor)):
                    setattr(m, attr_str, Empty())

    m.apply(_onnx_export_prep)
