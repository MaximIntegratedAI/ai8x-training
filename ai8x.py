###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Contains the limits of the AI84/AI85/AI87 implementations and custom PyTorch modules that take
the limits into account.
"""
import torch
import torch.nn as nn
from torch.autograd import Function


dev = None


class normalize:
    """
    Normalize input to either [-0.5, +0.5] or [-128, +127]
    """
    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        if self.args.act_mode_8bit:
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
        return img.sub(0.5)


class QuantizationFunction(Function):
    """
    Custom AI8X autograd function
    The forward pass divides by 2**(bits-1) (typically, 128) and rounds the result to the
    nearest integer.
    The backward pass is straight through.
    """
    @staticmethod
    def forward(ctx, x, bits=None):  # pylint: disable=arguments-differ
        if bits > 1:
            return x.add(.5).div(2**(bits-1)).add(.5).floor()
        if bits < 1:
            return x.mul(2**(1-bits)).add(.5).floor()
        return x.add(.5).floor()

    @staticmethod
    def backward(ctx, x):  # pylint: disable=arguments-differ
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x, None


class Quantize(nn.Module):
    """
    Post-activation integer quantization module
    Apply the custom autograd function
    """
    def __init__(self, num_bits=8):
        super(Quantize, self).__init__()
        self.num_bits = num_bits

    def forward(self, x):  # pylint: disable=arguments-differ
        return QuantizationFunction.apply(x, self.num_bits)


class FloorFunction(Function):
    """
    Custom AI8X autograd function
    The forward pass returns the integer floor.
    The backward pass is straight through.
    """
    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ
        return x.floor()

    @staticmethod
    def backward(ctx, x):  # pylint: disable=arguments-differ
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x


class Floor(nn.Module):
    """
    Post-pooling integer quantization module
    Apply the custom autograd function
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        return FloorFunction.apply(x)


class RoundFunction(Function):
    """
    Custom AI8X autograd function
    The forward pass returns the integer rounded.
    The backward pass is straight through.
    """
    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ
        return x.round()

    @staticmethod
    def backward(ctx, x):  # pylint: disable=arguments-differ
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x


class Round(nn.Module):
    """
    Post-pooling integer quantization module
    Apply the custom autograd function
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        return RoundFunction.apply(x)


class Clamp(nn.Module):
    """
    Post-Activation Clamping Module
    Clamp the output to the given range (typically, [-128, +127])
    """
    def __init__(self, min_val=None, max_val=None):
        super(Clamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):  # pylint: disable=arguments-differ
        return x.clamp(min=self.min_val, max=self.max_val)


class Abs(nn.Module):
    """
    Return abs(x)
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        return torch.abs_(x)  # abs_() is the in-place version


class Empty(nn.Module):
    """
    Do nothing
    """
    def forward(self, x):  # pylint: disable=arguments-differ
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


class Conv2d(nn.Module):
    """
    AI8X - 2D pooling ('Avg', 'Max' or None) optionally followed by
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
            output_shift=0,
            wide=False,
    ):
        super(Conv2d, self).__init__()

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

            assert stride == 1
        else:
            assert 0 < stride <= 3

        assert 0 <= padding <= 2

        if pooling == 'Max':
            self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride, padding=0)
        elif pooling == 'Avg':
            self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride, padding=0)
        else:
            self.pool = None

        if kernel_size is not None:
            if isinstance(kernel_size, tuple):
                assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
                kernel_size = kernel_size[0]

            assert kernel_size == 3 or dev.device != 84 and kernel_size == 1

            if op == 'Conv2d':
                self.conv2d = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, bias=bias)
            elif op == 'ConvTranspose2d':
                assert dev.device != 84
                self.conv2d = nn.ConvTranspose2d(in_channels, out_channels,
                                                 kernel_size=kernel_size, stride=stride,
                                                 padding=padding, bias=bias)
            else:
                raise ValueError('Unsupported operation')
        else:
            self.conv2d = None

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS + output_shift if not wide else 1)
            bits = dev.ACTIVATION_BITS if not wide else dev.FULL_ACC_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
            if pooling == 'Avg':
                self.quantize_pool = Round() if dev.round_avg else Floor()
            else:  # Max, None
                self.quantize_pool = Empty()
        else:
            self.quantize = Empty()
            self.quantize_pool = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        if pooling == 'Avg':
            self.clamp_pool = self.clamp
        else:  # Max, None
            self.clamp_pool = Empty()

        self.activate = get_activation(activation)

    def forward(self, x):  # pylint: disable=arguments-differ
        if self.pool is not None:
            x = self.clamp_pool(self.quantize_pool(self.pool(x)))
        if self.conv2d is not None:
            x = self.conv2d(x)
            x = self.clamp(self.quantize(self.activate(x)))
        return x


class FusedMaxPoolConv2d(Conv2d):
    """
    AI8X - Fused 2D Max Pool, 2D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super(FusedMaxPoolConv2d, self).__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConv2dReLU(FusedMaxPoolConv2d):
    """
    AI8X - Fused 2D Max Pool, 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedMaxPoolConv2dReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConv2dAbs(FusedMaxPoolConv2d):
    """
    AI8X - Fused 2D Max Pool, 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedMaxPoolConv2dAbs, self).__init__(*args, activation='Abs', **kwargs)


class MaxPool2d(FusedMaxPoolConv2d):
    """
    AI8X - 2D Max Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super(MaxPool2d, self).__init__(0, 0, None,
                                        pool_size=kernel_size, pool_stride=stride,
                                        activation=None, **kwargs)


class FusedAvgPoolConv2d(Conv2d):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super(FusedAvgPoolConv2d, self).__init__(*args, pooling='Avg', **kwargs)


class FusedAvgPoolConv2dReLU(FusedAvgPoolConv2d):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedAvgPoolConv2dReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedAvgPoolConv2dAbs(FusedAvgPoolConv2d):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedAvgPoolConv2dAbs, self).__init__(*args, activation='Abs', **kwargs)


class AvgPool2d(FusedAvgPoolConv2d):
    """
    AI8X - 2D Avg Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super(AvgPool2d, self).__init__(0, 0, None,
                                        pool_size=kernel_size, pool_stride=stride,
                                        activation=None, **kwargs)


class FusedConv2dReLU(Conv2d):
    """
    AI8X - Fused 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedConv2dReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedConv2dAbs(Conv2d):
    """
    AI8X - Fused 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedConv2dAbs, self).__init__(*args, activation='Abs', **kwargs)


class ConvTranspose2d(Conv2d):
    """
    AI8X - 2D pooling ('Avg', 'Max' or None) optionally followed by
    transposed 2D convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super(ConvTranspose2d, self).__init__(*args, op='ConvTranspose2d', **kwargs)


class FusedMaxPoolConvTranspose2d(ConvTranspose2d):
    """
    AI8X - Fused 2D Max Pool, Transposed 2D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super(FusedMaxPoolConvTranspose2d, self).__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConvTranspose2dReLU(FusedMaxPoolConvTranspose2d):
    """
    AI8X - Fused 2D Max Pool, Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedMaxPoolConvTranspose2dReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConvTranspose2dAbs(FusedMaxPoolConvTranspose2d):
    """
    AI8X - Fused 2D Max Pool, Transposed 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedMaxPoolConvTranspose2dAbs, self).__init__(*args, activation='Abs', **kwargs)


class FusedAvgPoolConvTranspose2d(ConvTranspose2d):
    """
    AI8X - Fused 2D Avg Pool, Transposed 2D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super(FusedAvgPoolConvTranspose2d, self).__init__(*args, pooling='Avg', **kwargs)


class FusedAvgPoolConvTranspose2dReLU(FusedAvgPoolConvTranspose2d):
    """
    AI8X - Fused 2D Avg Pool, Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedAvgPoolConvTranspose2dReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedAvgPoolConvTranspose2dAbs(FusedAvgPoolConvTranspose2d):
    """
    AI8X - Fused 2D Avg Pool, Transposed 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedAvgPoolConvTranspose2dAbs, self).__init__(*args, activation='Abs', **kwargs)


class FusedConvTranspose2dReLU(ConvTranspose2d):
    """
    AI8X - Fused Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedConvTranspose2dReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedConvTranspose2dAbs(ConvTranspose2d):
    """
    AI8X - Fused Transposed 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedConvTranspose2dAbs, self).__init__(*args, activation='Abs', **kwargs)


class FusedSoftwareLinearReLU(nn.Module):
    """
    AI84 - Fused Linear and ReLU using Software
    """
    def __init__(self, in_features, out_features, bias=None, relu=True):
        super(FusedSoftwareLinearReLU, self).__init__()

        if dev.device != 84:
            print('WARNING: SoftwareLinear should be used on AI84 only')

        self.linear = nn.Linear(in_features, out_features, bias)

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS)
            bits = dev.FC_ACTIVATION_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        if relu:
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = Empty()

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.linear(x)
        x = self.clamp(self.quantize(self.activate(x)))
        return x


class SoftwareLinear(FusedSoftwareLinearReLU):
    """
    AI84 - Linear using Software
    """
    def __init__(self, in_features, out_features, **kwargs):
        super(SoftwareLinear, self).__init__(in_features, out_features, relu=False, **kwargs)


class Linear(nn.Module):
    """
    AI85+ - Fused Linear and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, in_features, out_features, bias=None,
                 activation=None, output_shift=0, wide=False):
        super(Linear, self).__init__()

        assert dev.device != 84
        assert in_features <= 1024
        assert out_features <= 1024
        self.linear = nn.Linear(in_features, out_features, bias)

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS + output_shift if not wide else 1)
            bits = dev.ACTIVATION_BITS if not wide else dev.FULL_ACC_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        self.activate = get_activation(activation)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.linear(x)
        x = self.clamp(self.quantize(self.activate(x)))
        return x


class FusedLinearReLU(Linear):
    """
    AI85+ - Fused Linear and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedLinearReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedLinearAbs(Linear):
    """
    AI85+ - Fused Linear and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedLinearAbs, self).__init__(*args, activation='Abs', **kwargs)


class Conv1d(nn.Module):
    """
    AI8X - Fused 1D Pool ('Avg', 'Max' or None) followed by
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
            output_shift=0,
            wide=False,
    ):
        super(Conv1d, self).__init__()

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
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride, padding=0)
        elif pooling == 'Avg':
            self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride, padding=0)
        else:
            self.pool = None

        if kernel_size is not None:
            assert dev.device != 84 or padding in [0, 3, 6]
            assert dev.device == 84 or padding in [0, 1, 2]
            assert dev.device != 84 or kernel_size == 9
            assert dev.device == 84 or kernel_size in [1, 2, 3, 4, 5, 6, 7, 8, 9]

            self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                    padding=padding, bias=bias)
        else:
            self.conv1d = None

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS + output_shift if not wide else 1)
            bits = dev.ACTIVATION_BITS if not wide else dev.FULL_ACC_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
            if pooling == 'Avg':
                self.quantize_pool = Round() if dev.round_avg else Floor()
            else:  # Max, None
                self.quantize_pool = Empty()
        else:
            self.quantize = Empty()
            self.quantize_pool = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        if pooling == 'Avg':
            self.clamp_pool = self.clamp
        else:  # Max, None
            self.clamp_pool = Empty()

        self.activate = get_activation(activation)

    def forward(self, x):  # pylint: disable=arguments-differ
        if self.pool is not None:
            x = self.clamp_pool(self.quantize_pool(self.pool(x)))
        if self.conv1d is not None:
            x = self.conv1d(x)
            x = self.clamp(self.quantize(self.activate(x)))
        return x


class FusedMaxPoolConv1d(Conv1d):
    """
    AI8X - Fused 1D Max Pool, 1D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super(FusedMaxPoolConv1d, self).__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConv1dReLU(FusedMaxPoolConv1d):
    """
    AI8X - Fused 1D Max Pool, 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedMaxPoolConv1dReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConv1dAbs(FusedMaxPoolConv1d):
    """
    AI8X - Fused 1D Max Pool, 1D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedMaxPoolConv1dAbs, self).__init__(*args, activation='Abs', **kwargs)


class MaxPool1d(FusedMaxPoolConv1d):
    """
    AI8X - 1D Max Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super(MaxPool1d, self).__init__(0, 0, None,
                                        pool_size=kernel_size, pool_stride=stride,
                                        activation=None, **kwargs)


class FusedAvgPoolConv1d(Conv1d):
    """
    AI8X - Fused 1D Avg Pool, 1D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super(FusedAvgPoolConv1d, self).__init__(*args, pooling='Avg', **kwargs)


class FusedAvgPoolConv1dReLU(FusedAvgPoolConv1d):
    """
    AI8X - Fused 1D Avg Pool, 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedAvgPoolConv1dReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedAvgPoolConv1dAbs(FusedAvgPoolConv1d):
    """
    AI8X - Fused 1D Avg Pool, 1D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedAvgPoolConv1dAbs, self).__init__(*args, activation='Abs', **kwargs)


class AvgPool1d(FusedAvgPoolConv1d):
    """
    AI8X - 1D Avg Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super(AvgPool1d, self).__init__(0, 0, None,
                                        pool_size=kernel_size, pool_stride=stride,
                                        activation=None, **kwargs)


class FusedConv1dReLU(Conv1d):
    """
    AI8X - Fused 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super(FusedConv1dReLU, self).__init__(*args, activation='ReLU', **kwargs)


class FusedConv1dAbs(Conv1d):
    """
    AI8X - Fused 1D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super(FusedConv1dAbs, self).__init__(*args, activation='Abs', **kwargs)


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
        super(DevAI84, self).__init__(84, simulate, round_avg)

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
    Implementation limits for AI85
    """
    def __init__(self, simulate, round_avg):
        super(DevAI85, self).__init__(85, simulate, round_avg)

        self.WEIGHT_BITS = 8
        self.DATA_BITS = 8
        self.ACTIVATION_BITS = 8
        self.FULL_ACC_BITS = 32
        self.FC_ACTIVATION_BITS = 16

        self.WEIGHT_INPUTS = 256
        self.WEIGHT_DEPTH = 768

        self.MAX_AVG_POOL = 16

    def __str__(self):
        return self.__class__.__name__


class DevAI87(DevAI85):
    """
    Implementation limits for AI87. For now, the same as AI85.
    """
    def __str__(self):
        return self.__class__.__name__


def set_device(
        device,
        simulate,
        round_avg,
):
    """
    Change implementation configuration to match the AI84 or AI85, depending on the `device`
    integer input value and `simulate` bool. `round_avg` (AI85+) controls the average pooling
    rounding.
    """
    global dev  # pylint: disable=global-statement

    print(f'Configuring device: AI{device}, simulate={simulate}.')

    if device == 84:
        dev = DevAI84(simulate, round_avg)
    elif device == 85:
        dev = DevAI85(simulate, round_avg)
    elif device == 87:
        dev = DevAI87(simulate, round_avg)
    else:
        raise ValueError(f'Unkown device {device}.')
