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


class FusedMaxPoolConv2d(nn.Module):
    """
    AI8X - Fused 2D Pool ('Max', 'Avg', None), 2D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            pool_size=2,
            pool_stride=2,
            stride=1,
            padding=0,
            bias=True,
            activation=None,
            output_shift=0,
            wide=False,
    ):
        super(FusedMaxPoolConv2d, self).__init__()

        if pool_stride is None:
            pool_stride = pool_size
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2
            kernel_size = kernel_size[0]

        if isinstance(pool_size, int):
            assert dev.device != 84 or pool_size & 1 == 0
            assert pool_size <= 16
        elif isinstance(pool_size, tuple):
            assert len(pool_size) == 2
            assert dev.device != 84 or pool_size[0] & 1 == 0
            assert pool_size[0] <= 16
            assert dev.device != 84 or pool_size[1] & 1 == 0
            assert pool_size[1] <= 16
        else:
            raise ValueError('pool_size must be int or tuple')

        if isinstance(pool_stride, int):
            assert pool_stride > 0
            assert 0 < pool_stride <= 16
        elif isinstance(pool_stride, tuple):
            assert len(pool_stride) == 2
            assert dev.device != 84 or pool_stride[0] == pool_stride[1]
            assert pool_stride[0] > 0
            assert pool_stride[0] <= 16
            assert pool_stride[1] > 0
            assert pool_stride[1] <= 16
        else:
            raise ValueError('pool_stride must be int or tuple')

        assert 0 <= padding <= 2
        assert stride == 1

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride, padding=0)
        if kernel_size is not None:
            assert kernel_size == 3 or dev.device != 84 and kernel_size == 1

            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        else:
            self.conv2d = None

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS + output_shift if not wide else 1)
            bits = dev.ACTIVATION_BITS if not wide else dev.FULL_ACC_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        self.activate = get_activation(activation)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.pool(x)
        if self.conv2d is not None:
            x = self.conv2d(x)
            x = self.clamp(self.quantize(self.activate(x)))
        return x


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


class FusedAvgPoolConv2d(nn.Module):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, pool_stride=2,
                 stride=1, padding=0, bias=True, activation=None, output_shift=0, wide=False):
        super(FusedAvgPoolConv2d, self).__init__()

        if pool_stride is None:
            pool_stride = pool_size
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2
            kernel_size = kernel_size[0]

        if isinstance(pool_size, int):
            assert dev.device != 84 or pool_size & 1 == 0
            assert pool_size <= 16 and (dev.device != 84 or pool_size <= 4)
        elif isinstance(pool_size, tuple):
            assert len(pool_size) == 2
            assert dev.device != 84 or pool_size[0] & 1 == 0
            assert pool_size[0] <= 16 and (dev.device != 84 or pool_size[0] <= 4)
            assert dev.device != 84 or pool_size[1] & 1 == 0
            assert pool_size[1] <= 16 and (dev.device != 84 or pool_size[0] <= 4)
        else:
            raise ValueError('pool_size must be int or tuple')

        if isinstance(pool_stride, int):
            assert pool_stride > 0
            assert pool_stride <= 16 and (dev.device != 84 or pool_stride <= 4)
        elif isinstance(pool_stride, tuple):
            assert len(pool_stride) == 2
            assert dev.device != 84 or pool_stride[0] == pool_stride[1]
            assert pool_stride[0] > 0
            assert pool_stride[0] <= 16 and (dev.device != 84 or pool_stride[0] <= 4)
            assert pool_stride[1] > 0
            assert pool_stride[1] <= 16 and (dev.device != 84 or pool_stride[1] <= 4)
        else:
            raise ValueError('pool_stride must be int or tuple')

        assert 0 <= padding <= 2
        assert stride == 1

        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride, padding=0)
        if kernel_size is not None:
            assert kernel_size == 3 or dev.device != 84 and kernel_size == 1

            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        else:
            self.conv2d = None

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS + output_shift if not wide else 1)
            if dev.round_avg:
                self.quantize_pool = Round()
            else:
                self.quantize_pool = Floor()
            bits = dev.ACTIVATION_BITS if not wide else dev.FULL_ACC_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize = Empty()
            self.quantize_pool = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        self.activate = get_activation(activation)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.clamp(self.quantize_pool(self.pool(x)))
        if self.conv2d:
            x = self.conv2d(x)
            x = self.clamp(self.quantize(self.activate(x)))
        return x


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


class Conv2d(nn.Module):
    """
    AI8X - Fused 2D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 activation=None, output_shift=0, wide=False):
        super(Conv2d, self).__init__()

        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2
            kernel_size = kernel_size[0]

        assert 0 < stride <= 3
        assert 0 <= padding <= 2
        assert kernel_size == 3 or dev.device != 84 and kernel_size == 1

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, bias=bias)

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS + output_shift if not wide else 1)
            bits = dev.ACTIVATION_BITS if not wide else dev.FULL_ACC_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        self.activate = get_activation(activation)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv2d(x)
        x = self.clamp(self.quantize(self.activate(x)))
        return x


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
    AI8X - Fused 1D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=3, padding=0, bias=True,
                 activation=None, output_shift=0, wide=False):
        super(Conv1d, self).__init__()

        assert dev.device != 84 or stride == 3
        assert dev.device == 84 or stride == 1
        assert dev.device != 84 or padding in [0, 3, 6]
        assert dev.device == 84 or padding in [0, 1, 2]
        assert dev.device != 84 or kernel_size == 9
        assert dev.device == 84 or kernel_size in [1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, bias=bias)

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS + output_shift if not wide else 1)
            bits = dev.ACTIVATION_BITS if not wide else dev.FULL_ACC_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        self.activate = get_activation(activation)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv1d(x)
        x = self.clamp(self.quantize(self.activate(x)))
        return x


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
