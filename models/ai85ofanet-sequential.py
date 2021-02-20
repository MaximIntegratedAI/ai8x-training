###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Sequential Once For All network for AI85.
"""
import random

import torch
import torch.nn as nn

import ai8x
from ai8x_ofa import FusedConv2dReLU, FusedConv2dBNReLU, FusedMaxPoolConv2dReLU, \
                     FusedMaxPoolConv2dBNReLU, FusedConv1dReLU, FusedConv1dBNReLU, \
                     FusedMaxPoolConv1dReLU, FusedMaxPoolConv1dBNReLU


class OnceForAllSequentialUnit(nn.Module):
    """
    Base unit for sequential models used for Once For All NAS
    """
    def __init__(self, depth, kernel_size, width, init_width, bias, pooling, bn,
                 layer_op_list, **kwargs):
        super().__init__()
        self.depth = depth
        padding = kernel_size // 2
        self.layers = nn.ModuleList([])
        for i in range(depth):
            in_channels = width if i != 0 else init_width

            if i == 0 and pooling:
                if bn:
                    layer_op = layer_op_list[0]
                else:
                    layer_op = layer_op_list[1]

                self.layers.append(layer_op(in_channels, width, kernel_size, pool_size=2,
                                            pool_stride=2, stride=1, padding=padding, bias=bias,
                                            **kwargs))
            else:
                if bn:
                    layer_op = layer_op_list[2]
                else:
                    layer_op = layer_op_list[3]

                self.layers.append(layer_op(in_channels, width, kernel_size, stride=1,
                                            padding=padding, bias=bias, **kwargs))

    def get_max_elastic_depth_level(self):
        """Returns max depth level to be used OFA elastic depth search"""
        return len(self.layers) - 1

    def sample_subnet_depth(self, level):
        """OFA Elastic depth search strategy"""
        with torch.no_grad():
            max_depth = len(self.layers)
            min_depth = max_depth - level if 0 <= level < max_depth  else 1
            self.depth = random.randint(min_depth, max_depth)

    def reset_depth_sampling(self):
        """Resets depth to maximum depth"""
        self.depth = len(self.layers)

    def forward(self, x):
        """Forward prop"""
        for l in range(self.depth):
            x = self.layers[l](x)
        return x


class OnceForAll2DSequentialUnit(OnceForAllSequentialUnit):
    """
    2D sequential model used for Once For All NAS
    """
    def __init__(self, depth, kernel_size, width, init_width, bias, pooling=True, bn=False,
                 **kwargs):
        layer_op_list = [FusedMaxPoolConv2dBNReLU, FusedMaxPoolConv2dReLU, FusedConv2dBNReLU,
                         FusedConv2dReLU]

        super().__init__(depth, kernel_size, width, init_width, bias, pooling, bn,
                         layer_op_list, **kwargs)


class OnceForAll1DSequentialUnit(OnceForAllSequentialUnit):
    """
    1D sequential model used for Once For All NAS
    """
    def __init__(self, depth, kernel_size, width, init_width, bias, pooling=True, bn=False,
                 **kwargs):
        layer_op_list = [FusedMaxPoolConv1dBNReLU, FusedMaxPoolConv1dReLU, FusedConv1dBNReLU,
                         FusedConv1dReLU]

        super().__init__(depth, kernel_size, width, init_width, bias, pooling, bn, layer_op_list,
                         **kwargs)


class OnceForAllSequentialModel(nn.Module):
    """
    Sequential Once For All Model
    """
    def __init__(self, num_classes, num_channels, dimensions, bias, n_units, depth_list,
                 width_list, kernel_list, bn, unit, **kwargs):
        super().__init__()

        self.units = nn.ModuleList([])
        dim1 = dimensions[0]
        dim2 = dimensions[1] if len(dimensions) == 2 else 1

        last_width = num_channels
        for i in range(n_units):
            if i == 0:
                pooling = False
            else:
                pooling = True
                dim1 = dim1 // 2
                dim2 = (dim2 // 2) if len(dimensions) == 2 else 1

            self.units.append(unit(depth_list[i], kernel_list[i], width_list[i], last_width, bias,
                                   pooling, bn, **kwargs))
            last_width = width_list[i]

        self.classifier = ai8x.Linear(dim1*dim2*last_width, num_classes, bias=bias, wide=True,
                                      **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        for unit in self.units:
            x = unit(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_max_elastic_width_level(self):
        """Returns max width level to be used OFA elastic depth search"""
        max_depth = 0
        for unit in self.units:
            max_depth = max(unit.depth, max_depth)

        return max_depth-1

    def get_max_elastic_depth_level(self):
        """Returns max depth level to be used OFA elastic depth search"""
        max_depth_level = 0
        for unit in self.units:
            max_depth_level = max(unit.get_max_elastic_depth_level(), max_depth_level)

        return max_depth_level

    def get_max_elastic_kernel_level(self):
        """Returns max kernel level to be used OFA elastic depth search"""
        max_kernel_level = 0
        for unit in self.units:
            for layer in unit.layers:
                max_kernel_level = max(layer.op.kernel_size[0]//2, max_kernel_level)

        return max_kernel_level

    def sample_subnet_width(self, level=0):
        """OFA Elastic width search strategy"""
        max_unit_ind = len(self.units) - 1
        for u_ind, unit in enumerate(self.units):
            max_layer_ind = unit.depth - 1
            for l_ind in range(unit.depth):
                layer = unit.layers[l_ind]
                if not(u_ind == 0 and l_ind == 0):
                    layer.set_channels(in_channels=last_out_ch)

                if (u_ind == max_unit_ind) and (l_ind == max_layer_ind):
                    layer.set_channels(out_channels=self.units[-1].layers[-1].out_channels)
                else:
                    pos_width_list = [layer.op.out_channels]
                    for l in range(level):
                        pos_width_list.append(int((1.0 - (l+1)*0.25) * layer.op.out_channels))

                    random_width = random.choice(pos_width_list)
                    layer.set_channels(out_channels=random_width)
                    last_out_ch = layer.out_channels

        self.sort_channels()

    def reset_width_sampling(self):
        """Resets widths to maximum widths"""
        for unit in self.units:
            for layer in unit.layers:
                layer.set_channels(in_channels=layer.op.in_channels,
                                   out_channels=layer.op.out_channels)

    def sort_channels(self):
        """Sorts channels wrt output channel kernels importance"""
        with torch.no_grad():
            max_unit_ind = len(self.units) - 1
            for u_ind, unit in enumerate(self.units):
                max_layer_ind = unit.depth - 1
                for l_ind in range(unit.depth):
                    layer = unit.layers[l_ind]
                    if (u_ind == max_unit_ind) and (l_ind == max_layer_ind):
                        layer.reset_out_ch_order()
                    else:
                        reduce_dim = (1, 2, 3) if layer.op.weight.dim() == 4 else (1, 2)
                        importance = torch.sum(torch.abs(layer.op.weight.data), dim=reduce_dim)
                        _, inds = torch.sort(importance, descending=True)
                        layer.set_out_ch_order(inds, reset_order=False)

                        if l_ind < max_layer_ind:
                            next_layer = unit.layers[l_ind+1]
                        else:
                            next_layer = self.units[u_ind+1].layers[0]

                        next_layer.set_in_ch_order(layer.out_ch_order, reset_order=True)


class OnceForAll2DSequentialModel(OnceForAllSequentialModel):
    """
    2D Sequential Once For All Model
    """
    def __init__(self, num_classes, num_channels, dimensions, bias, n_units, depth_list,
                 width_list, kernel_list, bn, **kwargs):
        super().__init__(num_classes, num_channels, dimensions, bias, n_units, depth_list,
                         width_list, kernel_list, bn, OnceForAll2DSequentialUnit, **kwargs)

class OnceForAll1DSequentialModel(OnceForAllSequentialModel):
    """
    1D Sequential Once For All Model
    """
    def __init__(self, num_classes, num_channels, dimensions, bias, n_units, depth_list,
                 width_list, kernel_list, bn=False, **kwargs):
        super().__init__(num_classes, num_channels, dimensions, bias, n_units, depth_list,
                         width_list, kernel_list, bn, OnceForAll1DSequentialUnit, **kwargs)


def ai85ofanet_cifar100(pretrained=False, **kwargs):
    """
    Constructs a MofaNET v1 model.
    """
    assert not pretrained
    n_units = 5
    depth_list = [4, 3, 3, 3, 2]
    width_list = [64, 64, 128, 128, 128]
    kernel_list = [3, 3, 3, 3, 3]
    bn = True

    return OnceForAll2DSequentialModel(n_units=n_units, depth_list=depth_list, bn=bn,
                                       width_list=width_list, kernel_list=kernel_list, **kwargs)


def ai85ofanet_kws20(pretrained=False, **kwargs):
    """
    Constructs a MofaNET v1 model.
    """
    assert not pretrained
    n_units = 6
    depth_list = [3, 3, 2, 2, 2, 2]
    width_list = [128, 128, 128, 128, 128, 128]
    kernel_list = [5, 5, 5, 5, 5, 5]
    bn = True

    return OnceForAll1DSequentialModel(n_units=n_units, depth_list=depth_list, bn=bn,
                                       width_list=width_list, kernel_list=kernel_list, **kwargs)


models = [
    {
        'name': 'ai85ofanet_cifar100',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85ofanet_kws20',
        'min_input': 1,
        'dim': 1,
    }
]
