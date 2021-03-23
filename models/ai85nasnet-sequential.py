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
import copy
import random

import torch
import torch.nn as nn

import ai8x
from ai8x_nas import (FusedConv1dBNReLU, FusedConv1dReLU, FusedConv2dBNReLU, FusedConv2dReLU,
                      FusedMaxPoolConv1dBNReLU, FusedMaxPoolConv1dReLU, FusedMaxPoolConv2dBNReLU,
                      FusedMaxPoolConv2dReLU)


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

    def get_max_elastic_kernel_level(self):
        """Returns max depth level to be used OFA elastic kernel search"""
        max_kernel_level = 0
        for layer in self.layers:
            max_kernel_level = max(layer.op.kernel_size[0]//2, max_kernel_level)

        return max_kernel_level

    def get_max_elastic_depth_level(self):
        """Returns max depth level to be used OFA elastic depth search"""
        return len(self.layers) - 1

    def get_max_elastic_width_level(self):  # pylint: disable=no-self-use
        """Returns max width level to be used OFA elastic width search"""
        return 0

    def sample_subnet_kernel(self, level):
        """OFA Elastic kernel search strategy"""
        with torch.no_grad():
            for layer in self.layers:
                layer.sample_subnet_kernel(level)

    def reset_kernel_sampling(self):
        """Resets kernel to maximum kernel"""
        with torch.no_grad():
            for layer in self.layers:
                layer.reset_kernel_sampling()

    def sample_subnet_depth(self, level):
        """OFA Elastic depth search strategy"""
        with torch.no_grad():
            max_depth = len(self.layers)
            min_depth = max_depth - level if 0 <= level < max_depth else 1
            self.depth = random.randint(min_depth, max_depth)

    def reset_depth_sampling(self):
        """Resets depth to maximum depth"""
        with torch.no_grad():
            self.depth = len(self.layers)

    def sample_subnet_width(self, level):
        """OFA Elastic width search strategy"""
        pass  # pylint: disable=unnecessary-pass

    def reset_width_sampling(self):
        """Resets width sampling"""
        pass  # pylint: disable=unnecessary-pass

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        for l_idx in range(self.depth):
            x = self.layers[l_idx](x)
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

        self.num_classes = num_classes
        self.num_channels = num_channels
        self.dimensions = dimensions
        self.bias = bias
        self.n_units = n_units
        self.depth_list = depth_list
        self.width_list = width_list
        self.kernel_list = kernel_list
        self.bn = bn
        self.unit = unit

        self.units = nn.ModuleList([])

        inp_2d = True
        if len(dimensions) == 1 or dimensions[1] == 1:
            inp_2d = False

        dim1 = dimensions[0]
        dim2 = dimensions[1] if inp_2d else 1

        last_width = num_channels
        for i in range(n_units):
            if i == 0:
                pooling = False
            else:
                pooling = True
                dim1 = dim1 // 2
                dim2 = (dim2 // 2) if inp_2d else 1

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

    def sample_subnet_kernel(self, level=0):
        """OFA Elastic kernel search strategy"""
        with torch.no_grad():
            for unit in self.units:
                for layer in unit.layers:
                    layer.sample_subnet_kernel(level)

    def reset_kernel_sampling(self):
        """Resets kernel to maximum widths"""
        with torch.no_grad():
            for unit in self.units:
                for layer in unit.layers:
                    layer.reset_kernel_sampling()

    def sample_subnet_depth(self, level=0):
        """OFA Elastic depth search strategy"""
        with torch.no_grad():
            for unit in self.units:
                unit.sample_subnet_depth(level)

    def reset_depth_sampling(self):
        """Resets depth to maximum widths"""
        with torch.no_grad():
            for unit in self.units:
                unit.reset_depth_sampling()

    def sample_subnet_width(self, level=0):
        """OFA Elastic width search strategy"""
        assert level < 4, 'Elastic width level must be smaller than 4!!'

        with torch.no_grad():
            max_unit_ind = len(self.units) - 1
            last_out_ch = self.num_channels
            for u_ind, unit in enumerate(self.units):
                max_layer_ind = unit.depth - 1
                for l_ind in range(unit.depth):
                    layer = unit.layers[l_ind]
                    layer.set_channels(in_channels=last_out_ch)

                    if (u_ind == max_unit_ind) and (l_ind == max_layer_ind):
                        layer.set_channels(out_channels=self.units[-1].layers[-1].out_channels)
                    else:
                        pos_width_list = []
                        for lev in range(level+1):
                            pos_width_list.append(int((1.0 - lev*0.25) * layer.op.out_channels))

                        random_width = random.choice(pos_width_list)
                        layer.set_channels(out_channels=random_width)
                        last_out_ch = layer.out_channels

        self.sort_channels()

    def reset_width_sampling(self):
        """Resets widths to maximum widths"""
        with torch.no_grad():
            for unit in self.units:
                for layer in unit.layers:
                    layer.set_channels(in_channels=layer.op.in_channels,
                                       out_channels=layer.op.out_channels)
        self.sort_channels()

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

    def get_base_arch(self):
        """Returns architecture of the full model"""
        arch = {'num_classes': self.num_classes, 'num_channels': self.num_channels,
                'dimensions': self.dimensions, 'bias': self.bias, 'n_units': self.n_units,
                'bn': self.bn, 'unit': self.unit, 'depth_list': [], 'width_list': [],
                'kernel_list': []}

        for unit in self.units:
            arch['depth_list'].append(len(unit.layers))
            width_list = []
            kernel_list = []
            for layer in unit.layers:
                width_list.append(layer.op.weight.shape[0])
                kernel_list.append(layer.op.weight.shape[2])
            arch['width_list'].append(width_list)
            arch['kernel_list'].append(kernel_list)

        return arch

    def get_subnet_arch(self):
        """Returns architecture of the sampled model"""
        arch = {'num_classes': self.num_classes, 'num_channels': self.num_channels,
                'dimensions': self.dimensions, 'bias': self.bias, 'n_units': self.n_units,
                'bn': self.bn, 'unit': self.unit, 'depth_list': [], 'width_list': [],
                'kernel_list': []}

        for unit in self.units:
            arch['depth_list'].append(unit.depth)
            width_list = []
            kernel_list = []
            for l_ind in range(unit.depth):
                width_list.append(unit.layers[l_ind].out_channels)
                kernel_list.append(unit.layers[l_ind].kernel_size)
            arch['width_list'].append(width_list)
            arch['kernel_list'].append(kernel_list)

        return arch

    def set_subnet_arch(self, arch, sort_channels=False):
        """Sets given architecture as the sampled subnet"""
        assert arch['num_classes'] == self.num_classes
        assert arch['num_channels'] == self.num_channels
        assert arch['dimensions'] == self.dimensions
        assert arch['bias'] == self.bias
        assert arch['n_units'] == self.n_units
        assert arch['bn'] == self.bn
        assert arch['unit'] == self.unit

        for u_ind, unit in enumerate(self.units):
            unit.depth = arch['depth_list'][u_ind]
            for l_ind in range(unit.depth):
                unit.layers[l_ind].out_channels = arch['width_list'][u_ind][l_ind]
                unit.layers[l_ind].kernel_size = arch['kernel_list'][u_ind][l_ind]
                if l_ind == (unit.depth-1):
                    if u_ind != (self.n_units-1):
                        self.units[u_ind+1].layers[0].in_channels = \
                            arch['width_list'][u_ind][l_ind]
                else:
                    self.units[u_ind].layers[l_ind+1].in_channels = \
                            arch['width_list'][u_ind][l_ind]

        if sort_channels:
            self.sort_channels()

    def reset_arch(self, sort_channels=False):
        """Resets architecture to the full model"""
        for unit in self.units:
            unit.depth = len(unit.layers)
            for layer in unit.layers:
                layer.out_channels = layer.op.weight.shape[0]
                layer.in_channels = layer.op.weight.shape[1]
                layer.kernel_size = layer.max_kernel_size.detach().cpu().item()

        if sort_channels:
            self.sort_channels()

    @staticmethod
    def get_num_weights(model_arch):
        """Returns number of weights in the given arch"""
        num_params = 0
        dim1 = model_arch['dimensions'][0]
        dim2 = model_arch['dimensions'][1] if len(model_arch['dimensions']) == 2 else 1
        for u_ind, depth in enumerate(model_arch['depth_list']):
            if u_ind != 0:
                dim1 = dim1 // 2
                dim2 = (dim2 // 2) if len(model_arch['dimensions']) == 2 else 1
            for l_ind in range(depth):
                if l_ind != 0:
                    prev_layer_width = model_arch['width_list'][u_ind][l_ind-1]
                else:
                    if u_ind == 0:
                        prev_layer_width = model_arch['num_channels']
                    else:
                        prev_layer_width = model_arch['width_list'][u_ind-1][-1]

                num_layer_params = prev_layer_width * model_arch['width_list'][u_ind][l_ind] * \
                    model_arch['kernel_list'][u_ind][l_ind]
                if model_arch['unit'] == OnceForAll2DSequentialUnit:
                    num_layer_params *= model_arch['kernel_list'][u_ind][l_ind]

                num_params += num_layer_params

        num_linear_params = dim1*dim2*model_arch['width_list'][-1][-1]*model_arch['num_classes']

        return num_params+num_linear_params

    @staticmethod
    def mutate(model_arch, base_arch, prob_mutation, mutate_kernel=True, mutate_depth=True,
               mutate_width=True):
        """Mutates given architecture"""
        new_model_arch = copy.deepcopy(model_arch)

        depth_list = new_model_arch['depth_list']
        width_list = new_model_arch['width_list']
        kernel_list = new_model_arch['kernel_list']

        # mutate model depth
        if mutate_depth:
            for unit_idx in range(new_model_arch['n_units']):
                depth = depth_list[unit_idx]
                if random.random() < prob_mutation:
                    min_depth = 1
                    max_depth = base_arch['depth_list'][unit_idx]
                    new_depth = random.randint(min_depth, max_depth)
                    if new_depth <= depth:
                        width_list[unit_idx] = width_list[unit_idx][:new_depth]
                        kernel_list[unit_idx] = kernel_list[unit_idx][:new_depth]
                    else:
                        for i in range(new_depth - depth):
                            max_kernel = base_arch['kernel_list'][unit_idx][depth + i]
                            kernel_opts = list(range(1, max_kernel+1, 2))
                            kernel_list[unit_idx].append(random.choice(kernel_opts))

                            max_width = base_arch['width_list'][unit_idx][depth + i]
                            if mutate_width:
                                width_opts = []
                                for lev in range(4):
                                    width_opts.append(int((1.0 - lev*0.25) * max_width))
                                width_list[unit_idx].append(random.choice(width_opts))
                            else:
                                width_list[unit_idx].append(max_width)
                    depth_list[unit_idx] = new_depth

        # mutate layer parameters
        for unit_idx, _ in enumerate(width_list):
            for layer_idx, _ in enumerate(width_list[unit_idx]):
                if random.random() < prob_mutation:
                    if mutate_kernel:
                        max_kernel = base_arch['kernel_list'][unit_idx][layer_idx]
                        kernel_opts = list(range(1, max_kernel+1, 2))
                        kernel_list[unit_idx][layer_idx] = random.choice(kernel_opts)

                    if mutate_width:
                        max_width = base_arch['width_list'][unit_idx][layer_idx]
                        width_opts = []
                        for lev in range(4):
                            width_opts.append(int((1.0 - lev*0.25) * max_width))
                        width_list[unit_idx][layer_idx] = random.choice(width_opts)

        width_list[-1][-1] = base_arch['width_list'][-1][-1]

        return new_model_arch

    @staticmethod
    def crossover(model1, model2):
        """Crossovers the given architectures"""
        assert model1['num_classes'] == model2['num_classes']
        assert model1['num_channels'] == model2['num_channels']
        assert model1['dimensions'] == model2['dimensions']
        assert model1['bias'] == model2['bias']
        assert model1['n_units'] == model2['n_units']
        assert model1['bn'] == model2['bn']
        assert model1['unit'] == model2['unit']

        depth_list = []
        width_list = []
        kernel_list = []

        # crossover model depths
        for unit_idx in range(model1['n_units']):
            depth_list.append(random.choice([model1['depth_list'][unit_idx],
                                             model2['depth_list'][unit_idx]]))

        # crossover layers
        for unit_idx, depth in enumerate(depth_list):
            width_list.append([])
            kernel_list.append([])
            for d in range(depth):
                if d >= model1['depth_list'][unit_idx]:
                    width_list[unit_idx].append(model2['width_list'][unit_idx][d])
                    kernel_list[unit_idx].append(model2['kernel_list'][unit_idx][d])
                elif d >= model2['depth_list'][unit_idx]:
                    width_list[unit_idx].append(model1['width_list'][unit_idx][d])
                    kernel_list[unit_idx].append(model1['kernel_list'][unit_idx][d])
                else:
                    width_list[unit_idx].append(random.choice(
                        [model1['width_list'][unit_idx][d], model2['width_list'][unit_idx][d]]))
                    kernel_list[unit_idx].append(random.choice(
                        [model1['kernel_list'][unit_idx][d], model2['kernel_list'][unit_idx][d]]))

        width_list[-1][-1] = model1['width_list'][-1][-1]

        new_model_arch = {'num_classes': model1['num_classes'],
                          'num_channels': model1['num_channels'],
                          'dimensions': model1['dimensions'], 'bias': model1['bias'],
                          'n_units': model1['n_units'], 'bn': model1['bn'],
                          'unit': model1['unit'], 'depth_list': depth_list,
                          'width_list': width_list, 'kernel_list': kernel_list}

        return new_model_arch

    @staticmethod
    def get_unique_widths(sample):
        """Returns unique number of channel list of all layers in the model"""
        unique_widths = []
        for unit_idx, _ in enumerate(sample['width_list']):
            for layer_idx, _ in enumerate(sample['width_list'][unit_idx]):
                width = sample['width_list'][unit_idx][layer_idx]
                if width not in unique_widths:
                    unique_widths.append(width)

        return unique_widths


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


def ai85nasnet_sequential_cifar100(pretrained=False, **kwargs):
    """
    Constructs a sequential NAS model.
    """
    assert not pretrained
    n_units = 5
    depth_list = [4, 3, 3, 3, 2]
    width_list = [64, 64, 128, 128, 128]
    kernel_list = [3, 3, 3, 3, 3]
    bn = True

    return OnceForAll2DSequentialModel(n_units=n_units, depth_list=depth_list, bn=bn,
                                       width_list=width_list, kernel_list=kernel_list, **kwargs)


def ai85nasnet_sequential_kws20(pretrained=False, **kwargs):
    """
    Constructs a sequential NAS model.
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
        'name': 'ai85nasnet_sequential_cifar100',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85nasnet_sequential_kws20',
        'min_input': 1,
        'dim': 1,
    }
]
