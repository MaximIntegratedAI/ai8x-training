#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Test routine for Once For All training
"""
import importlib
import sys

import numpy as np
import torch
from torch import nn

import ai8x
import ai8x_nas

sys.path.insert(0, './models')

ofa_net = importlib.import_module('ai85ofanet-sequential')


def create_input_data_2d(num_channels, val=None, dims=(2, 2)):
    """Creates random data"""
    if val is not None:
        inp = torch.ones(1, num_channels, dims[0], dims[1]) * val
    else:
        inp = torch.randn(1, num_channels, dims[0], dims[1]) * 0.25

    return inp


def create_input_data_1d(num_channels, val):
    """Creates random data"""
    inp = torch.ones(1, num_channels, 2) * val

    return inp


def create_ofa_unit(depth, kernel_list, width_list, init_width, bias, pooling, bn,
                    kernel_weight_val, unit_dim):
    """Creates randomly initialized layer"""
    if unit_dim == 2:
        ofa_unit = ofa_net.OnceForAll2DSequentialUnit(depth, kernel_list, width_list, init_width,
                                                      bias, pooling, bn)
    else:
        ofa_unit = ofa_net.OnceForAll1DSequentialUnit(depth, kernel_list, width_list, init_width,
                                                      bias, pooling, bn)

    for layer in ofa_unit.layers:
        layer.op.weight.data.fill_(kernel_weight_val)
        if layer.op.bias:
            layer.op.bias.data.fill_(0.)

    return ofa_unit


def prep_test_unit(kernel_weight_val, unit_dim):
    """Prepares a unit for testing"""
    depth = 3
    if unit_dim == 2:
        kernel_size = 3
    elif unit_dim == 1:
        kernel_size = 5
    else:
        assert False, f'Not unit for dimension {unit_dim}'

    width_list = 12
    init_width = 1
    bias = False
    pooling = False
    bn = False

    if unit_dim == 2:
        return create_ofa_unit(depth, kernel_size, width_list, init_width, bias, pooling, bn,
                               kernel_weight_val, unit_dim)

    return create_ofa_unit(depth, kernel_size, width_list, init_width, bias, pooling, bn,
                           kernel_weight_val, unit_dim)


def prep_test_model(dims=(32, 32)):
    """Prepares a model for testing"""
    n_units = 6
    depth_list = [4, 3, 3, 3, 2, 2]
    width_list = [32, 64, 96, 96, 128, 256]
    kernel_list = [3, 3, 3, 3, 3, 3]

    model = ofa_net.OnceForAll2DSequentialModel(num_classes=8, num_channels=1, dimensions=dims,
                                                bias=True, n_units=n_units, depth_list=depth_list,
                                                width_list=width_list, kernel_list=kernel_list,
                                                bn=False)

    for unit in model.units:
        for layer in unit.layers:
            nn.init.normal_(layer.op.weight, mean=0, std=0.25)
            nn.init.normal_(layer.op.bias, mean=0, std=0.25)

    nn.init.normal_(model.classifier.op.weight, mean=0, std=0.25)
    nn.init.normal_(model.classifier.op.bias, mean=0, std=0.25)

    return model


def update_model_params(model, eps=1e-2):
    """Updates model with random values"""
    with torch.no_grad():
        for unit in model.units:
            for layer in unit.layers:
                layer.op.weight.data.add_(torch.randn(layer.op.weight.shape)*eps)
                layer.op.bias.data.add_(torch.randn(layer.op.bias.shape)*eps)


def test_unit2d():
    """Test routine for 2d Unit"""
    print('Test Unit2d')
    init_width = 1
    inp_val = 0.1
    kernel_weight_val = 0.1

    inp = create_input_data_2d(init_width, inp_val)
    ofa_unit = prep_test_unit(kernel_weight_val, 2)

    res = ofa_unit(inp)
    res_np = res.detach().cpu().numpy()

    print('\tTest for output shape:', end='\t')
    assert res_np.shape == (1, ofa_unit.layers[-1].out_channels, 2, 2), 'FAIL!!'
    print('PASS!')

    print('\tTest for output value:', end='\t')
    expected_out = inp_val
    width = init_width
    for i in range(ofa_unit.depth):
        expected_out = 4. * expected_out * kernel_weight_val * width
        width = ofa_unit.layers[i].out_channels

    res_unique = np.unique(res_np)
    assert res_unique.shape == (1,), f'FAIL!! Expected Output must have one unique value. ' \
                                     f'it has {res_unique.shape}'
    assert np.abs(res_unique[0] - expected_out) < 1e-6, 'FAIL!! ' \
                                                        f'Expected Output: {expected_out}, ' \
                                                        f'Unit Output: {res_unique[0]}'
    print('PASS!\n')


def test_unit1d():
    """Test routine for 1d Unit"""
    print('Test Unit1d')
    init_width = 1
    inp_val = 0.1
    kernel_weight_val = 0.1

    inp = create_input_data_1d(init_width, inp_val)
    ofa_unit = prep_test_unit(kernel_weight_val, 1)

    res = ofa_unit(inp)
    res_np = res.detach().cpu().numpy()

    print('\tTest for output shape:', end='\t')
    assert res_np.shape == (1, ofa_unit.layers[-1].out_channels, 2), 'FAIL!!'
    print('PASS!')

    print('\tTest for output value:', end='\t')
    expected_out = inp_val
    width = init_width
    for i in range(ofa_unit.depth):
        expected_out = 2. * expected_out * kernel_weight_val * width
        width = ofa_unit.layers[i].out_channels

    res_unique = np.unique(res_np)
    assert res_unique.shape == (1,), f'FAIL!! Expected Output must have one unique value. ' \
                                     f'It has {res_unique.shape}'
    assert np.abs(res_unique[0] - expected_out) < 1e-6, f'FAIL!! ' \
                                                        f'Expected Output: {expected_out}, ' \
                                                        f'Unit Output: {res_unique[0]}'
    print('PASS!\n')


def test_elastic_kernel_2d():
    """Test elastic kernel search for 2d model"""
    print('Test Elastic Kernel 2d')
    init_width = 1
    inp_val = 0.1
    kernel_weight_val = 0.1

    inp = create_input_data_2d(init_width, inp_val)
    ofa_unit = prep_test_unit(kernel_weight_val, 2)
    res_np_full = ofa_unit(inp).detach().cpu().numpy()

    level_list = [0, 1]
    num_trials = 100

    for level in level_list:
        print(f'\tTest elastic kernel sampling for level: {level}')
        k_size_list = []
        print('\t\tTest for kernel sample:', end='\t')
        for _ in range(num_trials):
            ai8x_nas.sample_subnet_kernel(ofa_unit, level=level)
            for layer in ofa_unit.layers:
                k_size_list.append(layer.kernel_size)
        assert np.unique(k_size_list).shape[0] == (level+1), f'FAIL!! Expected number of ' \
            f'observed kernels must be {level+1} but it has {np.unique(k_size_list).shape[0]}'
        print('PASS!')

        res = ofa_unit(inp)
        res_np = res.detach().cpu().numpy()

        print('\t\tTest for output shape:', end='\t')
        assert res_np.shape == (1, ofa_unit.layers[-1].out_channels, 2, 2), 'FAIL!!'
        print('PASS!')

        print('\t\tTest for output value:', end='\t')
        expected_out = inp_val
        width = init_width
        for i in range(ofa_unit.depth):
            multiplier = 4.0
            if ofa_unit.layers[i].kernel_size == 1:
                multiplier = 1.0
            expected_out = multiplier * expected_out * kernel_weight_val * width
            width = ofa_unit.layers[i].out_channels

        res_unique = np.unique(res_np)
        assert res_unique.shape == (1,), 'FAIL!! Expected Output should have one unique value. ' \
                                         f'It has {res_unique.shape} uniwue values.'
        assert np.abs(res_unique[0] - expected_out) < 1e-6, 'FAIL!! ' \
                                                            f'Expected Output: {expected_out}, ' \
                                                            f'Unit Output: {res_unique[0]}'
        print('PASS!')

        print('\t\tTest for model reset:', end='\t')
        ai8x_nas.reset_kernel_sampling(ofa_unit)
        res = ofa_unit(inp)
        res_np = res.detach().cpu().numpy()
        assert (res_np_full == res_np).all(), 'FAIL!'
        print('PASS!\n')


def test_elastic_kernel_1d():
    """Test elastic kernel search for 1d model"""
    print('Test Elastic Kernel 1d')
    init_width = 1
    inp_val = 0.1
    kernel_weight_val = 0.1

    inp = create_input_data_1d(init_width, inp_val)
    ofa_unit = prep_test_unit(kernel_weight_val, 1)
    res_np_full = ofa_unit(inp).detach().cpu().numpy()

    level_list = [0, 1, 2]
    num_trials = 100

    for level in level_list:
        print(f'\tTest elastic kernel sampling for level {level}')
        k_size_list = []
        print('\t\tTest for kernel sample:', end='\t')
        for _ in range(num_trials):
            ai8x_nas.sample_subnet_kernel(ofa_unit, level=level)
            for layer in ofa_unit.layers:
                k_size_list.append(layer.kernel_size)
        assert np.unique(k_size_list).shape[0] == (level+1), f'FAIL!! Expected number of ' \
            f'observed kernels must be {level+1} but it has {np.unique(k_size_list).shape[0]}'
        print('PASS!')

        res = ofa_unit(inp)
        res_np = res.detach().cpu().numpy()

        print('\t\tTest for output shape:', end='\t')
        assert res_np.shape == (1, ofa_unit.layers[-1].out_channels, 2), f'FAIL!! {res_np.shape}'
        print('PASS!')

        print('\t\tTest for output value:', end='\t')
        expected_out = inp_val
        width = init_width
        for i in range(ofa_unit.depth):
            multiplier = 2.0
            if ofa_unit.layers[i].kernel_size == 1:
                multiplier = 1.0
            expected_out = multiplier * expected_out * kernel_weight_val * width
            width = ofa_unit.layers[i].out_channels

        res_unique = np.unique(res_np)
        assert res_unique.shape == (1,), f'FAIL!! Expected Output should have one unique value. ' \
                                         f'It has {res_unique.shape} unique values.'
        assert np.abs(res_unique[0] - expected_out) < 1e-6, f'FAIL!! ' \
                                                            f'Expected Output: {expected_out}, ' \
                                                            f'Unit Output: {res_unique[0]}'
        print('PASS!')

        print('\t\tTest for model reset:', end='\t')
        ai8x_nas.reset_kernel_sampling(ofa_unit)
        res = ofa_unit(inp)
        res_np = res.detach().cpu().numpy()
        assert (res_np_full == res_np).all(), 'FAIL!'
        print('PASS!\n')


def test_elastic_depth_2d():
    """Test elastic depth search for 2d model"""
    print('Test Elastic Depth 2d')
    init_width = 1
    inp_val = 0.1
    kernel_weight_val = 0.1

    inp = create_input_data_2d(init_width, inp_val)
    ofa_unit = prep_test_unit(kernel_weight_val, 2)
    res_np_full = ofa_unit(inp).detach().cpu().numpy()

    level_list = [0, 1, 2]
    num_trials = 100

    for level in level_list:
        print(f'\tTest elastic depth sampling for level: {level}')
        d_size_list = []
        print('\t\tTest for kernel sample:', end='\t')
        for _ in range(num_trials):
            ai8x_nas.sample_subnet_depth(ofa_unit, level=level)
            d_size_list.append(ofa_unit.depth)
        assert np.unique(d_size_list).shape[0] == (level+1), f'FAIL!! Expected number of ' \
            f'observed kernels must be {level+1} but it has {np.unique(d_size_list).shape[0]}'
        print('PASS!')

        res = ofa_unit(inp)
        res_np = res.detach().cpu().numpy()

        print('\t\tTest for output shape:', end='\t')
        assert res_np.shape == (1, ofa_unit.layers[ofa_unit.depth-1].out_channels, 2, 2), \
               f'FAIL!! {res_np.shape}'
        print('PASS!')

        print('\t\tTest for output value:', end='\t')
        expected_out = inp_val
        width = init_width
        for i in range(ofa_unit.depth):
            multiplier = 4.0
            if ofa_unit.layers[i].kernel_size == 1:
                multiplier = 1.0
            expected_out = multiplier * expected_out * kernel_weight_val * width
            width = ofa_unit.layers[i].out_channels

        res_unique = np.unique(res_np)
        assert res_unique.shape == (1,), f'FAIL!! Expected Output should have one unique value. ' \
                                         f'It has {res_unique.shape} unique values'
        assert np.abs(res_unique[0] - expected_out) < 1e-6, f'FAIL!! ' \
                                                            f'Expected Output: {expected_out}, ' \
                                                            f'Unit Output: {res_unique[0]}'
        print('PASS!')

        print('\t\tTest for model reset:', end='\t')
        ai8x_nas.reset_depth_sampling(ofa_unit)
        res = ofa_unit(inp)
        res_np = res.detach().cpu().numpy()
        assert (res_np_full == res_np).all(), 'FAIL!'
        print('PASS!\n')


def test_elastic_depth_1d():
    """Test elastic depth search for 1d model"""
    print('Test Elastic Depth 1d')
    init_width = 1
    inp_val = 0.1
    kernel_weight_val = 0.1

    inp = create_input_data_1d(init_width, inp_val)
    ofa_unit = prep_test_unit(kernel_weight_val, 1)
    res_np_full = ofa_unit(inp).detach().cpu().numpy()

    level_list = [0, 1, 2]
    num_trials = 100

    for level in level_list:
        print(f'\tTest elastic depth sampling for level {level}')
        d_size_list = []
        print('\t\tTest for depth sample:', end='\t')
        for _ in range(num_trials):
            ai8x_nas.sample_subnet_depth(ofa_unit, level=level)
            d_size_list.append(ofa_unit.depth)
        assert np.unique(d_size_list).shape[0] == (level+1), f'FAIL!! Expected number of ' \
               f'observed depths must be {level+1} but it has {np.unique(d_size_list).shape[0]}'
        print('PASS!')

        res = ofa_unit(inp)
        res_np = res.detach().cpu().numpy()

        print('\t\tTest for output shape:', end='\t')
        assert res_np.shape == (1, ofa_unit.layers[ofa_unit.depth-1].out_channels, 2), \
               f'FAIL!! {res_np.shape}'
        print('PASS!')

        print('\t\tTest for output value:', end='\t')
        expected_out = inp_val
        width = init_width
        for i in range(ofa_unit.depth):
            multiplier = 2.0
            if ofa_unit.layers[i].kernel_size == 1:
                multiplier = 1.0
            expected_out = multiplier * expected_out * kernel_weight_val * width
            width = ofa_unit.layers[i].out_channels

        res_unique = np.unique(res_np)
        assert res_unique.shape == (1,), f'FAIL!! Expected Output should have one unique value ' \
                                         f'It has {res_unique.shape} unique values.'
        assert np.abs(res_unique[0] - expected_out) < 1e-6, f'FAIL!! ' \
                                                            f'Expected Output: {expected_out}, ' \
                                                            f'Unit Output: {res_unique[0]}'
        print('PASS!')

        print('\t\tTest for model reset:', end='\t')
        ai8x_nas.reset_depth_sampling(ofa_unit)
        res = ofa_unit(inp)
        res_np = res.detach().cpu().numpy()
        assert (res_np_full == res_np).all(), 'FAIL!'
        print('PASS!\n')


def test_elastic_width_2d():
    """Test elastic width search for 2d model"""
    print('Test Elastic Width 2d')
    init_width = 1
    inp_dim = (32, 32)
    num_trials = 50

    inp = create_input_data_2d(init_width, val=None, dims=inp_dim)
    seq_ofa_model = prep_test_model(dims=inp_dim)

    seq_ofa_model.eval()
    with torch.no_grad():
        res_np_full_init = seq_ofa_model(inp).detach().cpu().numpy()

    print('\tTest for elastic width sampling level 0')
    print('\t\tTest for width sample:', end='\t')
    for _ in range(num_trials):
        ai8x_nas.sample_subnet_width(seq_ofa_model, level=0, sample_depth=False)
        for u_idx, u in enumerate(seq_ofa_model.units):
            for i in range(u.depth):
                assert u.layers[i].in_channels == u.layers[i].op.in_channels, f'FAIL!! ' \
                    f'Unit {u_idx}, Layer {i}: ' \
                    f'Expected in channels is {u.layers[i].op.in_channels}. ' \
                    f'It has {u.layers[i].in_channels}'
                assert u.layers[i].out_channels == u.layers[i].op.out_channels, f'FAIL!! ' \
                    f'Unit {u_idx}, Layer {i}: ' \
                    f'Expected out channels is {u.layers[i].op.out_channels}. ' \
                    f'It has {u.layers[i].out_channels}'
    print('PASS!')

    print('\t\tTest for output value:', end='\t')
    inp = create_input_data_2d(init_width, val=None, dims=inp_dim)

    for n in range(num_trials):
        seq_ofa_model = prep_test_model(dims=inp_dim)

        seq_ofa_model.eval()
        with torch.no_grad():
            res_np_full_init = seq_ofa_model(inp).detach().cpu().numpy()

        ai8x_nas.sample_subnet_width(seq_ofa_model, level=0, sample_depth=False)
        ai8x_nas.reset_width_sampling(seq_ofa_model)

        seq_ofa_model.eval()
        with torch.no_grad():
            res_np_full_end = seq_ofa_model(inp).detach().cpu().numpy()

        diff = np.abs(res_np_full_init - res_np_full_end)
        diff_rat = diff / np.abs(res_np_full_init)
        assert diff_rat.max() < 1e-2, f'FAIL!! Iteration {n} ' \
                                      f'Maximum DIfference: {diff.max(), diff_rat.max()}'

    print('PASS!\n')


def test():
    """Test routine for nas implementation"""
    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=False)
    test_unit2d()
    test_unit1d()
    test_elastic_kernel_2d()
    test_elastic_kernel_1d()
    test_elastic_depth_2d()
    test_elastic_depth_1d()
    test_elastic_width_2d()


if __name__ == "__main__":
    test()
