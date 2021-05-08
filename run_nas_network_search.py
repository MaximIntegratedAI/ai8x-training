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
Application to run evolutionary search over trained Once For All model.
"""

import argparse
import fnmatch
import json
import os
import re
from pydoc import locate

import torch
from torch.utils.data import DataLoader

import ai8x
from nas import nas_utils, parse_nas_yaml
from nas.evo_search import EvolutionSearch


def parse_args(model_names, dataset_names):
    """Return the parsed arguments"""
    parser = argparse.ArgumentParser(description='Evolutionary search for a trained once '
                                                 'for all model')
    parser.add_argument('--model_path', metavar='DIR', required=True, help='path to model '
                                                                           'checkpoint')
    parser.add_argument('--arch', '-a', '--model', metavar='ARCH', required=True,
                        type=lambda s: s.lower(), dest='arch', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names))
    parser.add_argument('--dataset', metavar='S', required=True, choices=dataset_names,
                        help='dataset: ' + ' | '.join(dataset_names))
    parser.add_argument('--data', metavar='DIR', default='data', help='path to dataset')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--no-bias', action='store_true', default=False,
                        help='for models that support both bias and no bias, set the '
                             '`use bias` flag to true')
    parser.add_argument('--nas-policy', dest='nas_policy', required=True,
                        help='path to YAML file that defines the NAS '
                             '(once for all training) policy')
    parser.add_argument('--num-out-archs', default=1, type=int,
                        help='number of subnet architectures at the output')
    parser.add_argument('--export-archs', action='store_true', default=False,
                        help='exports found subnets to a json file if set to True')
    parser.add_argument('--arch-file', help='filepath where the json file is stores '
                                            'if `export-archs` is set True')

    return parser.parse_args()


def get_evo_search_params(nas_policy):
    """Get parameters used for evolutionary search from yaml file"""
    evo_search_params = {'population_size': 100, 'prob_mutation': 0.1, 'ratio_mutation': 0.5,
                         'ratio_parent': 0.25, 'num_iter': 500,
                         'constraints': {'max_num_weights': 4.5e5}}

    if 'evolution_search' in nas_policy:
        for key in evo_search_params:
            if key in nas_policy['evolution_search']:
                evo_search_params[key] = nas_policy['evolution_search'][key]

    return evo_search_params


def load_models():
    """Dynamically load models"""
    supported_models = []
    model_names = []

    for _, _, files in sorted(os.walk('models')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                fn = 'models.' + name[:-3]
                m = locate(fn)
                try:
                    for i in m.models:
                        i['module'] = fn
                    supported_models += m.models
                    model_names += [item['name'] for item in m.models]
                except AttributeError:
                    # Skip files that don't have 'models' or 'models.name'
                    pass

    return supported_models, model_names


def load_datasets():
    """Dynamically load datasets"""
    supported_sources = []
    dataset_names = []

    for _, _, files in sorted(os.walk('datasets')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                ds = locate('datasets.' + name[:-3])
                try:
                    supported_sources += ds.datasets
                    dataset_names += [item['name'] for item in ds.datasets]
                except AttributeError:
                    # Skip files that don't have 'datasets' or 'datasets.name'
                    pass

    return supported_sources, dataset_names


def get_data_loaders(supported_sources, args):
    """Dynamically loads data loaders"""
    selected_source = next((item for item in supported_sources if item['name'] == args.dataset))
    labels = selected_source['output']
    num_classes = len(labels)
    if num_classes == 1 or ('regression' in selected_source and selected_source['regression']):
        args.regression = True
    else:
        args.regression = False

    args.dimensions = selected_source['input']
    args.num_classes = len(selected_source['output'])

    train_dataset, val_dataset = selected_source['loader']((args.data, args))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader


def create_model(supported_models, args):
    """Create the model"""
    module = next(item for item in supported_models if item['name'] == args.arch)

    Model = locate(module['module'] + '.' + args.arch)

    if not Model:
        raise RuntimeError("Model " + args.arch + " not found\n")

    if module['dim'] > 1 and module['min_input'] > args.dimensions[2]:
        model = Model(pretrained=False, num_classes=args.num_classes,
                      num_channels=args.dimensions[0],
                      dimensions=(args.dimensions[1], args.dimensions[2]),
                      padding=(module['min_input'] - args.dimensions[2] + 1) // 2,
                      bias=not args.no_bias).to(args.device)
    else:
        model = Model(pretrained=False, num_classes=args.num_classes,
                      num_channels=args.dimensions[0],
                      dimensions=(args.dimensions[1], args.dimensions[2]),
                      bias=not args.no_bias).to(args.device)

    if '2D' in type(model).__name__:
        args.model_type = 'Conv2d'
    elif '1D' in type(model).__name__:
        args.model_type = 'Conv1d'
    else:
        args.model_type = 'Unknown'

    return model


def format_json_data(json_data):
    """Converts the json file content to human readable format"""
    json_data = re.sub(r'": \[\s+', '": [', json_data)
    json_data = re.sub(r'": \[\[\s+', '": [[', json_data)
    json_data = re.sub(r'\[\s+', '[', json_data)
    json_data = re.sub(r'\n\s+\]', ']', json_data)
    json_data = re.sub(r'\],\s+\[', '], [', json_data)
    json_data = re.sub(r',\s+(\d)', r', \1', json_data)
    json_data = re.sub(r'(\d),\s+(\d)', r'\1, \2', json_data)
    json_data = re.sub(r'true,\s+true',  'true, true', json_data)
    json_data = re.sub(r'true,\s+true]',  'true, true]', json_data)
    json_data = re.sub(r'true,\s+false',  'true, false', json_data)
    json_data = re.sub(r'true,\s+false]',  'true, false]', json_data)
    json_data = re.sub(r'false,\s+true',  'false, true', json_data)
    json_data = re.sub(r'false,\s+true]',  'false, true]', json_data)
    json_data = re.sub(r'false,\s+false',  'false, false', json_data)
    json_data = re.sub(r'false,\s+false]',  'false, false]', json_data)
    return json_data


def generate_out_file(arch_list, num_elems, in_shape, model_type, file_path):
    """Generates json file for the found subnet architectures"""
    file_content = []
    for idx in range(num_elems):
        arch = arch_list[idx][0]
        acc = arch_list[idx][1]

        bias_list = []
        for unit_idx in range(arch['n_units']):
            bias = []
            for _ in range(arch['depth_list'][unit_idx]):
                bias.append(arch['bias'])
            bias_list.append(bias)

        arch_dict = {
            'acc': acc,
            'type': model_type,
            'in_shape': in_shape,
            'out_class': arch['num_classes'],
            'n_units': arch['n_units'],
            'depth_list': arch['depth_list'],
            'width_list': arch['width_list'],
            'kernel_list': arch['kernel_list'],
            'bias_list': bias_list,
            'bn': arch['bn']
        }

        file_content.append(arch_dict)

    json_file_content = json.dumps(file_content, indent=4)
    json_file_content = format_json_data(json_file_content)

    with open(file_path, 'w') as fp:
        fp.write(json_file_content)


def main():
    """Main routine"""
    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=False)

    supported_models, model_names = load_models()
    supported_sources, dataset_names = load_datasets()

    args = parse_args(model_names, dataset_names)
    args.truncate_testset = False
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if use_cuda else "cpu")
    args.act_mode_8bit = False

    # Get policy for once for all training policy
    nas_policy = parse_nas_yaml.parse(args.nas_policy) \
        if args.nas_policy.lower() != '' else None

    # Get data loaders
    train_loader, val_loader = get_data_loaders(supported_sources, args)

    # Load model
    model = create_model(supported_models, args)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])

    # Calculate full model accuracy
    full_model_acc = nas_utils.calc_accuracy(None, model, train_loader, val_loader, args.device)
    print(f'Model Accuracy: {100*full_model_acc: .3f}%')

    # Run evolutionary search to find proper networks
    evo_search_params = get_evo_search_params(nas_policy)
    evo_search = EvolutionSearch(population_size=evo_search_params['population_size'],
                                 prob_mutation=evo_search_params['prob_mutation'],
                                 ratio_mutation=evo_search_params['ratio_mutation'],
                                 ratio_parent=evo_search_params['ratio_parent'],
                                 num_iter=evo_search_params['num_iter'])
    evo_search.set_model(model)
    arch_list = evo_search.run(evo_search_params['constraints'], train_loader,
                               val_loader, args.device)

    if args.export_archs:
        generate_out_file(arch_list, min(args.num_out_archs, len(arch_list)),
                          args.dimensions, args.model_type, args.arch_file)
    else:
        for idx in range(min(args.num_out_archs, len(arch_list))):
            print(f'Model-{idx}:')
            print(f'\tArch: {arch_list[idx][0]}')
            print(f'\tAcc: {arch_list[idx][1]}\n')


if __name__ == '__main__':
    main()
