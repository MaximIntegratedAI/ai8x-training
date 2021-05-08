###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Command line parser for the Training/Quantization software.
"""

import argparse

# pylint: disable=wrong-import-order
import distiller
import distiller.quantization
from distiller.utils import float_range_argparse_checker as float_range
from examples.auto_compression.amc import amc_args as adc

from devices import device

SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params', 'onnx',
                   'onnx_simplified']


def get_parser(model_names, dataset_names):
    """
    Return the argument parser
    """
    parser = argparse.ArgumentParser(description='Image classification model')
    parser.add_argument('--device', type=device, default=84,
                        help='set device (default: AI84)')
    parser.add_argument('--8-bit-mode', '-8', dest='act_mode_8bit', action='store_true',
                        default=False,
                        help='simluate device operation (8-bit data)')
    parser.add_argument('--arch', '-a', '--model', metavar='ARCH', required=True,
                        type=lambda s: s.lower(), dest='cnn',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names))
    parser.add_argument('--dataset', metavar='S', required=True,
                        choices=dataset_names,
                        help="dataset(s) (" + ', '.join(dataset_names) + ")")
    parser.add_argument('--truncate-testset', action='store_true', default=False,
                        help='get only the first image from the test set')
    parser.add_argument('--data', metavar='DIR', default='data', help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of total epochs to run (default: 90)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--kernel-stats', action='store_true', default=False,
                        help='compute kernel statistics')
    parser.add_argument('--use-bias', action='store_true', default=False,
                        help='for models that support both bias and no bias, set the '
                             '`use bias` flag to true')
    parser.add_argument('--avg-pool-rounding', action='store_true', default=False,
                        help='when simulating, use "round()" in AvgPool operations '
                             '(default: use "floor()")')

    qat_args = parser.add_argument_group('Quantization Arguments')
    qat_args.add_argument('--qat-policy', dest='qat_policy', default='qat_policy.yaml',
                          help='path to YAML file that defines the '
                               'QAT (quantization-aware training) policy')

    ofa_args = parser.add_argument_group('NAS Training Arguments')
    ofa_args.add_argument('--nas', action='store_true', default=False,
                          help='enables NAS (network architecture search) training')
    ofa_args.add_argument('--nas-kd-resume-from', default='', type=str, metavar='PATH',
                          help='path to checkpoint from which to load NAS teacher weights')
    ofa_args.add_argument('--nas-policy', dest='nas_policy', default='',
                          help='path to YAML file that defines the '
                               'NAS (network architecture search) policy')
    ofa_args.add_argument('--nas-stage-transition-list', dest='nas_stage_transition_list',
                          default=None, help='list of tuples to define epochs to change the '
                                             'stages and levels of NAS sampling policy. '
                                             'Use --nas-policy option instead!')

    optimizer_args = parser.add_argument_group('Optimizer Arguments')
    optimizer_args.add_argument('--optimizer', default='SGD',
                                help='optimizer for training (default: SGD)')
    optimizer_args.add_argument('--lr', '--learning-rate', default=0.1,
                                type=float, metavar='LR', help='initial learning rate')
    optimizer_args.add_argument('--momentum', default=0.9, type=float,
                                metavar='M', help='momentum')
    optimizer_args.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                                metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    load_checkpoint_group = parser.add_argument_group('Resuming Arguments')
    load_checkpoint_group_exc = load_checkpoint_group.add_mutually_exclusive_group()
    load_checkpoint_group_exc.add_argument('--resume-from', dest='resumed_checkpoint_path',
                                           default='', type=str, metavar='PATH',
                                           help='path to latest checkpoint. Use to resume paused '
                                                'training session.')
    load_checkpoint_group_exc.add_argument('--exp-load-weights-from', dest='load_model_path',
                                           default='', type=str, metavar='PATH',
                                           help='path to checkpoint to load weights from '
                                                '(excluding other fields) (experimental)')
    load_checkpoint_group.add_argument('--pretrained', dest='pretrained', action='store_true',
                                       help='use pre-trained model')
    load_checkpoint_group.add_argument('--reset-optimizer', action='store_true',
                                       help='Flag to override optimizer if resumed from '
                                            'checkpoint. This will reset epochs count.')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on test set')
    mgroup = parser.add_mutually_exclusive_group()
    mgroup.add_argument('--save-csv', dest='csv_prefix', default=None, type=str,
                        help='save as CSVs with the given prefix during evaluation')
    mgroup.add_argument('--save-sample', dest='generate_sample', type=int,
                        help='save the sample at given index as NumPy sample data')
    parser.add_argument('--shap', default=0, type=int,
                        help='select # of images from the test set and plot SHAP after evaluation')
    parser.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE',
                        default=list(),
                        help='collect activation statistics on phases: train, valid, and/or test'
                             ' (WARNING: this slows down training)')
    parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true',
                        default=False,
                        help='print masks sparsity table at end of each epoch')
    parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true',
                        default=False,
                        help='log the parameter tensors histograms to file (WARNING: this can use '
                             'significant disk space)')
    parser.add_argument('--summary', type=lambda s: s.lower(), choices=SUMMARY_CHOICES,
                        help='print a summary of the model, and exit - options: ' +
                        ' | '.join(SUMMARY_CHOICES))
    parser.add_argument('--summary-filename', default='model',
                        help='file name (w/o extension) for the model summary (default: "model"')

    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model '
                             '(default is to use hard-coded schedule)')
    parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                        type=lambda s: s.lower(),
                        help='test the sensitivity of layers to pruning')
    parser.add_argument('--sense-range', dest='sensitivity_range', type=float, nargs=3,
                        default=[0.0, 0.95, 0.05],
                        help='an optional parameter for sensitivity testing providing the range '
                             'of sparsities to test.\n'
                             'This is equivalent to creating sensitivities = np.arange(start, '
                             'stop, step)')
    parser.add_argument('--extras', default=None, type=str,
                        help='file with extra configuration information')
    parser.add_argument('--deterministic', '--det', action='store_true',
                        help='Ensure deterministic execution for re-producible results.')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used (default is to '
                             'use all available devices)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use CPU only. \n'
                        'Flag not set => uses GPUs according to the --gpus flag value.'
                        'Flag set => overrides the --gpus flag')
    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump '
                        'logs and checkpoints')
    parser.add_argument('--validation-split', '--valid-size', '--vs', dest='validation_split',
                        type=float_range(exc_max=True), default=0.1,
                        help='Portion of training dataset to set aside for validation')
    parser.add_argument('--effective-train-size', '--etrs', type=float_range(exc_min=True),
                        default=1.,
                        help='Portion of training dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument '
                             'is applied AFTER the train-validation split according to that '
                             'argument')
    parser.add_argument('--effective-valid-size', '--evs', type=float_range(exc_min=True),
                        default=1.,
                        help='Portion of validation dataset to be used in each epoch. '
                             'NOTE: If --validation-split is set, then the value of this argument '
                             'is applied AFTER the train-validation split according to that '
                             'argument')
    parser.add_argument('--effective-test-size', '--etes', type=float_range(exc_min=True),
                        default=1.,
                        help='Portion of test dataset to be used in each epoch')
    parser.add_argument('--confusion', dest='display_confusion', default=False,
                        action='store_true',
                        help='Display the confusion matrix')
    parser.add_argument('--embedding', dest='display_embedding', default=False,
                        action='store_true',
                        help='Display embedding (using projector)')
    parser.add_argument('--pr-curves', dest='display_prcurves', default=False,
                        action='store_true',
                        help='Display the precision-recall curves')
    mgroup = parser.add_mutually_exclusive_group()
    mgroup.add_argument('--no-tensorboard', default=True, action='store_false',
                        help='Disable TensorBoard (default)')
    mgroup.add_argument('--enable-tensorboard', '--tensorboard', dest='tblog', default=False,
                        action='store_true',
                        help='Enable TensorBoard')
    parser.add_argument('--regression', dest='regression', default=False,
                        action='store_true',
                        help='Force regression output')
    parser.add_argument('--earlyexit_lossweights', type=float, nargs='*',
                        dest='earlyexit_lossweights', default=None,
                        help='List of loss weights for early exits '
                             '(e.g. --earlyexit_lossweights 0.1 0.3)')
    parser.add_argument('--earlyexit_thresholds', type=float, nargs='*',
                        dest='earlyexit_thresholds', default=None,
                        help='List of EarlyExit thresholds (e.g. --earlyexit_thresholds 1.2 0.9)')
    parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                        help='number of best scores to track and report (default: 1)')
    parser.add_argument('--load-serialized', dest='load_serialized', action='store_true',
                        default=False,
                        help='Load a model without DataParallel wrapping it')
    parser.add_argument('--thinnify', dest='thinnify', action='store_true', default=False,
                        help='physically remove zero-filters and create a smaller model')
    parser.add_argument('--sparsity-perf', action='store_true', default=False,
                        help='when determining best epoch, use sparsity as primary key')

    distiller.knowledge_distillation.add_distillation_args(parser, model_names, True)
    distiller.quantization.add_post_train_quant_args(parser)
    distiller.pruning.greedy_filter_pruning.add_greedy_pruner_args(parser)
    adc.add_automl_args(parser)
    return parser
