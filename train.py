#!/usr/bin/env python3
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

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    save_checkpoint()
    compression_scheduler.on_epoch_end(epoch)

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This example application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""

import copy
import fnmatch
import logging
import operator
import os
import sys
import time
import traceback
from collections import OrderedDict
from functools import partial
from pydoc import locate

import numpy as np

import matplotlib
from pkg_resources import parse_version

# TensorFlow 2.x compatibility
try:
    import tensorboard  # pylint: disable=import-error
    import tensorflow  # pylint: disable=import-error
    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
except (ModuleNotFoundError, AttributeError):
    pass

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

# pylint: disable=wrong-import-order
import distiller
import distiller.apputils as apputils
import distiller.model_summaries as model_summaries
import examples.auto_compression.amc as adc
import shap
import torchnet.meter as tnt
from distiller.data_loggers import PythonLogger, TensorBoardLogger
# pylint: disable=no-name-in-module
from distiller.data_loggers.collector import (QuantCalibrationStatsCollector,
                                              RecordsActivationStatsCollector,
                                              SummaryActivationStatsCollector, collectors_context)
from distiller.quantization.range_linear import PostTrainLinearQuantizer

# pylint: enable=no-name-in-module
import ai8x
import ai8x_nas
import datasets
import nnplot
import parse_qat_yaml
import parsecmd
import sample
from nas import parse_nas_yaml

# from range_linear_ai84 import PostTrainLinearQuantizerAI84

matplotlib.use("pgf")

# Logger handle
msglogger = None

# Globals
weight_min = None
weight_max = None
weight_count = None
weight_sum = None
weight_stddev = None
weight_mean = None


def main():
    """main"""
    script_dir = os.path.dirname(__file__)
    global msglogger  # pylint: disable=global-statement

    supported_models = []
    supported_sources = []
    model_names = []
    dataset_names = []

    # Dynamically load models
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

    # Dynamically load datasets
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

    # Parse arguments
    args = parsecmd.get_parser(model_names, dataset_names).parse_args()

    # Set hardware device
    ai8x.set_device(args.device, args.act_mode_8bit, args.avg_pool_rounding)

    if args.epochs is None:
        args.epochs = 90

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.shap > 0:
        args.batch_size = 100 + args.shap

    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name,
                                         args.output_dir)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(args.compress, msglogger.logdir)
    msglogger.debug("Distiller: %s", distiller.__version__)

    start_epoch = 0
    ending_epoch = args.epochs
    perf_scores_history = []

    if args.evaluate and args.shap == 0:
        args.deterministic = True
    if args.deterministic:
        # torch.set_deterministic(True)
        distiller.set_deterministic(args.seed)  # For experiment reproducability
        if args.seed is not None:
            distiller.set_seed(args.seed)
    else:
        # Turn on CUDNN benchmark mode for best performance. This is usually "safe" for image
        # classification models, as the input sizes don't change during the run
        # See here:
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = True

    if args.cpu or not torch.cuda.is_available():
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError as exc:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated '
                                 'list of integers only') from exc
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} '
                                     'devices available'
                                     .format(dev_id, available_gpus))
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])

    if args.earlyexit_thresholds:
        args.num_exits = len(args.earlyexit_thresholds) + 1
        args.loss_exits = [0] * args.num_exits
        args.losses_exits = []
        args.exiterrors = []

    selected_source = next((item for item in supported_sources if item['name'] == args.dataset))
    args.labels = selected_source['output']
    args.num_classes = len(args.labels)
    if args.num_classes == 1 \
       or ('regression' in selected_source and selected_source['regression']):
        args.regression = True
    dimensions = selected_source['input']
    args.dimensions = dimensions

    args.datasets_fn = selected_source['loader']
    args.visualize_fn = selected_source['visualize'] \
        if 'visualize' in selected_source else datasets.visualize_data

    if args.regression and args.display_confusion:
        raise ValueError('ERROR: Argument --confusion cannot be used with regression')
    if args.regression and args.display_prcurves:
        raise ValueError('ERROR: Argument --pr-curves cannot be used with regression')
    if args.regression and args.display_embedding:
        raise ValueError('ERROR: Argument --embedding cannot be used with regression')

    model = create_model(supported_models, dimensions, args)

    # if args.add_logsoftmax:
    #     model = nn.Sequential(model, nn.LogSoftmax(dim=1))
    # if args.add_softmax:
    #     model = nn.Sequential(model, nn.Softmax(dim=1))

    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    pylogger = PythonLogger(msglogger, log_1d=True)
    all_loggers = [pylogger]
    if args.tblog:
        tflogger = TensorBoardLogger(msglogger.logdir, log_1d=True, comment='_'+args.dataset)

        tflogger.tblogger.writer.add_text('Command line', str(args))

        if dimensions[2] > 1:
            dummy_input = torch.randn((1, ) + dimensions)
        else:  # 1D input
            dummy_input = torch.randn((1, ) + dimensions[:-1])
        tflogger.tblogger.writer.add_graph(model.to('cpu'), (dummy_input, ), False)

        all_loggers.append(tflogger)
        all_tbloggers = [tflogger]
    else:
        tflogger = None
        all_tbloggers = []

    # Capture thresholds for early-exit training
    if args.earlyexit_thresholds:
        msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)

    # Get policy for quantization aware training
    qat_policy = parse_qat_yaml.parse(args.qat_policy) \
        if args.qat_policy.lower() != "none" else None

    # Get policy for once for all training policy
    nas_policy = parse_nas_yaml.parse(args.nas_policy) \
        if args.nas and args.nas_policy.lower() != '' else None

    # We can optionally resume from a checkpoint
    optimizer = None
    if args.resumed_checkpoint_path:
        update_old_model_params(args.resumed_checkpoint_path, model)
        if qat_policy is not None:
            checkpoint = torch.load(args.resumed_checkpoint_path,
                                    map_location=lambda storage, loc: storage)
            if checkpoint.get('epoch', None) >= qat_policy['start_epoch']:
                ai8x.fuse_bn_layers(model)
        model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
            model, args.resumed_checkpoint_path, model_device=args.device)
        ai8x.update_model(model)
    elif args.load_model_path:
        update_old_model_params(args.load_model_path, model)
        if qat_policy is not None:
            checkpoint = torch.load(args.load_model_path,
                                    map_location=lambda storage, loc: storage)
            if checkpoint.get('epoch', None) >= qat_policy['start_epoch']:
                ai8x.fuse_bn_layers(model)
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)
        ai8x.update_model(model)

    if not args.load_serialized and args.gpus != -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpus).to(args.device)

    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and '
                           'resetting epoch count to 0')

    # Define loss function (criterion)
    if not args.regression:
        if 'weight' in selected_source:
            criterion = nn.CrossEntropyLoss(
                torch.Tensor(selected_source['weight'])
            ).to(args.device)
        else:
            criterion = nn.CrossEntropyLoss().to(args.device)
    else:
        criterion = nn.MSELoss().to(args.device)

    if optimizer is None:
        optimizer = create_optimizer(model, args)
        msglogger.info('Optimizer Type: %s', type(optimizer))
        msglogger.info('Optimizer Args: %s', optimizer.defaults)

    if args.amc_cfg_file:
        return automated_deep_compression(model, criterion, optimizer, pylogger, args)
    if args.greedy:
        return greedy(model, criterion, optimizer, pylogger, args)

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        return summarize_model(model, args.dataset, which_summary=args.summary,
                               filename=args.summary_filename)

    activations_collectors = create_activation_stats_collectors(model, *args.activation_stats)

    if args.qe_calibration:
        msglogger.info('Quantization calibration stats collection enabled:')
        msglogger.info('\tStats will be collected for {:.1%} '
                       'of test dataset'.format(args.qe_calibration))
        msglogger.info('\tSetting constant seeds and converting model to serialized execution')
        distiller.set_deterministic()
        model = distiller.make_non_parallel_copy(model)
        activations_collectors.update(create_quantization_stats_collector(model))
        args.evaluate = True
        args.effective_test_size = args.qe_calibration

    # Load the datasets
    train_loader, val_loader, test_loader, _ = apputils.get_data_loaders(
        args.datasets_fn, (os.path.expanduser(args.data), args), args.batch_size,
        args.workers, args.validation_split, args.deterministic,
        args.effective_train_size, args.effective_valid_size, args.effective_test_size)
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    if args.sensitivity is not None:
        sensitivities = np.arange(args.sensitivity_range[0], args.sensitivity_range[1],
                                  args.sensitivity_range[2])
        return sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)

    if args.evaluate:
        return evaluate_model(model, criterion, test_loader, pylogger, activations_collectors,
                              args, compression_scheduler)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress,
                                                      compression_scheduler,
                                                      (start_epoch-1)
                                                      if args.resumed_checkpoint_path else None)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
    model.to(args.device)

    if args.thinnify:
        # zeros_mask_dict = distiller.create_model_masks_dict(model)
        assert args.resumed_checkpoint_path is not None, \
            "You must use --resume-from to provide a checkpoint file to thinnify"
        distiller.remove_filters(model, compression_scheduler.zeros_mask_dict, args.cnn,
                                 args.dataset, optimizer=None)
        apputils.save_checkpoint(0, args.cnn, model, optimizer=None,
                                 scheduler=compression_scheduler,
                                 name="{}_thinned".format(args.resumed_checkpoint_path.
                                                          replace(".pth.tar", "")),
                                 dir=msglogger.logdir)
        print("Note: your model may have collapsed to random inference, "
              "so you may want to fine-tune")
        return None

    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(supported_models, dimensions, args)
        if args.kd_resume:
            teacher = apputils.load_lean_checkpoint(teacher, args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt,
                                                args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch,
                                         ending_epoch=args.epochs, frequency=1)

        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)

    if start_epoch >= ending_epoch:
        msglogger.error('epoch count is too low, starting epoch is %d but total epochs set '
                        'to %d', start_epoch, ending_epoch)
        raise ValueError('Epochs parameter is too low. Nothing to do.')

    if args.nas:
        assert isinstance(model, ai8x_nas.OnceForAllModel), 'Model should implement ' \
                                        'OnceForAllModel interface for NAS training!'
        if nas_policy:
            args.nas_stage_transition_list = create_nas_training_stage_list(model, nas_policy)
            args.nas_kd_params = nas_policy['kd_params'] if 'kd_params' in nas_policy else None
            if args.nas_kd_resume_from == '':
                args.nas_kd_policy = None
            else:
                if args.nas_kd_params['teacher_model'] == 'full_model':
                    kd_end_epoch = args.epochs
                else:
                    kd_end_epoch = get_next_stage_start_epoch(start_epoch,
                                                              args.nas_stage_transition_list,
                                                              args.epochs)
                create_nas_kd_policy(model, compression_scheduler, start_epoch, kd_end_epoch, args)

    vloss = 10**6
    for epoch in range(start_epoch, ending_epoch):
        if qat_policy is not None and epoch > 0 and epoch == qat_policy['start_epoch']:
            # Fuse the BN parameters into conv layers before Quantization Aware Training (QAT)
            ai8x.fuse_bn_layers(model)

            # Switch model from unquantized to quantized for QAT
            ai8x.initiate_qat(model, qat_policy)

            # Model is re-transferred to GPU in case parameters were added
            model.to(args.device)

            # Empty the performance scores list for QAT operation
            perf_scores_history = []
            if args.name:
                args.name = f'{args.name}_qat'
            else:
                args.name = 'qat'

        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch, metrics=vloss)

        # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
                  loggers=all_loggers, args=args)
            distiller.log_weights_sparsity(model, epoch, loggers=all_loggers)
            distiller.log_activation_statistics(epoch, "train", loggers=all_tbloggers,
                                                collector=collectors["sparsity"])
            if args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(model, compression_scheduler))

        # evaluate on validation set
        run_validation = not args.nas or (args.nas and (epoch < nas_policy['start_epoch']))
        run_nas_validation = args.nas and (epoch >= nas_policy['start_epoch']) and \
            ((epoch+1) % nas_policy['validation_freq'] == 0)

        if run_validation or run_nas_validation:
            checkpoint_name = args.name

            # run pre validation steps if NAS is running
            if run_nas_validation:
                update_bn_stats(train_loader, model, args)
                stage, level = get_nas_training_stage(epoch, args.nas_stage_transition_list)
                if args.name:
                    checkpoint_name = f'{args.name}_nas_stg{stage}_lev{level}'
                else:
                    checkpoint_name = f'nas_stg{stage}_lev{level}'

            with collectors_context(activations_collectors["valid"]) as collectors:
                top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args, epoch,
                                             tflogger)
                distiller.log_activation_statistics(epoch, "valid", loggers=all_tbloggers,
                                                    collector=collectors["sparsity"])
                save_collectors_data(collectors, msglogger.logdir)

            if not args.regression:
                stats = ('Performance/Validation/', OrderedDict([('Loss', vloss), ('Top1', top1)]))
                if args.num_classes > 5:
                    stats[1]['Top5'] = top5
            else:
                stats = ('Performance/Validation/', OrderedDict([('Loss', vloss), ('MSE', top1)]))

            distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1,
                                            log_freq=1, loggers=all_tbloggers)

            # Update the list of top scores achieved so far
            update_training_scores_history(perf_scores_history, model, top1, top5, epoch, args)

            # Save the checkpoint
            if run_validation:
                is_best = epoch == perf_scores_history[0].epoch
                checkpoint_extras = {'current_top1': top1,
                                     'best_top1': perf_scores_history[0].top1,
                                     'best_epoch': perf_scores_history[0].epoch}
            else:
                is_best = False
                checkpoint_extras = {'current_top1': top1}

            apputils.save_checkpoint(epoch, args.cnn, model, optimizer=optimizer,
                                     scheduler=compression_scheduler, extras=checkpoint_extras,
                                     is_best=is_best, name=checkpoint_name,
                                     dir=msglogger.logdir)

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

    # Finally run results on the test set
    test(test_loader, model, criterion, [pylogger], activations_collectors, args=args)
    return None


OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


def create_model(supported_models, dimensions, args):
    """Create the model"""
    module = next(item for item in supported_models if item['name'] == args.cnn)

    # Override distiller's input shape detection. This is not a very clean way to do it since
    # we're replacing a protected member.
    distiller.utils._validate_input_shape = (  # pylint: disable=protected-access
        lambda _a, _b: (1, ) + dimensions[:module['dim'] + 1]
    )

    Model = locate(module['module'] + '.' + args.cnn)
    if not Model:
        raise RuntimeError("Model " + args.cnn + " not found\n")

    # Set model paramaters
    if args.act_mode_8bit:
        weight_bits = 8
        bias_bits = 8
        quantize_activation = True
    else:
        weight_bits = None
        bias_bits = None
        quantize_activation = False

    if module['dim'] > 1 and module['min_input'] > dimensions[2]:
        model = Model(pretrained=False, num_classes=args.num_classes,
                      num_channels=dimensions[0],
                      dimensions=(dimensions[1], dimensions[2]),
                      padding=(module['min_input'] - dimensions[2] + 1) // 2,
                      bias=args.use_bias,
                      weight_bits=weight_bits,
                      bias_bits=bias_bits,
                      quantize_activation=quantize_activation).to(args.device)
    else:
        model = Model(pretrained=False, num_classes=args.num_classes,
                      num_channels=dimensions[0],
                      dimensions=(dimensions[1], dimensions[2]),
                      bias=args.use_bias,
                      weight_bits=weight_bits,
                      bias_bits=bias_bits,
                      quantize_activation=quantize_activation).to(args.device)

    return model


def create_optimizer(model, args):
    """Create the optimizer"""
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        msglogger.info('Unknown optimizer type: %s. SGD is set as optimizer!!!', args.optimizer)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    return optimizer


def create_nas_kd_policy(model, compression_scheduler, epoch, next_state_start_epoch, args):
    """Create knowledge distillation policy for nas"""
    teacher = copy.deepcopy(model)
    dlw = distiller.DistillationLossWeights(args.nas_kd_params['distill_loss'],
                                            args.nas_kd_params['student_loss'], 0)
    args.nas_kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher,
                                                               args.nas_kd_params['temperature'],
                                                               dlw)
    compression_scheduler.add_policy(args.nas_kd_policy, starting_epoch=epoch,
                                     ending_epoch=next_state_start_epoch, frequency=1)

    msglogger.info('\nStudent-Teacher knowledge distillation enabled for NAS:')
    msglogger.info('\tStart Epoch: %d, End Epoch: %d', epoch, next_state_start_epoch)
    msglogger.info('\tTemperature: %s', args.nas_kd_params['temperature'])
    msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                   ' | '.join(['{:.2f}'.format(val) for val in dlw]))


def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args):
    """Training loop for one epoch."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    if not args.regression:
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(args.num_classes, 5)))
    else:
        classerr = tnt.MSEMeter()
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    # For Early Exit, we define statistics for each exit
    # So exiterrors is analogous to classerr for the non-Early Exit case
    if args.earlyexit_lossweights:
        args.exiterrors = []
        for exitnum in range(args.num_exits):
            if not args.regression:
                args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))
            else:
                args.exiterrors.append(tnt.MSEMeter())

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = (total_samples + batch_size - 1) // batch_size
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    if args.nas:
        if args.nas_stage_transition_list is not None:
            stage, level = get_nas_training_stage(epoch, args.nas_stage_transition_list)
            prev_stage, _ = get_nas_training_stage(epoch-1, args.nas_stage_transition_list)
        else:
            stage = prev_stage = 0
            level = 0

        if prev_stage != stage:
            if args.nas_kd_params:
                if ('teacher_model' not in args.nas_kd_params) or \
                    ('teacher_model' in args.nas_kd_params and
                     args.nas_kd_params['teacher_model'] == 'full_model' and prev_stage == 0):
                    create_nas_kd_policy(model, compression_scheduler, epoch, args.epochs, args)
                elif 'teacher_model' in args.nas_kd_params and \
                     args.nas_kd_params['teacher_model'] == 'prev_stage_model':
                    next_stage_start_epoch = get_next_stage_start_epoch(
                        epoch, args.nas_stage_transition_list, args.epochs)
                    create_nas_kd_policy(model, compression_scheduler, epoch,
                                         next_stage_start_epoch, args)

    # Switch to train mode
    model.train()
    acc_stats = []
    end = time.time()
    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Set nas parameters if necessary
        if args.nas:
            if stage == 1:
                ai8x_nas.sample_subnet_kernel(model, level)
            elif stage == 2:
                ai8x_nas.sample_subnet_depth(model, level)
            elif stage == 3:
                ai8x_nas.sample_subnet_width(model, level)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        if not hasattr(args, 'kd_policy') or args.kd_policy is None:
            if not hasattr(args, 'nas_kd_policy') or args.nas_kd_policy is None:
                output = model(inputs)
            else:
                output = args.nas_kd_policy.forward(inputs)
        else:
            output = args.kd_policy.forward(inputs)

        if not args.earlyexit_lossweights:
            loss = criterion(output, target)
            # Measure accuracy
            classerr.add(output.data, target)
            if not args.regression:
                acc_stats.append([classerr.value(1), classerr.value(min(args.num_classes, 5))])
            else:
                acc_stats.append([classerr.value()])
        else:
            # Measure accuracy and record loss
            loss = earlyexit_loss(output, target, criterion, args)
        # Record loss
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step,
                                                                  steps_per_epoch, loss,
                                                                  optimizer=optimizer,
                                                                  return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())

            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
        else:
            losses[OVERALL_LOSS_KEY].add(loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if compression_scheduler:
            compression_scheduler.before_parameter_optimization(epoch, train_step,
                                                                steps_per_epoch, optimizer)
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        # Reset elastic sampling wrt NAS stage if necessary
        if args.nas:
            if stage == 1:
                ai8x_nas.reset_kernel_sampling(model)
            elif stage == 2:
                ai8x_nas.reset_depth_sampling(model)
            elif stage == 3:
                ai8x_nas.reset_width_sampling(model)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0 or steps_completed == steps_per_epoch:
            # Log some statistics
            errs = OrderedDict()
            if not args.earlyexit_lossweights:
                if not args.regression:
                    errs['Top1'] = classerr.value(1)
                    if args.num_classes > 5:
                        errs['Top5'] = classerr.value(5)
                else:
                    errs['MSE'] = classerr.value()
            else:
                # for Early Exit case, the Top1 and Top5 stats are computed for each exit.
                for exitnum in range(args.num_exits):
                    if not args.regression:
                        errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
                        if args.num_classes > 5:
                            errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)
                    else:
                        errs['MSE_exit' + str(exitnum)] = args.exiterrors[exitnum].value()

            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            stats_dict.update(errs)
            if args.nas:
                stats_dict['NAS-Stage'] = stage
                if stage != 0:
                    stats_dict['NAS-Level'] = level
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Performance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()
    return acc_stats


def update_bn_stats(train_loader, model, args):
    """Routine to update BatchNorm statistics"""
    model.train()
    for (inputs, target) in train_loader:
        inputs, target = inputs.to(args.device), target.to(args.device)
        _ = model(inputs)


def validate(val_loader, model, criterion, loggers, args, epoch=-1, tflogger=None):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch, tflogger)


def test(test_loader, model, criterion, loggers, activations_collectors, args):
    """Model Test"""
    msglogger.info('--- test ---------------------')
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)
    with collectors_context(activations_collectors["test"]) as collectors:
        top1, top5, losses = _validate(test_loader, model, criterion, loggers, args)
        distiller.log_activation_statistics(-1, "test", loggers, collector=collectors['sparsity'])

        if args.kernel_stats:
            print("==> Kernel Stats")
            with torch.no_grad():
                global weight_min, weight_max, weight_count  # pylint: disable=global-statement
                global weight_sum, weight_stddev, weight_mean  # pylint: disable=global-statement
                weight_min = torch.tensor(float('inf'))  # pylint: disable=not-callable
                weight_max = torch.tensor(float('-inf'))  # pylint: disable=not-callable
                weight_count = torch.tensor(0, dtype=torch.int)  # pylint: disable=not-callable
                weight_sum = torch.tensor(0.0)  # pylint: disable=not-callable
                weight_stddev = torch.tensor(0.0)  # pylint: disable=not-callable

                def traverse_pass1(m):
                    """
                    Traverse model to build weight stats
                    """
                    global weight_min, weight_max  # pylint: disable=global-statement
                    global weight_count, weight_sum  # pylint: disable=global-statement
                    if isinstance(m, nn.Conv2d):
                        weight_min = torch.min(torch.min(m.weight), weight_min)
                        weight_max = torch.max(torch.max(m.weight), weight_max)
                        weight_count += len(m.weight.flatten())
                        weight_sum += m.weight.flatten().sum()
                        if hasattr(m, 'bias') and m.bias is not None:
                            weight_min = torch.min(torch.min(m.bias), weight_min)
                            weight_max = torch.max(torch.max(m.bias), weight_max)
                            weight_count += len(m.bias.flatten())
                            weight_sum += m.bias.flatten().sum()

                def traverse_pass2(m):
                    """
                    Traverse model to build weight stats
                    """
                    global weight_stddev, weight_mean  # pylint: disable=global-statement
                    if isinstance(m, nn.Conv2d):
                        weight_stddev += ((m.weight.flatten() - weight_mean) ** 2).sum()
                        if hasattr(m, 'bias') and m.bias is not None:
                            weight_stddev += ((m.bias.flatten() - weight_mean) ** 2).sum()

                model.apply(traverse_pass1)

                weight_mean = weight_sum / weight_count

                model.apply(traverse_pass2)

                weight_stddev = torch.sqrt(weight_stddev / weight_count)

                print(f"Total 2D kernel weights: {weight_count} --> min: {weight_min}, "
                      f"max: {weight_max}, stddev: {weight_stddev}")

        save_collectors_data(collectors, msglogger.logdir)
    return top1, top5, losses


def _validate(data_loader, model, criterion, loggers, args, epoch=-1, tflogger=None):
    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    if not args.regression:
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(args.num_classes, 5)))
    else:
        classerr = tnt.MSEMeter()

    def save_tensor(t, f, regression=True):
        """ Save tensor `t` to file handle `f` in CSV format """
        if t.dim() > 1:
            if not regression:
                t = torch.nn.functional.softmax(t, dim=1)
            np.savetxt(f, t.reshape(t.shape[0], t.shape[1], -1).cpu().numpy().mean(axis=2),
                       delimiter=",")
        else:
            if regression:
                np.savetxt(f, t.cpu().numpy(), delimiter=",")
            else:
                for _, i in enumerate(t):
                    f.write(f'{args.labels[i.int()]}\n')

    if args.csv_prefix is not None:
        with open(f'{args.csv_prefix}_ytrue.csv', 'w') as f_ytrue:
            f_ytrue.write('truth\n')
        with open(f'{args.csv_prefix}_ypred.csv', 'w') as f_ypred:
            f_ypred.write(','.join(args.labels) + '\n')
        with open(f'{args.csv_prefix}_x.csv', 'w') as f_x:
            for i in range(args.dimensions[0]-1):
                f_x.write(f'x_{i}_mean,')
            f_x.write(f'x_{args.dimensions[0]-1}_mean\n')

    if args.earlyexit_thresholds:
        # for Early Exit, we have a list of errors and losses for each of the exits.
        args.exiterrors = []
        args.losses_exits = []
        for exitnum in range(args.num_exits):
            if not args.regression:
                args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True,
                                                           topk=(1, min(args.num_classes, 5))))
            else:
                args.exiterrors.append(tnt.MSEMeter())
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exit_taken = [0] * args.num_exits

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = (total_samples + batch_size - 1) // batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    class_probs = []
    class_preds = []
    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = inputs.to(args.device), target.to(args.device)
            # compute output from model
            output = model(inputs)

            if args.generate_sample is not None:
                sample.generate(args.generate_sample, inputs, target, output, args.dataset, False)
                return .0, .0, .0

            if args.csv_prefix is not None:
                save_tensor(inputs, f_x)
                save_tensor(output, f_ypred, regression=args.regression)
                save_tensor(target, f_ytrue, regression=args.regression)

            if not args.earlyexit_thresholds:
                # compute loss
                loss = criterion(output, target)
                # measure accuracy and record loss
                losses['objective_loss'].add(loss.item())
                classerr.add(output.data, target)
                if args.display_confusion:
                    confusion.add(output.data, target)
            else:
                earlyexit_validate_loss(output, target, criterion, args)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % args.print_freq == 0 or steps_completed == total_steps:
                if args.display_prcurves and tflogger is not None:
                    class_probs_batch = [torch.nn.functional.softmax(el, dim=0) for el in output]
                    _, class_preds_batch = torch.max(output, 1)
                    class_probs.append(class_probs_batch)
                    class_preds.append(class_preds_batch)

                if not args.earlyexit_thresholds:
                    if not args.regression:
                        stats = (
                            '',
                            OrderedDict([('Loss', losses['objective_loss'].mean),
                                         ('Top1', classerr.value(1))])
                        )
                        if args.num_classes > 5:
                            stats[1]['Top5'] = classerr.value(5)
                    else:
                        stats = (
                            '',
                            OrderedDict([('Loss', losses['objective_loss'].mean),
                                         ('MSE', classerr.value())])
                        )
                else:
                    stats_dict = OrderedDict()
                    stats_dict['Test'] = validation_step
                    for exitnum in range(args.num_exits):
                        la_string = 'LossAvg' + str(exitnum)
                        stats_dict[la_string] = args.losses_exits[exitnum].mean
                        # Because of the nature of ClassErrorMeter, if an exit is never taken
                        # during the batch, then accessing the value(k) will cause a divide by
                        # zero. So we'll build the OrderedDict accordingly and we will not print
                        # for an exit error when that exit is never taken.
                        if args.exit_taken[exitnum]:
                            if not args.regression:
                                t1 = 'Top1_exit' + str(exitnum)
                                stats_dict[t1] = args.exiterrors[exitnum].value(1)
                                if args.num_classes > 5:
                                    t5 = 'Top5_exit' + str(exitnum)
                                    stats_dict[t5] = args.exiterrors[exitnum].value(5)
                            else:
                                t1 = 'MSE_exit' + str(exitnum)
                                stats_dict[t1] = args.exiterrors[exitnum].value()
                    stats = ('Performance/Validation/', stats_dict)

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)

                if args.display_prcurves and tflogger is not None:
                    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
                    test_preds = torch.cat(class_preds)
                    for i in range(args.num_classes):
                        tb_preds = test_preds == i
                        tb_probs = test_probs[:, i]
                        tflogger.tblogger.writer.add_pr_curve(str(args.labels[i]), tb_preds,
                                                              tb_probs, global_step=epoch)

                if args.display_embedding and tflogger is not None \
                   and steps_completed == total_steps:
                    def select_n_random(data, labels, features, n=100):
                        """Selects n random datapoints, their corresponding labels and features"""
                        assert len(data) == len(labels) == len(features)

                        perm = torch.randperm(len(data))
                        return data[perm][:n], labels[perm][:n], features[perm][:n]

                    # Select up to 100 random images and their target indices
                    images, labels, features = select_n_random(inputs, target, output,
                                                               n=min(100, len(inputs)))

                    # Get the class labels for each image
                    class_labels = [args.labels[lab] for lab in labels]

                    tflogger.tblogger.writer.add_embedding(
                        features,
                        metadata=class_labels,
                        label_img=args.visualize_fn(images, args),
                        global_step=epoch,
                        tag='verification/embedding'
                    )

    if args.csv_prefix is not None:
        f_ytrue.close()
        f_ypred.close()
        f_x.close()

    if not args.earlyexit_thresholds:
        if not args.regression:
            if args.num_classes > 5:
                msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                               classerr.value()[0], classerr.value()[1],
                               losses['objective_loss'].mean)
            else:
                msglogger.info('==> Top1: %.3f    Loss: %.3f\n',
                               classerr.value()[0], losses['objective_loss'].mean)
        else:
            msglogger.info('==> MSE: %.5f    Loss: %.3f\n',
                           classerr.value(), losses['objective_loss'].mean)

        if args.display_confusion:
            msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
            if tflogger is not None:
                cf = nnplot.confusion_matrix(confusion.value(), args.labels)
                tflogger.tblogger.writer.add_image('Validation/ConfusionMatrix', cf, epoch,
                                                   dataformats='HWC')
        if not args.regression:
            return classerr.value(1), classerr.value(min(args.num_classes, 5)), \
                losses['objective_loss'].mean
        # else:
        return classerr.value(), .0, \
            losses['objective_loss'].mean
    # else:
    total_top1, total_top5, losses_exits_stats = earlyexit_validate_stats(args)
    return total_top1, total_top5, losses_exits_stats[args.num_exits-1]


def update_training_scores_history(perf_scores_history, model, top1, top5, epoch, args):
    """ Update the list of top training scores achieved so far, and log the best scores so far"""

    model_sparsity, _, params_nnz_cnt = distiller.model_params_stats(model)

    if not args.regression:
        perf_scores_history.append(distiller.MutableNamedTuple({'params_nnz_cnt': -params_nnz_cnt,
                                                                'sparsity': model_sparsity,
                                                                'top1': top1, 'top5': top5,
                                                                'epoch': epoch}))
        # Keep perf_scores_history sorted from best to worst
        if not args.sparsity_perf:
            # Sort by top1 as main sort key, then sort by top5 and epoch
            perf_scores_history.sort(key=operator.attrgetter('top1', 'top5', 'epoch'),
                                     reverse=True)
        else:
            # Sort by sparsity as main sort key, then sort by top1, top5 and epoch
            perf_scores_history.sort(key=operator.attrgetter('params_nnz_cnt', 'top1',
                                                             'top5', 'epoch'),
                                     reverse=True)
        for score in perf_scores_history[:args.num_best_scores]:
            if args.num_classes > 5:
                msglogger.info('==> Best [Top1: %.3f   Top5: %.3f   Sparsity:%.2f   '
                               'Params: %d on epoch: %d]',
                               score.top1, score.top5, score.sparsity, -score.params_nnz_cnt,
                               score.epoch)
            else:
                msglogger.info('==> Best [Top1: %.3f   Sparsity:%.2f   '
                               'Params: %d on epoch: %d]',
                               score.top1, score.sparsity, -score.params_nnz_cnt,
                               score.epoch)
    else:
        perf_scores_history.append(distiller.MutableNamedTuple({'params_nnz_cnt': -params_nnz_cnt,
                                                                'sparsity': model_sparsity,
                                                                'top1': 1. - top1,
                                                                'epoch': epoch}))
        # Keep perf_scores_history sorted from best to worst
        if not args.sparsity_perf:
            # Sort by mse as main sort key, then sort by epoch
            perf_scores_history.sort(key=operator.attrgetter('top1', 'epoch'), reverse=True)
        else:
            # Sort by sparsity as main sort key, then sort by mse, and epoch
            perf_scores_history.sort(key=operator.attrgetter('params_nnz_cnt', 'top1', 'epoch'),
                                     reverse=True)
        for score in perf_scores_history[:args.num_best_scores]:
            msglogger.info('==> Best [MSE: %.5f   Sparsity:%.2f   '
                           'Params: %d on epoch: %d]',
                           1. - score.top1, score.sparsity, -score.params_nnz_cnt,
                           score.epoch)


def earlyexit_loss(output, target, criterion, args):
    """earlyexit_loss"""
    loss = 0
    sum_lossweights = 0
    for exitnum in range(args.num_exits-1):
        loss += (args.earlyexit_lossweights[exitnum] * criterion(output[exitnum], target))
        sum_lossweights += args.earlyexit_lossweights[exitnum]
        args.exiterrors[exitnum].add(output[exitnum].data, target)
    # handle final exit
    loss += (1.0 - sum_lossweights) * criterion(output[args.num_exits-1], target)
    args.exiterrors[args.num_exits-1].add(output[args.num_exits-1].data, target)
    return loss


def earlyexit_validate_loss(output, target, _criterion, args):
    """
    We need to go through each sample in the batch itself - in other words, we are
    not doing batch processing for exit criteria - we do this as though it were batchsize of 1
    but with a grouping of samples equal to the batch size.
    Note that final group might not be a full batch - so determine actual size.
    """
    this_batch_size = target.size()[0]
    earlyexit_validate_criterion = nn.CrossEntropyLoss(reduce=False).to(args.device)

    for exitnum in range(args.num_exits):
        # calculate losses at each sample separately in the minibatch.
        args.loss_exits[exitnum] = earlyexit_validate_criterion(output[exitnum], target)
        # for batch_size > 1, we need to reduce this down to an average over the batch
        args.losses_exits[exitnum].add(torch.mean(args.loss_exits[exitnum]).cpu())

    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(args.num_exits - 1):
            if args.loss_exits[exitnum][batch_index] < args.earlyexit_thresholds[exitnum]:
                # take the results from early exit since lower than threshold
                args.exiterrors[exitnum].add(
                    torch.tensor(  # pylint: disable=not-callable
                        np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)
                    ),
                    torch.full([1], target[batch_index], dtype=torch.long))
                args.exit_taken[exitnum] += 1
                earlyexit_taken = True
                break  # since exit was taken, do not affect the stats of subsequent exits
        # this sample does not exit early and therefore continues until final exit
        if not earlyexit_taken:
            exitnum = args.num_exits - 1
            args.exiterrors[exitnum].add(
                torch.tensor(  # pylint: disable=not-callable
                    np.array(output[exitnum].data[batch_index].cpu(), ndmin=2)
                ),
                torch.full([1], target[batch_index], dtype=torch.long))
            args.exit_taken[exitnum] += 1


def earlyexit_validate_stats(args):
    """Print some interesting summary stats for number of data points that could exit early"""
    top1k_stats = [0] * args.num_exits
    top5k_stats = [0] * args.num_exits
    losses_exits_stats = [0] * args.num_exits
    sum_exit_stats = 0
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            sum_exit_stats += args.exit_taken[exitnum]
            msglogger.info("Exit %d: %d", exitnum, args.exit_taken[exitnum])
            if not args.regression:
                top1k_stats[exitnum] += args.exiterrors[exitnum].value(1)
                top5k_stats[exitnum] += args.exiterrors[exitnum].value(
                    min(args.num_classes, 5)
                )
            else:
                top1k_stats[exitnum] += args.exiterrors[exitnum].value()
            losses_exits_stats[exitnum] += args.losses_exits[exitnum].mean
    for exitnum in range(args.num_exits):
        if args.exit_taken[exitnum]:
            msglogger.info("Percent Early Exit %d: %.3f", exitnum,
                           (args.exit_taken[exitnum]*100.0) / sum_exit_stats)
    total_top1 = 0
    total_top5 = 0
    for exitnum in range(args.num_exits):
        total_top1 += (top1k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
        if not args.regression:
            total_top5 += (top5k_stats[exitnum] * (args.exit_taken[exitnum] / sum_exit_stats))
            msglogger.info("Accuracy Stats for exit %d: top1 = %.3f, top5 = %.3f", exitnum,
                           top1k_stats[exitnum], top5k_stats[exitnum])
        else:
            msglogger.info("Accuracy Stats for exit %d: top1 = %.3f", exitnum,
                           top1k_stats[exitnum])
    msglogger.info("Totals for entire network with early exits: top1 = %.3f, top5 = %.3f",
                   total_top1, total_top5)
    return total_top1, total_top5, losses_exits_stats


def evaluate_model(model, criterion, test_loader, loggers, activations_collectors, args,
                   scheduler=None):
    """
    This sample application can be invoked to evaluate the accuracy of your model on
    the test dataset.
    You can optionally quantize the model to 8-bit integer before evaluation.
    For example:
    python3 compress_classifier.py --arch resnet20_cifar \
             ../data.cifar10 -p=50 --resume-from=checkpoint.pth.tar --evaluate
    """

    if not isinstance(loggers, list):
        loggers = [loggers]

    if args.quantize_eval:
        model.cpu()
        # if args.ai84:
        #     quantizer = PostTrainLinearQuantizerAI84.from_args(model, args)
        # else:
        quantizer = PostTrainLinearQuantizer.from_args(model, args)
        quantizer.prepare_model()
        model.to(args.device)

    top1, _, _ = test(test_loader, model, criterion, loggers, activations_collectors, args=args)

    if args.shap > 0:
        matplotlib.use('TkAgg')
        print("Generating plot...")
        images, _ = iter(test_loader).next()
        background = images[:100]
        test_images = images[100:100 + args.shap]

        # pylint: disable=protected-access
        shap.explainers._deep.deep_pytorch.op_handler['Clamp'] = \
            shap.explainers._deep.deep_pytorch.passthrough
        shap.explainers._deep.deep_pytorch.op_handler['Empty'] = \
            shap.explainers._deep.deep_pytorch.passthrough
        shap.explainers._deep.deep_pytorch.op_handler['Floor'] = \
            shap.explainers._deep.deep_pytorch.passthrough
        shap.explainers._deep.deep_pytorch.op_handler['Quantize'] = \
            shap.explainers._deep.deep_pytorch.passthrough
        shap.explainers._deep.deep_pytorch.op_handler['Scaler'] = \
            shap.explainers._deep.deep_pytorch.passthrough
        # pylint: enable=protected-access
        e = shap.DeepExplainer(model.to(args.device), background.to(args.device))
        shap_values = e.shap_values(test_images.to(args.device))
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
        # Plot the feature attributions
        shap.image_plot(shap_numpy, -test_numpy)

    if args.quantize_eval:
        checkpoint_name = 'quantized'
        apputils.save_checkpoint(0, args.cnn, model, optimizer=None, scheduler=scheduler,
                                 name='_'.join([args.name, checkpoint_name])
                                 if args.name else checkpoint_name,
                                 dir=msglogger.logdir, extras={'quantized_top1': top1})


def summarize_model(model, dataset, which_summary, filename='model'):
    """summarize_model"""
    if which_summary.startswith('png'):
        model_summaries.draw_img_classifier_to_file(model, filename + '.png', dataset,
                                                    which_summary == 'png_w_params')
    elif which_summary in ['onnx', 'onnx_simplified']:
        ai8x.onnx_export_prep(model, simplify=(which_summary == 'onnx_simplified'))
        model_summaries.export_img_classifier_to_onnx(
            model,
            filename + '.onnx',
            dataset,
            add_softmax=False,
            opset_version=13,
        )
    else:
        distiller.model_summary(model, which_summary, dataset)


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    """
    This sample application can be invoked to execute Sensitivity Analysis on your
    model.  The ouptut is saved to CSV and PNG.
    """
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity)
    distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
    distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')


def automated_deep_compression(model, criterion, _optimizer, loggers, args):
    """automated_deep_compression"""
    train_loader, _val_loader, test_loader, _ = apputils.get_data_loaders(
        args.datasets_fn, (os.path.expanduser(args.data), args), args.batch_size,
        args.workers, args.validation_split, args.deterministic,
        args.effective_train_size, args.effective_valid_size, args.effective_test_size)

    args.display_confusion = True
    validate_fn = partial(test, test_loader=test_loader, criterion=criterion,
                          loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(train, train_loader=train_loader, criterion=criterion,
                       loggers=loggers, args=args)

    save_checkpoint_fn = partial(apputils.save_checkpoint, arch=args.cnn, dir=msglogger.logdir)
    optimizer_data = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}
    adc.amc.train_auto_compressor(model, args, optimizer_data,
                                  validate_fn, save_checkpoint_fn, train_fn)


def greedy(model, criterion, _optimizer, loggers, args):
    """greedy"""
    train_loader, _val_loader, test_loader, _ = apputils.get_data_loaders(
        args.datasets_fn, (os.path.expanduser(args.data), args), args.batch_size,
        args.workers, args.validation_split, args.deterministic,
        args.effective_train_size, args.effective_valid_size, args.effective_test_size)

    test_fn = partial(test, test_loader=test_loader, criterion=criterion,
                      loggers=loggers, args=args, activations_collectors=None)
    train_fn = partial(train, train_loader=train_loader, criterion=criterion, args=args)
    assert args.greedy_target_density is not None
    distiller.pruning.greedy_filter_pruning.greedy_pruner(model, args,
                                                          args.greedy_target_density,
                                                          args.greedy_pruning_step,
                                                          test_fn, train_fn)


def create_nas_training_stage_list(model, nas_policy):
    """Create list to define NAS stage transition epochs"""
    stage_transition_list = []
    msglogger.info('NAS Policy: %s', nas_policy)

    stage_transition_list.append((nas_policy['start_epoch'], 0, 0))

    max_kernel_level = model.get_max_elastic_kernel_level()
    if nas_policy['elastic_kernel']['leveling']:
        for level in range(max_kernel_level):
            stage_transition_list.append((stage_transition_list[-1][0] +
                                          nas_policy['elastic_kernel']['num_epochs'], 1, level+1))
    else:
        stage_transition_list.append((stage_transition_list[-1][0] +
                                      nas_policy['elastic_kernel']['num_epochs'], 1,
                                      max_kernel_level))

    max_depth_level = model.get_max_elastic_depth_level()
    if nas_policy['elastic_depth']['leveling']:
        for level in range(max_depth_level):
            stage_transition_list.append((stage_transition_list[-1][0] +
                                          nas_policy['elastic_depth']['num_epochs'], 2, level+1))
    else:
        stage_transition_list.append((stage_transition_list[-1][0] +
                                      nas_policy['elastic_depth']['num_epochs'], 2,
                                      max_depth_level))

    max_width_level = model.get_max_elastic_width_level()
    if nas_policy['elastic_width']['leveling']:
        for level in range(max_width_level):
            stage_transition_list.append((stage_transition_list[-1][0] +
                                          nas_policy['elastic_width']['num_epochs'], 3, level+1))
    else:
        stage_transition_list.append((stage_transition_list[-1][0] +
                                      nas_policy['elastic_width']['num_epochs'], 3,
                                      max_width_level))

    return stage_transition_list


def get_nas_training_stage(epoch, stage_transition_list):
    """Returns current stage of NAS"""
    for t in stage_transition_list:
        if epoch < t[0]:
            break

    return t[1], t[2]  # pylint: disable=undefined-loop-variable


def get_next_stage_start_epoch(epoch, stage_transition_list, num_epochs):
    """Returns the starting epoch of the following stage"""
    current_stage = None
    for stg_idx, t in enumerate(stage_transition_list):
        if epoch < t[0]:
            if current_stage is None:
                current_stage = t[1]
            if stg_idx == len(stage_transition_list)-1:
                return num_epochs
            if current_stage != stage_transition_list[stg_idx+1][1]:
                return t[0]
    return num_epochs


class missingdict(dict):
    """This is a little trick to prevent KeyError"""
    def __missing__(self, key):
        return None  # note, does *not* set self[key] - we don't want defaultdict's behavior


def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    distiller.utils.assign_layer_fq_names(model)

    genCollectors = lambda: missingdict({  # noqa E731
        "sparsity":      SummaryActivationStatsCollector(model, "sparsity",
                                                         lambda t:
                                                         100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.
                                                         activation_channels_means),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


def create_quantization_stats_collector(model):
    """create_quantization_stats_collector"""
    distiller.utils.assign_layer_fq_names(model)
    return {'test': missingdict({
        'quantization_stats': QuantCalibrationStatsCollector(model, classes=None,
                                                             inplace_runtime_check=True,
                                                             disable_inplace_attrs=True)})}


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to Excel workbooks
    """
    for name, collector in collectors.items():
        workbook = os.path.join(directory, name)
        msglogger.info("Generating %s", workbook)
        collector.save(workbook)


def check_pytorch_version():
    """Ensure PyTorch >= 1.5.0"""
    if parse_version(torch.__version__) < parse_version('1.5.0'):
        print("\nNOTICE:")
        print("This software requires at least PyTorch version 1.5.0 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 1.5.0 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        sys.exit(1)


def update_old_model_params(model_path, model_new):
    """Adds missing model parameters added with default values.
    This is mainly due to the saved checkpoint is from previous versions of the repo.
    New model is saved to `model_path` and the old model copied into the same file_path with
    `__obselete__` prefix."""
    is_model_old = False
    model_old = torch.load(model_path)
    for new_key, new_val in model_new.state_dict().items():
        if new_key not in model_old['state_dict'] and 'bn' not in new_key:
            is_model_old = True
            model_old['state_dict'][new_key] = new_val
            if 'compression_sched' in model_old:
                if 'masks_dict' in model_old['compression_sched']:
                    model_old['compression_sched']['masks_dict'][new_key] = None

    if is_model_old:
        dir_path, file_name = os.path.split(model_path)
        new_file_name = '__obselete__' + file_name
        old_model_path = os.path.join(dir_path, new_file_name)
        os.rename(model_path, old_model_path)
        torch.save(model_old, model_path)
        msglogger.info('Model `%s` is old. Missing parameters added with default values!',
                       model_path)


if __name__ == '__main__':
    try:
        check_pytorch_version()
        np.set_printoptions(threshold=np.inf, linewidth=190)
        torch.set_printoptions(threshold=np.inf, linewidth=190)

        # import builtins, sys
        # print(distiller.config.__name__)
        # print(distiller.config.__builtins__)
        # # print(distiller.config.__import__)
        # builtins.QuantAwareTrainRangeLinearQuantizerAI84 = \
        #   range_linear_ai84.QuantAwareTrainRangeLinearQuantizerAI84
        # globals()['range_linear_ai84'] = __import__('range_linear_ai84')
        # sys.path.append('/home/robertmuchsel/Documents/Source/ai84')
        # site.addsitedir("/home/robertmuchsel/Documents/Source/ai84")
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in
            # stdout - once from the logging operation and once from re-raising the exception.
            # So we remove the stdout logging handler before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers
                                  if not isinstance(h, logging.StreamHandler)]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: %s', os.path.realpath(msglogger.log_filename))
