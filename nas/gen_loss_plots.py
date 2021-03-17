#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""Plots the losses from the given log file"""

import argparse
import os

import numpy as np

import matplotlib.pyplot as plt


def get_log_file(log_file_dir):
    """Returns the path of the log file in the given log directory"""
    for filename in os.listdir(log_file_dir):
        name, extension = os.path.splitext(filename)
        file_path = os.path.join(log_file_dir, name + extension)
        if os.path.isfile(file_path) and extension == '.log':
            return file_path, name

    return None, None


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description='Loss Plotter')
    parser.add_argument('-l', '--log-dir', action='append', required=True,
                        help='path to log folders')
    return parser.parse_args()


def prepare_loss_plot():
    """Prepares the data for the plot"""
    args = parse_args()

    epoch_list = []
    tr_loss_list = []
    tr_time_list = []
    ofa_stage_list = []
    ofa_level_list = []
    log_dir = None

    for log_dir in args.log_dir:
        log_file_path, log_file_name = get_log_file(log_dir)

        epoch = -1

        training_info_line = ''
        temp_line = None
        cur_line = None

        with open(log_file_path, 'r') as f:
            while True:
                if cur_line is not None:
                    if temp_line is not None:
                        training_info_line = temp_line
                    temp_line = cur_line

                cur_line = f.readline()
                if not cur_line:
                    break

                if cur_line == 'Parameters:\n':
                    epoch_start_idx = training_info_line.find('Epoch: [') + 8
                    epoch_end_idx = epoch_start_idx + \
                        training_info_line[epoch_start_idx:].find(']')
                    epoch = int(training_info_line[epoch_start_idx:epoch_end_idx])
                    if epoch in epoch_list:
                        idx = epoch_list.index(epoch)
                        epoch_list = epoch_list[:idx]
                        tr_loss_list = tr_loss_list[:idx]
                        tr_time_list = tr_time_list[:idx]
                        ofa_stage_list = ofa_stage_list[:idx]
                        ofa_level_list = ofa_level_list[:idx]

                    epoch_list.append(epoch)

                    batch_size_start_idx = training_info_line.find('/') + 2
                    batch_size_end_idx = batch_size_start_idx + \
                        training_info_line[batch_size_start_idx:].find(']')
                    batch_size = int(training_info_line[batch_size_start_idx:batch_size_end_idx])

                    tr_loss_start_idx = training_info_line.find('Objective Loss') + 15
                    tr_loss_end_idx = tr_loss_start_idx + \
                        training_info_line[tr_loss_start_idx:].find('    ')
                    tr_loss = float(training_info_line[tr_loss_start_idx:tr_loss_end_idx])
                    tr_loss_list.append(tr_loss)

                    tr_time_start_idx = training_info_line.find('Time') + 5
                    tr_time_end_idx = tr_time_start_idx + \
                        training_info_line[tr_time_start_idx:].find('    ')
                    tr_time = float(training_info_line[tr_time_start_idx:tr_time_end_idx])
                    if len(tr_time_list) == 0:
                        tr_time_list.append(batch_size*tr_time/3600)
                    else:
                        tr_time_list.append(batch_size*tr_time/3600 + tr_time_list[-1])

                    ofa_stage_start_idx = training_info_line.find('OFA-Stage ') + 10
                    ofa_stage_end_idx = ofa_stage_start_idx + 1
                    ofa_stage = int(training_info_line[ofa_stage_start_idx:ofa_stage_end_idx])
                    ofa_stage_list.append(ofa_stage)

                    ofa_level_start_idx = training_info_line.find('OFA-Level ') + 10
                    ofa_level_end_idx = ofa_level_start_idx + 1
                    ofa_level = int(training_info_line[ofa_level_start_idx:ofa_level_end_idx])
                    ofa_level_list.append(ofa_level)

    stage_ch_idx = np.where(np.diff(ofa_stage_list) != 0)[0]
    level_ch_idx = np.where(np.diff(ofa_level_list) != 0)[0]
    level_ch_idx = np.setdiff1d(level_ch_idx, stage_ch_idx)

    if stage_ch_idx.size > 0:
        stage_ch_idx += 1

    if level_ch_idx.size > 0:
        level_ch_idx += 1

    def forward(x):
        return np.interp(x, epoch_list, tr_time_list)

    def inverse(x):
        return np.interp(x, tr_time_list, epoch_list)

    dpi = 96
    f, ax = plt.subplots(1, figsize=(1280/dpi, 800/dpi), dpi=dpi)

    ax.plot(epoch_list, tr_loss_list)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Objective Loss')

    secax = ax.secondary_xaxis('top', functions=(forward, inverse))
    secax.set_xlabel('Time (hours)')

    y_range = max(tr_loss_list) - min(tr_loss_list)
    y_min = min(tr_loss_list)
    y_max = max(tr_loss_list)

    ax_y_min = y_min - 0.15*y_range
    ax_y_max = y_max + 0.15*y_range

    for stage_idx, ep_idx in enumerate(stage_ch_idx):
        ax.axvline(epoch_list[ep_idx], ymin=-0.5, ymax=1.05, color='r', linestyle='--', alpha=0.5)
        ep_min = epoch_list[ep_idx]
        ep_max = epoch_list[-1]
        if stage_idx != (len(stage_ch_idx) - 1):
            ep_max = epoch_list[stage_ch_idx[stage_idx+1]]
        ax.annotate('', xy=(ep_min+1, y_min-0.1*y_range), xycoords='data',
                    xytext=(ep_max-1, y_min-0.1*y_range), textcoords='data',
                    arrowprops={'arrowstyle': '<->'})
        ax.annotate(f'Stage-{stage_idx+1}', xy=(0.5*(ep_min+ep_max), y_min-0.09*y_range),
                    xycoords='data', ha='center', va='bottom', color='k')

    for ep_idx in level_ch_idx:
        ax.axvline(epoch_list[ep_idx], ymin=-0.5, ymax=1.05, color='g', linestyle='--', alpha=0.5)

    ax.set_ylim([ax_y_min, ax_y_max])
    ax.grid(True, zorder=55)
    if log_dir and log_dir != '':
        plt.savefig(os.path.join(log_dir, log_file_name + '.png'))


if __name__ == '__main__':
    prepare_loss_plot()
