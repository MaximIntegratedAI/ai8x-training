#
# Copyright (c) 2018 Intel Corporation
# Portions Copyright (C) 2019-2023 Maxim Integrated Products, Inc.
# Portions Copyright (C) 2023-2024 Analog Devices, Inc.
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
Classes and functions used to create noisy keyword spotting dataset.
"""
import numpy as np
import torch


class signalmixer:
    """
    Signal mixer dataloader to create datasets with specified
    length using a noise dataset and a speech dataset and a specified SNR level.

    Args:
    signal_dataset(object): Dataset object.
    snr_range(list,int): SNR level range to be used in the snr mixing operation.
    noise_type(string): Noise kind that will be applied to the speech dataset.
    apply_prob(int): Probability of noise that will be applied to speech dataset.
    noise_dataset(object, optional): Noise dataset object.
    """

    def __init__(self, signal_dataset, snr_range, noise_type, apply_prob=1, noise_dataset=None):

        self.data = signal_dataset.data
        self.targets = signal_dataset.targets
        self.signal_dataset = signal_dataset
        self.data_type = signal_dataset.data_type

        if noise_type != ('WhiteNoise' or None):
            self.noise_dataset = noise_dataset

        self.snr_range = snr_range

        if apply_prob > 1:
            print('Noise will be applied to the whole dataset! \n')
            self.apply_prob = 1
        else:
            self.apply_prob = apply_prob
        self.noise_type = noise_type

    def __getitem__(self, index):
        inp, target = self.signal_dataset[index]
        data_type = self.data_type[index]

        # applying noise to train set using per-snr
        if data_type == 0:
            if np.random.uniform(0, 1) < self.apply_prob:
                if self.noise_type is not None:
                    if not torch.all(inp == 0):
                        inp = self.snr_mixer(inp)

        # applying noise to all of the test set
        else:
            if 'WhiteNoise' in self.noise_type:
                if not torch.all(inp == 0):
                    inp = self.white_noise_mixer(inp)
            elif self.noise_type is not None:
                if not torch.all(inp == 0):
                    inp = self.snr_mixer(inp)
        return inp.type(torch.FloatTensor), int(target)

    def __len__(self):
        return len(self.data)

    def snr_mixer(self, signal):
        '''
        Creates mixed signals using an SNR level and the noise dataset
        '''
        idx_noise = np.random.randint(0, len(self.data))
        noise, _ = self.noise_dataset[idx_noise]
        rms_noise = self.noise_dataset.rms[idx_noise]

        if self.snr_range[0] == self.snr_range[-1]:
            snr = self.snr_range[0]
        else:
            snr = np.random.uniform(self.snr_range[0], self.snr_range[-1])
        snr = torch.tensor(snr)

        rmsclean = torch.sqrt(torch.mean(signal**2))
        scalarclean = 1 / rmsclean
        signal = signal * scalarclean

        scalarnoise = 1 / rms_noise
        noise = noise * scalarnoise

        cleanfactor = 10**(snr/20)
        noisyspeech = cleanfactor * signal + noise
        noisyspeech = noisyspeech / (torch.tensor(scalarnoise) + cleanfactor * scalarclean)

        return noisyspeech

    def white_noise_mixer(self, signal):

        '''Creates White Noise and apply it to the signal using the specified SNR level.
        Returns the mixed signal with white noise.
        '''
        if self.snr_range[0] == self.snr_range[-1]:
            snr = self.snr_range[0]
        else:
            snr = np.random.uniform(self.snr_range[0], self.snr_range[-1])
        snr = torch.tensor(snr)

        mean = 0
        std = 1
        noise = np.random.normal(mean, std, signal.shape)
        noise = torch.tensor(noise, dtype=torch.float32)

        rmsclean = torch.sqrt(torch.mean(signal**2))
        scalarclean = 1 / rmsclean
        signal = signal * scalarclean

        rmsnoise = torch.sqrt(torch.mean(noise**2))
        scalarnoise = 1 / rmsnoise
        noise = noise * scalarnoise

        cleanfactor = 10**(snr/20)
        noisyspeech = cleanfactor * signal + noise
        noisyspeech = noisyspeech / (scalarnoise + cleanfactor * scalarclean)

        return noisyspeech
