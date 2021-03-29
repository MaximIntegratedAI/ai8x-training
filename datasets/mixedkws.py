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
"""
Classes and functions used to create noisy keyword spotting dataset.
"""
import errno
import os

import numpy as np
import torch
from torchvision import transforms

import ai8x

from .kws20 import KWS_35_get_unquantized_datasets
from .msnoise import MSnoise_get_unquantized_datasets


class MixedKWS:
    """
    Dataset for adding noise to SpeechCom dataset, 1D folded.

    Args:
    root (string): Root directory of dataset where ``KWS/processed/dataset.pt``
        exist.
    classes(array): List of keywords to be used.
    d_type(string): Option for the created dataset. ``train`` or ``test``.
    snr(int): Signal-to-noise ratio for the train and test set
    n_augment(int, optional): Number of augmented samples added to the dataset from
        each sample by random modifications, i.e. stretching, shifting and random noise.
    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version.
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.

    """

    class_dict = {'backward': 0, 'bed': 1, 'bird': 2, 'cat': 3, 'dog': 4, 'down': 5,
                  'eight': 6, 'five': 7, 'follow': 8, 'forward': 9, 'four': 10, 'go': 11,
                  'happy': 12, 'house': 13, 'learn': 14, 'left': 15, 'marvin': 16, 'nine': 17,
                  'no': 18, 'off': 19, 'on': 20, 'one': 21, 'right': 22, 'seven': 23,
                  'sheila': 24, 'six': 25, 'stop': 26, 'three': 27, 'tree': 28, 'two': 29,
                  'up': 30, 'visual': 31, 'wow': 32, 'yes': 33, 'zero': 34}

    def __init__(self, root, classes, d_type, snr, n_augment=3,
                 transform=None, quantization_scheme=None, download=False):

        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.snr = snr
        self.n_augment = n_augment
        self.transform = transform

        self.save_unquantized = False
        self.__parse_quantization(quantization_scheme)

        if self.save_unquantized:
            self.data_file = f'dataset_unquantized_{str(self.snr)}dB.pt'
        else:
            self.data_file = f'dataset_quantized_{str(self.snr)}dB.pt'

        if download:
            self.__download()

        self.data, self.targets, self.data_type = torch.load(os.path.join(
            self.processed_folder, self.data_file))

        self.__filter_dtype()
        self.__filter_classes()

    def __download(self):

        if self.__check_exists():
            return

        self.__makedir_exist_ok(self.processed_folder)

        self.__gen_datasets()

    def __check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.data_file))

    def __makedir_exist_ok(self, dirpath):  # pylint: disable=no-self-use
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def __filter_dtype(self):
        if self.d_type == 'train':
            idx_to_select = (self.data_type == 0)[:, -1]
        elif self.d_type == 'test':
            idx_to_select = (self.data_type == 1)[:, -1]
        else:
            print('Unknown data type: %s' % self.d_type)
            return

        print(self.data.shape)
        self.data = self.data[idx_to_select, :]
        self.targets = self.targets[idx_to_select, :]
        del self.data_type

    def __filter_classes(self):
        print('\n')
        initial_new_class_label = len(self.class_dict)
        new_class_label = initial_new_class_label
        for c in self.classes:
            if c not in self.class_dict.keys():
                print('Class is not in the data: %s' % c)
                return
            # else:
            print('Class %s, %d' % (c, self.class_dict[c]))
            num_elems = (self.targets == self.class_dict[c]).cpu().sum()
            print('Number of elements in class %s: %d' % (c, num_elems))
            self.targets[(self.targets == self.class_dict[c])] = new_class_label
            new_class_label += 1

        num_elems = (self.targets < initial_new_class_label).cpu().sum()
        print('Number of elements in class unknown: %d' % (num_elems))
        self.targets[(self.targets < initial_new_class_label)] = new_class_label
        self.targets -= initial_new_class_label
        print(np.unique(self.targets.data.cpu()))
        print('\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inp, target = self.data[index].type(torch.FloatTensor), int(self.targets[index])
        if not self.save_unquantized:
            inp /= 256
        if self.transform is not None:
            inp = self.transform(inp)
        return inp, target

    @property
    def raw_folder(self):
        """Folder for the raw data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        """Folder for the processed data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def __parse_quantization(self, quantization_scheme):
        if quantization_scheme:
            self.quantization = quantization_scheme
            if 'bits' not in self.quantization:
                self.quantization['bits'] = 8
            if self.quantization['bits'] == 0:
                self.save_unquantized = True
            if 'compand' not in self.quantization:
                self.quantization['compand'] = False
            elif 'mu' not in self.quantization:
                self.quantization['mu'] = 255
        else:
            print('No define quantization schema!, ',
                  'Number of bits set to 8.')
            self.quantization = {'bits': 8, 'compand': False}

    @staticmethod
    def compand(data, mu=255):
        """Compand the signal level to warp from Laplacian distribution to uniform distribution"""
        data = np.sign(data) * np.log(1 + mu*np.abs(data)) / np.log(1 + mu)
        return data

    @staticmethod
    def expand(data, mu=255):
        """Undo the companding"""
        data = np.sign(data) * (1 / mu) * (np.power((1 + mu), np.abs(data)) - 1)
        return data

    @staticmethod
    def quantize_audio(data, num_bits=8, compand=False, mu=255):
        """Quantize audio"""
        if compand:
            data = MixedKWS.compand(data, mu)

        step_size = 2.0 / 2**(num_bits)
        max_val = 2**(num_bits) - 1
        q_data = np.round((data - (-1.0)) / step_size)
        q_data = np.clip(q_data, 0, max_val)

        if compand:
            data_ex = (q_data - 2**(num_bits - 1)) / 2**(num_bits - 1)
            data_ex = MixedKWS.expand(data_ex)
            q_data = np.round((data_ex - (-1.0)) / step_size)
            q_data = np.clip(q_data, 0, max_val)
        return np.uint8(q_data)

    @staticmethod
    def __snr_mixer(clean, noise, snr):
        # Normalizing to rms equal to 1
        rmsclean = np.mean(clean[:, :125]**2)**0.5
        scalarclean = 1 / rmsclean
        clean = clean * scalarclean

        rmsnoise = np.mean(noise[:, :125]**2)**0.5
        scalarnoise = 1 / rmsnoise
        noise = noise * scalarnoise

        # Set the noise level for a given SNR
        cleanfactor = 10**(snr/20)
        noisyspeech = cleanfactor*clean + noise
        noisyspeech = noisyspeech / (scalarnoise + cleanfactor * scalarclean)
        return noisyspeech

    def __gen_datasets(self, exp_len=16384, row_len=128, overlap_ratio=0):

        # PARAMETERS
        overlap = int(np.ceil(row_len * overlap_ratio))
        num_rows = int(np.ceil(exp_len / (row_len - overlap)))

        class Args:
            """Args to call speech and noise datasets"""
            # pylint: disable=too-few-public-methods
            def __init__(self):
                self.truncate_testset = False
                self.act_mode_8bit = False

        args = Args()
        train_speech, test_speech = KWS_35_get_unquantized_datasets((self.root, args))
        train_noise, test_noise = MSnoise_get_unquantized_datasets((self.root, args))

        train_speech.data = train_speech.data.numpy()
        test_speech.data = test_speech.data.numpy()
        train_noise.data = train_noise.data.numpy()
        test_noise.data = test_noise.data.numpy()

        total_size = self.n_augment*(train_speech.data.shape[0] + test_speech.data.shape[0])

        if not self.save_unquantized:
            data_in = np.empty((total_size, row_len, num_rows), dtype=np.uint8)
        else:
            data_in = np.empty((total_size, row_len, num_rows), dtype=np.float32)

        data_type = np.empty((total_size, 1), dtype=np.uint8)
        data_class = np.empty((total_size, 1), dtype=np.uint8)

        speeches = [train_speech.data, test_speech.data]
        noises = [train_noise.data, test_noise.data]

        new_ind = 0
        for ind_s, speech in enumerate(speeches):
            noise = noises[ind_s]
            for i in range(speech.shape[0]):
                for _ in range(self.n_augment):
                    while True:
                        rand_ind = np.random.randint(noise.shape[0])
                        random_noise = noise[rand_ind]
                        if np.any(random_noise):
                            break

                    noisy_speech = self.__snr_mixer(speech[i, :, :],
                                                    random_noise, self.snr)
                    if not self.save_unquantized:
                        data_in[new_ind, :, :] = (self.quantize_audio(noisy_speech,
                                                  num_bits=self.quantization['bits'],
                                                  compand=self.quantization['compand'],
                                                  mu=self.quantization['mu']))
                    else:
                        data_in[new_ind, :, :] = noisy_speech

                    if ind_s == 0:
                        data_type[new_ind] = np.uint8(0)
                    elif ind_s == 1:
                        data_type[new_ind] = np.uint8(1)

                    if ind_s == 0:
                        data_class[new_ind] = np.uint8(train_speech.targets[i].item())
                    elif ind_s == 1:
                        data_class[new_ind] = np.uint8(test_speech.targets[i].item())

                    new_ind += 1

        data_in = torch.from_numpy(data_in)
        data_class = torch.from_numpy(data_class)
        data_type = torch.from_numpy(data_type)

        noise_dataset = (data_in, data_class, data_type)
        torch.save(noise_dataset, os.path.join(self.processed_folder, self.data_file))
        print('Dataset for Mixed KWS is generated!')


def MixedKWS_get_datasets(data, snr, load_train=True, load_test=True, num_classes=6):
    """
    Load the folded 1D version of SpeechCom dataset

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 30 keywords. A dataset is formed with 7 or 21 classes which
    includes 6 or 20 of the original keywords and the rest of the
    dataset is used to form the last class, i.e class of the others.
    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by random stretch/shift and randomly adding noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """
    (data_dir, args) = data

    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])

    if num_classes == 6:
        classes = ['up', 'down', 'left', 'right', 'stop', 'go']
    elif num_classes == 20:
        classes = ['up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero']
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')

    n_augment = 3
    quantization_scheme = {'compand': False, 'mu': 10}

    if load_train:
        train_dataset = MixedKWS(root=data_dir, classes=classes, d_type='train',
                                 snr=snr, n_augment=n_augment, transform=transform,
                                 quantization_scheme=quantization_scheme,
                                 download=True)

    else:
        train_dataset = None

    if load_test:
        test_dataset = MixedKWS(root=data_dir, classes=classes, d_type='test',
                                snr=snr, n_augment=n_augment, transform=transform,
                                quantization_scheme=quantization_scheme,
                                download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def MixedKWS_20_get_datasets_0dB(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MixedKWS dataset for 20 classes and 0 dB SNR

    """
    return MixedKWS_get_datasets(data, snr=0, load_train=load_train,
                                 load_test=load_test, num_classes=20)


def MixedKWS_20_get_datasets_5dB(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MixedKWS dataset for 20 classes and 5 dB SNR

    """
    return MixedKWS_get_datasets(data, snr=5, load_train=load_train,
                                 load_test=load_test, num_classes=20)


def MixedKWS_20_get_datasets_10dB(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MixedKWS dataset for 20 classes and 10 dB SNR

    """
    return MixedKWS_get_datasets(data, snr=10, load_train=load_train,
                                 load_test=load_test, num_classes=20)


def MixedKWS_20_get_datasets_15dB(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MixedKWS dataset for 20 classes and 15 dB SNR

    """
    return MixedKWS_get_datasets(data, snr=15, load_train=load_train,
                                 load_test=load_test, num_classes=20)


def MixedKWS_20_get_datasets_20dB(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MixedKWS dataset for 20 classes and 20 dB SNR

    """
    return MixedKWS_get_datasets(data, snr=20, load_train=load_train,
                                 load_test=load_test, num_classes=20)


def MixedKWS_20_get_datasets_25dB(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MixedKWS dataset for 20 classes and 20 dB SNR

    """
    return MixedKWS_get_datasets(data, snr=25, load_train=load_train,
                                 load_test=load_test, num_classes=20)


def MixedKWS_20_get_datasets_30dB(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MixedKWS dataset for 20 classes and 20 dB SNR

    """
    return MixedKWS_get_datasets(data, snr=30, load_train=load_train,
                                 load_test=load_test, num_classes=20)


def MixedKWS_20_get_datasets_100dB(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MixedKWS dataset for 20 classes and 20 dB SNR

    """
    return MixedKWS_get_datasets(data, snr=100, load_train=load_train,
                                 load_test=load_test, num_classes=20)


datasets = [
    {
        'name': 'MixedKWS20_0dB',  # 20 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': MixedKWS_20_get_datasets_0dB,
    },
    {
        'name': 'MixedKWS20_5dB',  # 20 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': MixedKWS_20_get_datasets_5dB,
    },
    {
        'name': 'MixedKWS20_10dB',  # 20 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': MixedKWS_20_get_datasets_10dB,
    },
    {
        'name': 'MixedKWS20_15dB',  # 20 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': MixedKWS_20_get_datasets_15dB,
    },
    {
        'name': 'MixedKWS20_20dB',  # 20 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': MixedKWS_20_get_datasets_20dB,
    },
    {
        'name': 'MixedKWS20_25dB',  # 20 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': MixedKWS_20_get_datasets_25dB,
    },
    {
        'name': 'MixedKWS20_30dB',  # 20 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': MixedKWS_20_get_datasets_30dB,
    },
    {
        'name': 'MixedKWS20_100dB',  # 20 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': MixedKWS_20_get_datasets_100dB,
    },
]
