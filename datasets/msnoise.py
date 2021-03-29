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
Classes and functions used to create audio noise dataset.
"""
import errno
import json
import os
import sys
import warnings

import numpy as np
import torch
from torchvision import transforms

import librosa
from six.moves import urllib

import ai8x


class MSnoise:
    """
    `Microsoft Scalable Noisy Speech <https://github.com/microsoft/MS-SNSD>`
    Dataset, 1D folded.

    Args:
    root (string): Root directory of dataset where ``MSnoise/processed/dataset.pt``
        exist.
    classes(array): List of keywords to be used.
    d_type(string): Option for the created dataset. ``train`` or ``test``.
    remove_unknowns (bool, optional): If true, unchosen classes are not gathered as
        the unknown class.
    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version.
    quantize (bool, optional): If true, the datasets are prepared and saved as
        quantized. If this dataset is to be used with MixedKWS class, then this
        argument must be false.
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.

    """

    class_dict = {'AirConditioner': 0, 'AirportAnnouncements': 1,
                  'Babble': 2, 'Bus': 3, 'CafeTeria': 4, 'Car': 5,
                  'CopyMachine': 6, 'Field': 7, 'Hallway': 8, 'Kitchen': 9,
                  'LivingRoom': 10, 'Metro': 11, 'Munching': 12, 'NeighborSpeaking': 13,
                  'Office': 14, 'Park': 15, 'Restaurant': 16, 'ShuttingDoor': 17,
                  'Square': 18, 'SqueakyChair': 19, 'Station': 20, 'Traffic': 21,
                  'Typing': 22, 'VacuumCleaner': 23, 'WasherDryer': 24, 'Washing': 25}

    def __init__(self, root, classes, d_type, remove_unknowns=False,
                 transform=None, quantize=False, download=False):
        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.remove_unknowns = remove_unknowns
        self.transform = transform

        self.noise_train_folder = os.path.join(self.raw_folder, 'noise_train')
        self.noise_test_folder = os.path.join(self.raw_folder, 'noise_test')
        self.url_train = \
            'https://api.github.com/repos/microsoft/MS-SNSD/contents/noise_train?ref=master'
        self.url_test = \
            'https://api.github.com/repos/microsoft/MS-SNSD/contents/noise_test?ref=master'
        self.quantize = quantize

        if self.quantize:
            self.data_file = 'dataset_quantized.pt'
        else:
            self.data_file = 'dataset_unquantized.pt'

        if download:
            self.__download()

        self.data, self.targets, self.data_type = torch.load(os.path.join(
            self.processed_folder, self.data_file))

        self.__filter_dtype()
        self.__filter_classes()

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

    def __download(self):

        if self.__check_exists():
            return

        self.__makedir_exist_ok(self.noise_train_folder)
        self.__makedir_exist_ok(self.noise_test_folder)
        self.__makedir_exist_ok(self.processed_folder)

        self.__download_raw(self.url_train)
        self.__download_raw(self.url_test)
        self.__gen_datasets()

    def __download_raw(self, api_url):
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        response = urllib.request.urlretrieve(api_url)

        total_files = 0
        with open(response[0], "r") as f:
            data = json.load(f)
            total_files += len(data)
            for file in data:
                file_url = file["download_url"]
                file_name = file["name"]
                path = file["path"]
                path = os.path.join(self.raw_folder, path)
                try:
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)
                    urllib.request.urlretrieve(file_url, path)
                    print(f'Downloaded: {file_name}')
                except KeyboardInterrupt:
                    print('Interrupted while downloading!')
                    sys.exit()

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
        if self.remove_unknowns:
            idx_to_remove = (self.targets == new_class_label)[:, -1]
            idx_to_keep = torch.logical_not(idx_to_remove)
            self.data = self.data[idx_to_keep, :]
            self.targets = self.targets[idx_to_keep, :]
        self.targets -= initial_new_class_label
        print(np.unique(self.targets.data.cpu()))
        print('\n')

    @staticmethod
    def quantize_audio(data, num_bits=8):
        """Quantize audio
        """
        step_size = 2.0 / 2**(num_bits)
        max_val = 2**(num_bits) - 1
        q_data = np.round((data - (-1.0)) / step_size)
        q_data = np.clip(q_data, 0, max_val)

        return np.uint8(q_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inp, target = self.data[index].type(torch.FloatTensor), int(self.targets[index])
        if self.quantize:
            inp /= 256
        if self.transform is not None:
            inp = self.transform(inp)
        return inp, target

    def __gen_datasets(self, exp_len=16384, row_len=128, overlap_ratio=0,
                       noise_time_step=0.25, train_ratio=0.6):
        print('Generating dataset from raw data samples for the first time. ')
        print('Warning: This process could take 5-10 minutes!')
        with warnings.catch_warnings():
            warnings.simplefilter('error')

            # PARAMETERS
            overlap = int(np.ceil(row_len * overlap_ratio))
            num_rows = int(np.ceil(exp_len / (row_len - overlap)))
            data_len = int((num_rows*row_len - (num_rows-1)*overlap))
            print('data_len: %s' % data_len)

            # Cleaning the duplicate labels
            train_list = sorted(os.listdir(self.noise_train_folder))
            test_list = sorted(os.listdir(self.noise_test_folder))
            labels_train = set(sorted({i.split('_')[0] for i in train_list if '_' in i}))
            labels_test = set(sorted({i.split('_')[0] for i in test_list if '_' in i}))
            labels = labels_train | labels_test
            labels_to_remove = set()
            for label in labels:
                other_labels = labels - {label}
                for other_label_name in other_labels:
                    if label in other_label_name:
                        labels_to_remove.add(label)
                        break
            labels = labels - labels_to_remove
            labels = sorted(labels)
            print(f'Labels: {labels}')

            # Folders
            train_test_folders = [self.noise_train_folder, self.noise_test_folder]

            # Determining the array sizes
            num_seqs = 0
            for label in labels:
                train_count = 0
                test_count = 0
                for folder in train_test_folders:
                    for record_name in os.listdir(folder):
                        if record_name.split('_')[0] in label:
                            record_path = os.path.join(folder, record_name)
                            record, fs = librosa.load(record_path, offset=0, sr=None)
                            rec_len = np.size(record)
                            max_start_time = ((rec_len / fs - 1) -
                                              (rec_len / fs % noise_time_step))
                            num_seqs += int(max_start_time / noise_time_step + 1)
            print(f'Num sequences: {num_seqs}')

            # Creating the empty arrays
            if self.quantize:
                data_in = np.zeros((num_seqs, row_len, num_rows), dtype=np.uint8)
            else:
                data_in = np.zeros((num_seqs, row_len, num_rows), dtype=np.float32)
            data_type = np.zeros((num_seqs, 1), dtype=np.uint8)
            data_class = np.zeros((num_seqs, 1), dtype=np.uint8)

            data_idx = 0
            for i, label in enumerate(labels):
                print(f'Processing label:{label}')
                train_count = 0
                test_count = 0
                for folder in train_test_folders:
                    for record_name in sorted(os.listdir(folder)):
                        if record_name.split('_')[0] in label:
                            if hash(record_name) % 10 < 10*train_ratio:
                                d_type = np.uint8(0)  # train+val
                                train_count += 1
                            else:
                                d_type = np.uint8(1)  # test
                                test_count += 1
                            record_path = os.path.join(folder, record_name)
                            record, fs = librosa.load(record_path, offset=0, sr=None)
                            rec_len = np.size(record)
                            max_start_time = \
                                ((rec_len / fs - 1) - (rec_len / fs % noise_time_step))
                            for start_time in np.arange(0,
                                                        int((max_start_time+noise_time_step)*fs),
                                                        int(noise_time_step*fs)):
                                end_time = start_time + fs
                                audio_seq = record[start_time:end_time]
                                data_type[data_idx, 0] = d_type
                                data_class[data_idx, 0] = i
                                for n_r in range(num_rows):
                                    start_idx = n_r*(row_len - overlap)
                                    end_idx = start_idx + row_len
                                    audio_chunk = audio_seq[start_idx:end_idx]
                                    audio_chunk = \
                                        np.pad(audio_chunk, [0, row_len-audio_chunk.size])
                                    if self.quantize:
                                        data_in[data_idx, :, n_r] = \
                                            self.quantize_audio(audio_chunk)
                                    else:
                                        data_in[data_idx, :, n_r] = audio_chunk
                                data_idx += 1

            data_in = torch.from_numpy(data_in)
            data_class = torch.from_numpy(data_class)
            data_type = torch.from_numpy(data_type)

            noise_dataset = (data_in, data_class, data_type)
            torch.save(noise_dataset, os.path.join(self.processed_folder, self.data_file))
        print('Dataset created!')


def MSnoise_get_datasets(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MS Scalable Noisy Speech dataset (MS-SNSD)

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 26 different noise types.
    15 of them are chosen classification, others are labeled as the unkown class.
    """
    (data_dir, args) = data

    classes = ['AirConditioner', 'AirportAnnouncements',
               'Babble', 'Bus', 'CafeTeria', 'Car',
               'CopyMachine', 'Metro',
               'Office', 'Restaurant', 'ShuttingDoor',
               'Traffic', 'Typing', 'VacuumCleaner', 'Washing']

    remove_unkowns = True
    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])
    quantize = True

    if load_train:
        train_dataset = MSnoise(root=data_dir, classes=classes, d_type='train',
                                remove_unknowns=remove_unkowns, transform=transform,
                                quantize=quantize, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = MSnoise(root=data_dir, classes=classes, d_type='test',
                               remove_unknowns=remove_unkowns, transform=transform,
                               quantize=quantize, download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def MSnoise_get_unquantized_datasets(data, load_train=True, load_test=True):
    """
    Load the folded 1D and unquantized version of MS Scalable Noisy Speech dataset (MS-SNSD)

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 26 different noise types.
    """
    (data_dir, args) = data

    # classes = ['AirConditioner', 'AirportAnnouncements',
    #            'Babble', 'Bus', 'CafeTeria', 'Car',
    #            'CopyMachine', 'Field', 'Hallway', 'Kitchen',
    #            'LivingRoom', 'Metro', 'Munching', 'NeighborSpeaking',
    #            'Office', 'Park', 'Restaurant', 'ShuttingDoor',
    #            'Square', 'SqueakyChair', 'Station', 'Traffic',
    #            'Typing', 'VacuumCleaner', 'WasherDryer', 'Washing']

    classes = ['AirConditioner',
               'CafeTeria', 'Car',
               'CopyMachine',
               'Office', 'Restaurant',
               'Typing', 'VacuumCleaner', 'WasherDryer']

    remove_unkowns = True
    transform = None
    quantize = False

    if load_train:
        train_dataset = MSnoise(root=data_dir, classes=classes, d_type='train',
                                remove_unknowns=remove_unkowns, transform=transform,
                                quantize=quantize, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = MSnoise(root=data_dir, classes=classes, d_type='test',
                               remove_unknowns=remove_unkowns, transform=transform,
                               quantize=quantize, download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'MSnoise',
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        'loader': MSnoise_get_datasets,
    },
    {
        'name': 'MSnoise_unquantized',
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
        'loader': MSnoise_get_unquantized_datasets,
    },
]
