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
Classes and functions used to create keyword spotting dataset.
"""
import errno
import hashlib
import os
import tarfile
import time
import warnings

import numpy as np
import torch
from torch.utils.model_zoo import tqdm
from torchvision import transforms

import librosa
import pytsmod as tsm
from six.moves import urllib

import ai8x


class KWS:
    """
    `SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset, 1D folded.

    Args:
    root (string): Root directory of dataset where ``KWS/processed/dataset.pt``
        exist.
    classes(array): List of keywords to be used.
    d_type(string): Option for the created dataset. ``train`` or ``test``.
    n_augment(int, optional): Number of augmented samples added to the dataset from
        each sample by random modifications, i.e. stretching, shifting and random noise.
    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version.
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    save_unquantized (bool, optional): If true, folded but unquantized data is saved.

    """

    url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    fs = 16000

    class_dict = {'backward': 0, 'bed': 1, 'bird': 2, 'cat': 3, 'dog': 4, 'down': 5,
                  'eight': 6, 'five': 7, 'follow': 8, 'forward': 9, 'four': 10, 'go': 11,
                  'happy': 12, 'house': 13, 'learn': 14, 'left': 15, 'marvin': 16, 'nine': 17,
                  'no': 18, 'off': 19, 'on': 20, 'one': 21, 'right': 22, 'seven': 23,
                  'sheila': 24, 'six': 25, 'stop': 26, 'three': 27, 'tree': 28, 'two': 29,
                  'up': 30, 'visual': 31, 'wow': 32, 'yes': 33, 'zero': 34}

    def __init__(self, root, classes, d_type, t_type, transform=None, quantization_scheme=None,
                 augmentation=None, download=False, save_unquantized=False):

        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.t_type = t_type
        self.transform = transform
        self.save_unquantized = save_unquantized

        self.__parse_quantization(quantization_scheme)
        self.__parse_augmentation(augmentation)

        if not self.save_unquantized:
            self.data_file = 'dataset2.pt'
        else:
            self.data_file = 'unquantized.pt'

        if download:
            self.__download()

        print(self.t_type)
        self.data, self.targets, self.data_type = torch.load(os.path.join(
            self.processed_folder, self.data_file))

        print(self.d_type)
        print(self.data.shape)

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

    def __parse_augmentation(self, augmentation):
        self.augmentation = augmentation
        if augmentation:
            if 'aug_num' not in augmentation:
                print('No key `aug_num` in input augmentation dictionary! ',
                      'It is set to 0.')
                self.augmentation['aug_num'] = 0
            elif self.augmentation['aug_num'] != 0:
                if 'noise_var' not in augmentation:
                    print('No key `noise_var` in input augmentation dictionary! ',
                          'It is set to defaults: [Min: 0., Max: 1.]')
                    self.augmentation['noise_var'] = {'min': 0., 'max': 1.}
                if 'shift' not in augmentation:
                    print('No key `shift` in input augmentation dictionary! '
                          'It is set to defaults: [Min:-0.1, Max: 0.1]')
                    self.augmentation['shift'] = {'min': -0.1, 'max': 0.1}
                if 'strech' not in augmentation:
                    print('No key `strech` in input augmentation dictionary! '
                          'It is set to defaults: [Min: 0.8, Max: 1.3]')
                    self.augmentation['strech'] = {'min': 0.8, 'max': 1.3}

    def __download(self):

        if self.__check_exists():
            return

        self.__makedir_exist_ok(self.raw_folder)
        self.__makedir_exist_ok(self.processed_folder)

        filename = self.url.rpartition('/')[2]
        self.__download_and_extract_archive(self.url, download_root=self.raw_folder,
                                            filename=filename)

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

    def __gen_bar_updater(self):  # pylint: disable=no-self-use
        pbar = tqdm(total=None)

        def bar_update(count, block_size, total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

    def __download_url(self, url, root, filename=None, md5=None):
        root = os.path.expanduser(root)
        if not filename:
            filename = os.path.basename(url)
        fpath = os.path.join(root, filename)

        self.__makedir_exist_ok(root)

        # downloads file
        if self.__check_integrity(fpath, md5):
            print('Using downloaded and verified file: ' + fpath)
        else:
            try:
                print('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath, reporthook=self.__gen_bar_updater())
            except (urllib.error.URLError, IOError) as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead.'
                          ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(url, fpath, reporthook=self.__gen_bar_updater())
                else:
                    raise e

    def __calculate_md5(self, fpath, chunk_size=1024 * 1024):  # pylint: disable=no-self-use
        md5 = hashlib.md5()
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def __check_md5(self, fpath, md5, **kwargs):
        return md5 == self.__calculate_md5(fpath, **kwargs)

    def __check_integrity(self, fpath, md5=None):
        if not os.path.isfile(fpath):
            return False
        if md5 is None:
            return True
        return self.__check_md5(fpath, md5)

    def __extract_archive(self, from_path,  # pylint: disable=no-self-use
                          to_path=None, remove_finished=False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        if from_path.endswith('.tar.gz'):
            with tarfile.open(from_path, 'r:gz') as tar:
                tar.extractall(path=to_path)
        else:
            raise ValueError("Extraction of {} not supported".format(from_path))

        if remove_finished:
            os.remove(from_path)

    def __download_and_extract_archive(self, url, download_root, extract_root=None, filename=None,
                                       md5=None, remove_finished=False):
        download_root = os.path.expanduser(download_root)
        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        self.__download_url(url, download_root, filename, md5)

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        self.__extract_archive(archive, extract_root, remove_finished)

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
        # print(inp)
        # print("Shape 1:", inp.shape)
        if self.transform is not None:
            inp = self.transform(inp)
        # print("Shape 2:", inp.shape)
        return inp, target

    @staticmethod
    def add_white_noise(audio, noise_var_coeff):
        """Adds zero mean Gaussian noise to image with specified variance.
        """
        coeff = noise_var_coeff * np.mean(np.abs(audio))
        noisy_audio = audio + coeff * np.random.randn(len(audio))
        return noisy_audio

    @staticmethod
    def shift(audio, shift_sec, fs):
        """Shifts audio.
        """
        shift_count = int(shift_sec * fs)
        return np.roll(audio, shift_count)

    @staticmethod
    def stretch(audio, rate=1):
        """Stretches audio with specified ratio.
        """
        input_length = 16000
        audio2 = librosa.effects.time_stretch(audio, rate)
        if len(audio2) > input_length:
            audio2 = audio2[:input_length]
        else:
            audio2 = np.pad(audio2, (0, max(0, input_length - len(audio2))), "constant")

        return audio2

    def augment(self, audio, fs, verbose=False):
        """Augments audio by adding random noise, shift and stretch ratio.
        """
        random_noise_var_coeff = np.random.uniform(self.augmentation['noise_var']['min'],
                                                   self.augmentation['noise_var']['max'])
        random_shift_time = np.random.uniform(self.augmentation['shift']['min'],
                                              self.augmentation['shift']['max'])
        random_strech_coeff = np.random.uniform(self.augmentation['strech']['min'],
                                                self.augmentation['strech']['max'])

        aug_audio = tsm.wsola(audio, random_strech_coeff)
        aug_audio = self.shift(aug_audio, random_shift_time, fs)
        aug_audio = self.add_white_noise(aug_audio, random_noise_var_coeff)
        if verbose:
            print(f'random_noise_var_coeff: {random_noise_var_coeff:.2f}\nrandom_shift_time: \
                    {random_shift_time:.2f}\nrandom_strech_coeff: {random_strech_coeff:.2f}')
        return aug_audio

    def augment_multiple(self, audio, fs, n_augment, verbose=False):
        """Calls `augment` function for n_augment times for given audio data.
        Finally the original audio is added to have (n_augment+1) audio data.
        """
        aug_audio = [self.augment(audio, fs, verbose=verbose) for i in range(n_augment)]
        aug_audio.insert(0, audio)
        return aug_audio

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
        """Quantize audio
        """
        if compand:
            data = KWS.compand(data, mu)

        step_size = 2.0 / 2**(num_bits)
        max_val = 2**(num_bits) - 1
        q_data = np.round((data - (-1.0)) / step_size)
        q_data = np.clip(q_data, 0, max_val)

        if compand:
            data_ex = (q_data - 2**(num_bits - 1)) / 2**(num_bits - 1)
            data_ex = KWS.expand(data_ex)
            q_data = np.round((data_ex - (-1.0)) / step_size)
            q_data = np.clip(q_data, 0, max_val)
        return np.uint8(q_data)

    def __gen_datasets(self, exp_len=16384, row_len=128, overlap_ratio=0):
        print('Generating dataset from raw data samples for the first time. ')
        print('Warning: This process could take an hour!')
        with warnings.catch_warnings():
            warnings.simplefilter('error')

            lst = sorted(os.listdir(self.raw_folder))
            labels = [d for d in lst if os.path.isdir(os.path.join(self.raw_folder, d))
                      and d[0].isalpha()]

            # PARAMETERS
            overlap = int(np.ceil(row_len * overlap_ratio))
            num_rows = int(np.ceil(exp_len / (row_len - overlap)))
            data_len = int((num_rows*row_len - (num_rows-1)*overlap))
            print('data_len: %s' % data_len)

            # show the size of dataset for each keyword
            print('------------- Label Size ---------------')
            for i, label in enumerate(labels):
                record_list = os.listdir(os.path.join(self.raw_folder, label))
                print('%8s:  \t%d' % (label, len(record_list)))
            print('------------------------------------------')

            for i, label in enumerate(labels):
                print(f'Processing the label: {label}. {i + 1} of {len(labels)}')
                record_list = sorted(os.listdir(os.path.join(self.raw_folder, label)))

                # dimension: row_length x number_of_rows
                if not self.save_unquantized:
                    data_in = np.empty(((self.augmentation['aug_num'] + 1) * len(record_list),
                                        row_len, num_rows), dtype=np.uint8)
                else:
                    data_in = np.empty(((self.augmentation['aug_num'] + 1) * len(record_list),
                                        row_len, num_rows), dtype=np.float32)
                data_type = np.empty(((self.augmentation['aug_num'] + 1) * len(record_list), 1),
                                     dtype=np.uint8)
                # create data classes
                data_class = np.full(((self.augmentation['aug_num'] + 1) * len(record_list), 1), i,
                                     dtype=np.uint8)

                time_s = time.time()
                train_count = 0
                test_count = 0
                for r, record_name in enumerate(record_list):
                    if r % 1000 == 0:
                        print('\t%d of %d' % (r + 1, len(record_list)))

                    if hash(record_name) % 10 < 9:
                        d_typ = np.uint8(0)  # train+val
                        train_count += 1
                    else:
                        d_typ = np.uint8(1)  # test
                        test_count += 1

                    record_pth = os.path.join(self.raw_folder, label, record_name)
                    record, fs = librosa.load(record_pth, offset=0, sr=None)
                    audio_seq_list = self.augment_multiple(record, fs,
                                                           self.augmentation['aug_num'])
                    for n_a, audio_seq in enumerate(audio_seq_list):
                        # store set type: train+validate or test
                        data_type[(self.augmentation['aug_num'] + 1) * r + n_a, 0] = d_typ

                        # Write audio 128x128=16384 samples without overlap
                        for n_r in range(num_rows):
                            start_idx = n_r*(row_len - overlap)
                            end_idx = start_idx + row_len
                            audio_chunk = audio_seq[start_idx:end_idx]
                            # pad zero if the length of the chunk is smaller than row_len
                            audio_chunk = np.pad(audio_chunk, [0, row_len-audio_chunk.size])
                            # store input data after quantization
                            data_idx = (self.augmentation['aug_num'] + 1) * r + n_a
                            if not self.save_unquantized:
                                data_in[data_idx, :, n_r] = \
                                    KWS.quantize_audio(audio_chunk,
                                                       num_bits=self.quantization['bits'],
                                                       compand=self.quantization['compand'],
                                                       mu=self.quantization['mu'])
                            else:
                                data_in[data_idx, :, n_r] = audio_chunk

                dur = time.time() - time_s
                print('Done in %.3fsecs.' % dur)
                print(data_in.shape)
                time_s = time.time()
                if i == 0:
                    data_in_all = data_in.copy()
                    data_class_all = data_class.copy()
                    data_type_all = data_type.copy()
                else:
                    data_in_all = np.concatenate((data_in_all, data_in), axis=0)
                    data_class_all = np.concatenate((data_class_all, data_class), axis=0)
                    data_type_all = np.concatenate((data_type_all, data_type), axis=0)
                dur = time.time() - time_s
                print('Data concat done in %.3fsecs.' % dur)

            data_in_all = torch.from_numpy(data_in_all)
            data_class_all = torch.from_numpy(data_class_all)
            data_type_all = torch.from_numpy(data_type_all)

            mfcc_dataset = (data_in_all, data_class_all, data_type_all)
            torch.save(mfcc_dataset, os.path.join(self.processed_folder, self.data_file))

        print('Dataset created!')
        print('Training+Validation: %d,  Test: %d' % (train_count, test_count))


class KWS_20(KWS):
    """
    `SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset, 1D folded.
    """

    def __str__(self):
        return self.__class__.__name__


def KWS_get_datasets(data, load_train=True, load_test=True, num_classes=6):
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

    augmentation = {'aug_num': 2}
    quantization_scheme = {'compand': False, 'mu': 10}

    if load_train:
        train_dataset = KWS(root=data_dir, classes=classes, d_type='train',
                            transform=transform, t_type='keyword',
                            quantization_scheme=quantization_scheme,
                            augmentation=augmentation, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = KWS(root=data_dir, classes=classes, d_type='test',
                           transform=transform, t_type='keyword',
                           quantization_scheme=quantization_scheme,
                           augmentation=augmentation, download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def KWS_20_get_datasets(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of SpeechCom dataset for 20 classes

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 35 keywords. A dataset is formed with 21 classes which includes
    20 of the original keywords and the rest of the dataset is used to form the last class, i.e.,
    class of the others.
    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by random stretch/shift and randomly adding noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """
    return KWS_get_datasets(data, load_train, load_test, num_classes=20)


def KWS_get_unquantized_datasets(data, load_train=True, load_test=True, num_classes=6):
    """
    Load the folded 1D version of SpeechCom dataset without quantization and augmentation
    """
    (data_dir, args) = data

    transform = None

    if num_classes == 6:
        classes = ['up', 'down', 'left', 'right', 'stop', 'go']
    elif num_classes == 20:
        classes = ['up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero']
    elif num_classes == 35:
        classes = ['backward', 'bed', 'bird', 'cat', 'dog', 'down',
                   'eight', 'five', 'follow', 'forward', 'four', 'go',
                   'happy', 'house', 'learn', 'left', 'marvin', 'nine',
                   'no', 'off', 'on', 'one', 'right', 'seven',
                   'sheila', 'six', 'stop', 'three', 'tree', 'two',
                   'up', 'visual', 'wow', 'yes', 'zero']
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')

    augmentation = {'aug_num': 0}
    quantization_scheme = {'bits': 0}

    if load_train:
        train_dataset = KWS(root=data_dir, classes=classes, d_type='train',
                            transform=transform, t_type='keyword',
                            quantization_scheme=quantization_scheme,
                            augmentation=augmentation, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = KWS(root=data_dir, classes=classes, d_type='test',
                           transform=transform, t_type='keyword',
                           quantization_scheme=quantization_scheme,
                           augmentation=augmentation, download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def KWS_35_get_unquantized_datasets(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of unquantized SpeechCom dataset for 20 classes.
    """
    return KWS_get_unquantized_datasets(data, load_train, load_test, num_classes=35)


datasets = [
    {
        'name': 'KWS',  # 6 keywords
        'input': (512, 64, 1),
        'output': (0, 1, 2, 3, 4, 5, 6),
        'weight': (1, 1, 1, 1, 1, 1, 0.06),
        'loader': KWS_get_datasets,
    },
    {
        'name': 'KWS_20',  # 20 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': KWS_20_get_datasets,
    },
    {
        'name': 'KWS_35_unquantized',  # 35 keywords
        'input': (128, 128, 1),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        'loader': KWS_35_get_unquantized_datasets,
    },
]
