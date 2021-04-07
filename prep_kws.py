#!/usr/bin/env python3
###################################################################################################
"""
Preparation code for KWS
TODO: A safety mechanism to check if the _classes_ is a subset of the __kwlist_
"""
import datetime
import os
import random
import shutil

kwlist = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
          'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
          'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
          'wow', 'yes', 'zero', 'helplively']

classes_1 = ['helplively']

classes_5 = ['up', 'down', 'left', 'right', 'stop', 'go']

classes_20 = ['up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one', 'two',
              'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero']


def prep_files(__kwlist):
    """ Prepares the /data/KWS/raw folder for the given _kwlist """
    _kwlist = sorted(__kwlist)
    datasetdir = 'data/KWS'
    # remove the processed dir to apply preprocessing on new files
    processed_dataset_file = os.path.join(datasetdir, 'processed/dataset.pt')
    corpus_dir = os.path.join(datasetdir, 'corpus')
    if os.path.isfile(processed_dataset_file):
        os.remove(processed_dataset_file)

    raw_data_dir = os.path.join(datasetdir, 'raw')
    # clear raw data dir if it exists and makenew
    if os.path.isdir(raw_data_dir):
        shutil.rmtree(raw_data_dir)
    os.makedirs(raw_data_dir)

    for _class in _kwlist:
        s = os.path.join(corpus_dir, _class)
        d = os.path.join(raw_data_dir, _class)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


def prep_few_files(_kwlist, classes):
    """ Prepares the /data/KWS/raw folder for the given _kwlist """
    _kwlist = sorted(_kwlist)
    classes = sorted(classes)
    others = sorted(list(set(_kwlist) - set(classes)))
    datasetdir = 'data/KWS'
    # remove the processed dir to apply preprocessing on new files
    processed_dataset_file = os.path.join(datasetdir, 'processed/dataset.pt')
    corpus_dir = os.path.join(datasetdir, 'corpus')
    if os.path.isfile(processed_dataset_file):
        os.remove(processed_dataset_file)
    raw_data_dir = os.path.join(datasetdir, 'raw')
    if os.path.isdir(raw_data_dir):
        shutil.rmtree(raw_data_dir)
    os.makedirs(raw_data_dir)

    # copy others completely

    for _class in others:
        s = os.path.join(corpus_dir, _class)
        d = os.path.join(raw_data_dir, _class)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # Calculate how many samples per class on average
    counter = 0
    for _class in others:
        datadir = os.path.join(raw_data_dir, _class)
        counter += len(os.listdir(datadir))
    average = counter/len(others)

    # Now Calculate the few shot average
    counter = 0
    for _class in classes:
        datadir = os.path.join(corpus_dir, _class)
        counter += len(os.listdir(datadir))
    shot = int(counter/len(classes))

    # Repeat the few shots
    to_repeat = int(average/shot)
    for f in classes:
        fullname = os.path.join(corpus_dir, f)
        if os.path.isdir(fullname):
            out_folder_name = os.path.join(raw_data_dir, f)
            if not os.path.exists(out_folder_name):
                os.mkdir(out_folder_name)
            few_shots = random.sample(os.listdir(fullname), shot)
            for f2 in few_shots:
                for _ in range(to_repeat):
                    now = str(datetime.datetime.now())
                    now = now.replace(":", "_")
                    now = now.replace(" ", "_")
                    out_fname = os.path.join(out_folder_name, now + f2)
                    shutil.copyfile(os.path.join(fullname, f2), out_fname)


def calculate_weights(_kwlist, classes):
    """ Calculates variables necessary for other modules"""
    classes = sorted(classes)
    _kwlist = sorted(_kwlist)
    datasetdir = 'data/KWS'
    raw_data_dir = os.path.join(datasetdir, 'raw')

    class_dict = {_class: nr for nr, _class in enumerate(_kwlist)}
    total_number_of_samples = 0
    for _class in _kwlist:
        datadir = os.path.join(raw_data_dir, _class)
        total_number_of_samples += len(os.listdir(datadir))

    samples_dict = {}
    counter = 0

    for _class in classes:
        datadir = os.path.join(raw_data_dir, _class)
        samples_dict[_class] = len(os.listdir(datadir))
        counter += len(os.listdir(datadir))
    samples_dict['others'] = total_number_of_samples - counter

    class_string = "classes = ["
    ctr = 0
    for _class in classes:

        class_string += '\'' + _class + '\''
        class_string = class_string + ']\n' if ctr == len(classes)-1 else class_string + ', '
        ctr += 1
    classes.append('others')
    commands_dict = {_class: nr for nr, _class in enumerate(classes)}
    weights = [0]*len(classes)
    bench_class = samples_dict[classes[0]]

    for _class in classes:
        weights[commands_dict[_class]] = bench_class/float(samples_dict[_class])
    weights_string = 'weights = ('
    ctr = 0

    for weight in weights:
        _weight = str(weight)
        weights_string += _weight
        weights_string = weights_string + ')\n' if ctr == len(weights)-1 else weights_string + ', '
        ctr += 1
    output_string = 'output = ('
    ctr = 0

    for weight in weights:
        _output = str(ctr)
        output_string += _output
        output_string = output_string + ')\n' if ctr == len(weights)-1 else output_string + ', '
        ctr += 1
    dict_string = 'class_dict = {'
    ctr = 0

    for key, val in class_dict.items():

        dict_string += '\'' + key + '\'' + ':'
        nr = str(val)
        dict_string += nr
        dict_string = dict_string + '}\n' if ctr == len(class_dict)-1 else dict_string + ',\n'
        ctr += 1

    prep_file = open("kws_config.py", "w")
    prep_file.write("'''KWS config'''\n")
    prep_file.write(weights_string)
    prep_file.write(class_string)
    prep_file.write(dict_string)
    prep_file.write(output_string)
    prep_file.close()


def main():
    """ main function, select which classes you wish to proceed """
    prep_few_files(_kwlist=kwlist, classes=classes_1)
    calculate_weights(_kwlist=kwlist, classes=classes_1)


if __name__ == "__main__":
    main()
