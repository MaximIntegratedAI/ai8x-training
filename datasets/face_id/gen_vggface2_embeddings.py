###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Script to generate dataset for FaceID training and validation from VGGFace-2 dataset.
"""

import argparse
import json
import os

import numpy as np
import torch

import scipy.ndimage
from facenet_pytorch import MTCNN, InceptionResnetV1  # pylint: disable=no-name-in-module
from matplotlib.image import imread


def generate_image(img, box, count):  # pylint: disable=too-many-locals
    """
    Generates images in size 120x160x3 that includes the detected face in the image.

    img, box are the original image and box.
    count is how many pics you wanna generate

    box format: x1, y1, x3, y3
    """
    box[0] = np.max((box[0], 0))
    box[1] = np.max((box[1], 0))
    box[2] = np.min((box[2], img.shape[1]))
    box[3] = np.min((box[3], img.shape[0]))

    factor = 1
    height = img.shape[0]
    width = img.shape[1]
    new_img = img
    new_box = box
    while True:
        if height < 160 or width < 120:
            factor += 1
            new_img = scipy.ndimage.zoom(img, [factor, factor, 1], order=1)
            new_box = box * factor
            height = new_img.shape[0]
            width = new_img.shape[1]
        else:
            break
    new_box = np.round(new_box).astype(np.int)
    new_box_height = new_box[3] - new_box[1]
    new_box_width = new_box[2] - new_box[0]

    scale_list = np.concatenate((np.arange(0.9, 0, -0.1), np.arange(0.09, 0, -0.02)))
    ind = 0
    while (new_box_height > 160 or new_box_width > 120):
        if ind < scale_list.size:
            new_img = scipy.ndimage.zoom(img, [scale_list[ind], scale_list[ind], 1], order=1)
            new_box = box * scale_list[ind]
            new_box = np.round(new_box).astype(np.int)
            new_box_height = new_box[3] - new_box[1]
            new_box_width = new_box[2] - new_box[0]
            ind += 1
        else:
            pass

    min_x = np.max((0, new_box[0] - (120 - new_box_width)))
    min_y = np.max((0, new_box[1] - (160 - new_box_height)))
    max_x = np.min((new_box[0], width-120))
    max_y = np.min((new_box[1], height-160))

    start_x = np.random.choice(np.arange(min_x, max_x+1), count, replace=True)
    start_y = np.random.choice(np.arange(min_y, max_y+1), count, replace=True)
    img_arr = []
    box_arr = []
    for i in range(count):
        img_arr.append(new_img[start_y[i]:start_y[i]+160, start_x[i]:start_x[i]+120])
        temp_box = new_box.copy()
        temp_box[0] -= start_x[i]
        temp_box[2] -= start_x[i]
        temp_box[1] -= start_y[i]
        temp_box[3] -= start_y[i]
        box_arr.append(temp_box)
    new_img = img_arr
    new_box = box_arr
    return new_img, new_box, img, box


def main(source_path, dest_path):  # pylint: disable=too-many-locals
    """
    Main function to iterate over the images in the raw data and generate data samples
    to train/test FaceID model.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=80, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    data_dir_list = os.listdir(source_path)
    for i, folder in enumerate(data_dir_list):
        if i % 10 == 0:
            print('%d of %d' % (i, len(data_dir_list)))
        folder_path = os.path.join(source_path, folder)
        prcssd_folder_path = os.path.join(dest_path, folder)
        if not os.path.exists(prcssd_folder_path):
            os.makedirs(prcssd_folder_path)
        else:
            continue
        embedding_dict = dict()
        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image)
            img = imread(image_path)
            x_aligned, prob, box = mtcnn(img, return_prob=True)
            if box is not None and prob > 0.9:
                x_aligned = x_aligned[None, :]
                x_aligned = x_aligned.to(device)
                embeddings = resnet(x_aligned).detach().cpu()
                embedding_list = embeddings.numpy().ravel().tolist()
                img_arr, _, img, box = generate_image(img, box, 1)
                for ind, new_img in enumerate(img_arr):
                    new_img_name = image+'_160_120_'+str(ind)+'.npy'
                    new_img_path = os.path.join(prcssd_folder_path, new_img_name)
                    np.save(new_img_path, new_img)
                    embedding_dict[new_img_name] = embedding_list
        json_bin = json.dumps(embedding_dict)
        with open(os.path.join(prcssd_folder_path, "embeddings.json"), "w") as out_file:
            out_file.write(json_bin)


def parse_args():
    """Parses command line arguments"""
    data_folder = os.path.abspath(__file__)
    for _ in range(3):
        data_folder = os.path.dirname(data_folder)
    data_folder = os.path.join(data_folder, 'data')

    parser = argparse.ArgumentParser(description='Generate VGGFace-2 dataset to train/test \
                                                  FaceID model.')
    parser.add_argument('-r', '--raw', dest='raw_data_path', type=str,
                        default=os.path.join(data_folder, 'VGGFace-2', 'raw'),
                        help='Path to raw VGG-Face-2 dataset folder.')
    parser.add_argument('-d', '--dest', dest='dest_data_path', type=str,
                        default=os.path.join(data_folder, 'VGGFace-2'),
                        help='Folder path to store processed data')
    parser.add_argument('--type', dest='data_type', type=str, required=True,
                        help='Data type to generate (train/test)')
    args = parser.parse_args()

    source_path = os.path.join(args.raw_data_path, args.data_type)
    dest_path = os.path.join(args.dest_data_path, args.data_type, 'temp')
    return source_path, dest_path


if __name__ == "__main__":
    raw_data_path, dest_data_path = parse_args()
    main(raw_data_path, dest_data_path)
