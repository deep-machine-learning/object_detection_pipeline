from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import glob
import os
import uuid


def save_image(folder, image, data_format):
    """
    saves an image in a folder, with a defined name and format
    :param folder:
    :param image:
    :param data_format:
    """
    generated_id = str(uuid.uuid4())
    image_filename = ''.join([folder, 'JPEGImages/', generated_id, data_format])
    labels_filename = ''.join([folder, 'labels/', generated_id, '.txt'])

    cv2.imwrite(image_filename, image.image)
    with open(labels_filename, 'w') as f:
        for value in image.labels.values():
            line = " ".join(str(v) for v in value.values())
            f.writelines(line)


def convert_yolo_opencv(image_shape, x, y, w, h):
    """
    converts yolo coordinates to pascal_voc bounding boxed.
    :param image_shape:
    :param x:
    :param y:
    :param w:
    :param h:
    :return xmin ymin xmax ymax:
    """
    dw = 1. / image_shape[1]
    dh = 1. / image_shape[0]
    x = x / dw
    y = y / dh
    w = (w / dw) / 2
    h = (h / dh) / 2

    xmin = x - w
    xmax = x + w
    ymin = y - h
    ymax = y + h
    return int(xmin), int(ymin), int(xmax), int(ymax)


def get_list_filenames(folder, pattern):
    """
    get list of files of specified format, from a specified folder
    :param folder:
    :param pattern:
    :return list of filenames:
    """
    filenames = []
    for fname in glob.glob(folder + pattern):
        filenames.append(os.path.splitext(os.path.basename(fname))[0])
    return filenames


def get_image_labels(label_path):
    """
    reads an image annotation file and returns the labels in a dictionary
    :param label_path:
    :return dictionary of labels:
    """
    labels = {}
    with open(label_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            labels[i] = {}
            labels[i]['class_id'], labels[i]['x'], labels[i]['y'], labels[i]['w'], labels[i]['h'] = line.split(" ")
    return labels


def get_labelled_images(folder_path):
    """
    builds a dictionary of image_paths with their labels from a specified folder
    :param folder_path:
    :return the dictionary of images:labels:
    """
    list_images = get_list_filenames('{0}/JPEGImages/'.format(folder_path), '*.jpg')
    list_labels = get_list_filenames('{0}/labels/'.format(folder_path), '*.txt')
    images_with_labels = list(set(list_images) & set(list_labels))  # get list of non-duplicated images having labels

    labelled_images = {}  # dict of images with their labels
    for image in images_with_labels:
        labelled_images[image] = "{0}/JPEGImages/{1}.jpg".format(folder_path, image), get_image_labels("{0}/labels/{1}.txt".format(folder_path, image))
    return labelled_images
