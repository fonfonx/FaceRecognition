""" main functions to create the dictionaries """

from os import listdir
from os.path import isfile, join
from random import shuffle
import cv2
import dlib
import numpy as np
import random

from alignment import preprocess, detect_face
from config import *


def column_from_image(img, verbose=True):
    """ Create a column vector from input image """

    if verbose:
        print img
    imref = cv2.imread(IMREF_PATH)
    im = cv2.imread(img)
    im = preprocess(im, imref)

    # im = cv2.resize(im, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    # im = cv2.resize(im,(30,30),interpolation=cv2.INTER_CUBIC)

    rep = np.transpose(im).flatten()
    return rep


def create_dictionaries_from_db(repo, train_size, test_size, verbose=True):
    """ Create training and testing sets from a database with a fixed number of images in both sets """

    train_images = []
    test_images = []
    name_labels = {}
    directories = sorted(listdir(repo))
    label = 0
    print "Processing images ..."
    for d in directories:
        images = sorted(listdir(repo + d))
        shuffle(images)
        if len(images) >= 10:  # in the paper we consider only these images - can be replaced by train_size + test_size
            nb_img = 0
            i = 0
            while nb_img < train_size + test_size and i < len(images):
                path_image = repo + d + "/" + images[i]
                i += 1
                try:
                    if nb_img < train_size:
                        train_images.append(column_from_image(path_image, verbose))
                    else:
                        test_images.append(column_from_image(path_image, verbose))
                    nb_img += 1
                except (cv2.error, TypeError, ValueError) as e:
                    print "error image " + path_image + " " + str(e)
            if nb_img < train_size + test_size:
                print "Removing " + d
                if nb_img <= train_size and nb_img > 0:
                    del train_images[-nb_img:]
                elif nb_img > 0:
                    del train_images[-train_size:]
                    del test_images[-(nb_img - train_size):]
            else:
                label += 1
                name_labels[label] = d

    train_set = (np.column_stack(train_images)).astype(float)
    test_set = (np.column_stack(test_images)).astype(float)

    print "Training and Test sets have been created with success!"
    print "There are " + str(label) + " classes"

    return train_set, test_set, label, name_labels
