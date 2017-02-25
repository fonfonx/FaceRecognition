""" main functions to create the dictionaries """

from numpy.fft import fft2
from os import listdir
from os.path import isfile, join
from PIL import Image
from random import shuffle
from scipy.fftpack import dct, fft
import cv2
import dlib
import numpy as np
import random

from alignment import align, dist, meshAlign, preprocess, landmarks, detectFace
from config import *


def column_from_image(img):
    """ Create a column vector from input image """

    imref = cv2.imread(IMREF_PATH)
    print img
    im = cv2.imread(img)
    im = preprocess(im, imref)

    # im = detectFace(im)
    # cv2.imshow("al",im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # im = cv2.resize(im, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    # im = cv2.resize(im,(30,30),interpolation=cv2.INTER_CUBIC)

    # im = im.astype(float)
    rep = np.transpose(im).flatten()
    # print rep.shape
    return rep


def create_dictionaries_from_db(repo, train_size, test_size):
    """ Create training and testing sets from a database with a fixed number of images in both sets """

    train_images = []
    test_images = []
    name_labels = {}
    directories = sorted(listdir(repo))
    label = 0
    for d in directories:
        images = sorted(listdir(repo + d))
        shuffle(images)
        if len(images) >= 10:  # train_size + test_size:
            nb_img = 0
            i = 0
            while nb_img < train_size+test_size and i < len(images):
                path_image = repo + d + "/" + images[i]
                i += 1
                try:
                    if nb_img < train_size:
                        train_images.append(column_from_image(path_image))
                    else:
                        test_images.append(column_from_image(path_image))
                    nb_img += 1
                except (cv2.error, TypeError, ValueError) as e:
                    print "error image "+path_image+" "+str(e)
            if nb_img < train_size + test_size:
                print "removing " + d
                if nb_img <= train_size and nb_img > 0:
                    del train_images[-nb_img:]
                elif nb_img > 0:
                    del train_images[-train_size:]
                    del test_images[-(nb_img-train_size):]
            else:
                label += 1
                name_labels[label] = d

    train_set = (np.column_stack(train_images)).astype(float)
    test_set = (np.column_stack(test_images)).astype(float)

    print "Training and Test sets have been created with success!"
    print "There are " + str(label) + " classes"

    return train_set, test_set, label, name_labels
