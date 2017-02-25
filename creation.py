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


# create training and testing sets from a database with a fixed number of images in both sets
def createDicosFromDirectory_fixed(repo, trainSize, testSize):
    trainImages = []
    testImages = []
    nameLabels = {}
    directories = sorted(listdir(repo))
    label = 0
    for d in directories:
        images = sorted(listdir(repo + d))
        shuffle(images)
        if len(images) >= 10:  # trainSize+testSize:
            nb_img = 0
            i = 0
            while nb_img < trainSize+testSize and i < len(images):
                pathImage = repo + d + "/" + images[i]
                i += 1
                try:
                    if nb_img < trainSize:
                        trainImages.append(column_from_image(pathImage))
                    else:
                        testImages.append(column_from_image(pathImage))
                    nb_img += 1
                except (cv2.error, TypeError, ValueError) as e:
                    print "error image "+pathImage+" "+str(e)
            if nb_img < trainSize + testSize:
                print "removing "+ d
                if nb_img <= trainSize and nb_img > 0:
                    del trainImages[-nb_img:]
                elif nb_img > 0:
                    del trainImages[-trainSize:]
                    del testImages[-(nb_img-trainSize):]
            else:
                label += 1
                nameLabels[label] = d
    trainSet = (np.column_stack(trainImages)).astype(float)
    testSet = (np.column_stack(testImages)).astype(float)
    print "Training and Test sets have been created with success!"
    print "There are " + str(label) + " classes"
    return trainSet, testSet, label, nameLabels
