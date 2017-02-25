"""
Main file for face recognition
"""

from math import *
from numpy import linalg as LA
from numpy.fft import fft2
from os import listdir
from os.path import isfile, join
import l1ls as L
import numpy as np
import sys
import time

from pca import pca_reductor
from creation import *
from matrix import *
from config import *
from rsc import RSC_identif


def test_recognizer(test_set):
    """ Perform the recognition task on the test_set and output accuracy """
    tot = 0
    good = 0
    p, n = test_set.shape
    for i in range(n):
        y = test_set[:, i]
        trueClass = 1 + int(i / TEST_FACES)
        classif = RSC_identif(dico, y, mean, reductor, dico_norm, nb_classes)
        print "Class " + str(trueClass) + " identified as " + str(classif)
        if classif == trueClass:
            good += 1
        tot += 1
    rate = good * 1.0 / (tot * 1.0)
    print "Recognition rate: ", rate


if __name__ == '__main__':
    dico, test_set, nb_classes, name_labels = create_dictionaries_from_db(DATABASE_PATH, TRAINING_FACES, TEST_FACES)
    reductor = pca_reductor(dico, NB_DIM)
    mean = mean_sample(dico)
    dico_norm = normalize_matrix(dico)
    test_recognizer(test_set)
