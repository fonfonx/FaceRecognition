"""
RSC algorithm (adapted version, cf our paper)
For the general RSC algorithm and details about the code see the paper 'Robust Sparse Coding for Face Recognition'
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

from matrix import *
from config import *


def f_delta(residual):
    """ Perform intermediate computation (see RSC paper) """
    n = len(residual)
    psi = residual ** 2
    psi = np.sort(psi)
    return psi[int(abs(PARAM_TAU * n))]


def classif(D, y, x, train_size, nb_classes):
    """ Perform classification of y using sparse encoding x """
    diff_tab = np.zeros(nb_classes)
    for c in range(nb_classes):
        xclass = x[train_size * c:train_size * (c + 1)]
        Dclass = D[:, train_size * c:train_size * (c + 1)]
        diff = y - Dclass.dot(xclass)
        diff_tab[c] = diff.dot(diff)
    return np.argmin(diff_tab) + 1


def to_diag(before_exp):
    """ Perform intermediate computation (see RSC paper) """
    n = len(before_exp)
    rep = np.zeros(n)
    for i in range(n):
        if before_exp[i] <= 700.0:
            rep[i] = 1.0 / (1.0 + exp(before_exp[i]))
        else:
            rep[i] = 0.0
        rep[i] = sqrt(rep[i])
    return rep


def l2_ls(D, y, lmbda):
    """ Analytically solve a l2-LASSO problem min ||x-Dy||^2 + lmbda ||x||^2"""
    pr = D.transpose().dot(D)
    n, n = pr.shape
    toinv = pr + lmbda * np.identity(n)
    inv = LA.inv(toinv)
    rest = D.transpose().dot(y)
    rep = inv.dot(rest)
    return rep


def RSC_identif(train_set, test_image, mean, reductor, dico_norm, nb_classes):
    """ Perform the identification of a test_image thanks to the adapted RSC algorithm """
    e = np.array((test_image - mean).astype(float))
    norm_y = norm_column(test_image)
    test_normalized = normalize_column(test_image)
    for j in range(NB_ITER):
        delta = f_delta(e)
        mu = PARAM_C / delta
        before_exp = mu * (e ** 2 - delta)
        todiag = to_diag(before_exp)

        w_train = normalize_matrix(train_set * todiag[:, np.newaxis])
        w_test = normalize_column(todiag * test_image)

        w_train_red = dim_reduct(w_train, reductor, DIM_REDUCTION)
        w_test_red = dim_reduct(w_test, reductor, DIM_REDUCTION)
        D = normalize_matrix(w_train_red)
        y = normalize_column(w_test_red)

        if REG_METHOD == 'l1':
            [x, status, hist] = L.l1ls(D, y, LAMBDA, quiet=True)
        else:
            x = l2_ls(D, y, LAMBDA)

        e = norm_y * (test_normalized - dico_norm.dot(x))

    return classif(D, y, x, TRAINING_FACES, nb_classes)
