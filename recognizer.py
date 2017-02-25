"""
Main file for face recognition
Uses the paper about Robust Sparse Coding for Face Recognition
"""

from math import *
from numpy import linalg as LA
from numpy.fft import fft2
from os import listdir
from os.path import isfile, join
from PIL import Image
from scipy.fftpack import dct, fft
from sklearn.decomposition import PCA
import l1ls as L
import numpy as np
import sys
import time

from pca import PCA_reductor, KEigen
from creation import *
from matrix import *


def fdelta(residual):
    n = len(residual)
    psi = residual ** 2
    psi = np.sort(psi)
    return psi[int(abs(param_tau * n))]


def classif(D, y, x, nbFaces):
    diff_tab = np.zeros(classNum)
    for c in range(classNum):
        xclass = x[nbFaces * c:nbFaces * (c + 1)]
        Dclass = D[:, nbFaces * c:nbFaces * (c + 1)]
        diff = y - Dclass.dot(xclass)
        diff_tab[c] = diff.dot(diff)
    if not (silence):
        debug_diff_tab(diff_tab)
    return np.argmin(diff_tab) + 1


def toDiag(before_exp):
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
    pr = D.transpose().dot(D)
    n, n = pr.shape
    toinv = pr + lmbda * np.identity(n)
    inv = LA.inv(toinv)
    reste = D.transpose().dot(y)
    rep = inv.dot(reste)
    return rep


def RSC_identif(train_set, Test):
    e = np.array((Test - mean).astype(float))
    norm_y = norm_column(Test)
    NTest = normalize_column(Test)
    for j in range(nbIter):
        delta = fdelta(e)
        mu = param_c / delta
        before_exp = mu * (e ** 2 - delta)
        todiag = toDiag(before_exp)

        # choice 1: create diagonal matrix
        # W = np.diag(todiag.flatten())
        # WTrain = normalize_matrix(W.dot(train_set))
        # WTest = normalize_column(W.dot(Test))

        # choice 2: direct computation
        WTrain = normalize_matrix(train_set * todiag[:, np.newaxis])
        WTest = normalize_column(todiag * Test)

        WTrainRed = dim_reduct(WTrain, reductor)
        WTestRed = dim_reduct(WTest, reductor)
        D = normalize_matrix(WTrainRed)
        y = normalize_column(WTestRed)

        # [x, status, hist] = L.l1ls(D, y, lmbda, quiet=True)
        x = l2_ls(D, y, lmbda)

        if j == 0:
            alpha = x
        else:
            alpha = alpha + eta * (x - alpha)

        if not (silence):
            debug_alpha(alpha)

        e = norm_y * (NTest - dico_norm.dot(alpha))
    return classif(D, y, alpha, nbFaces)


def testRecognizer(test_set):
    tot = 0
    good = 0
    p, n = test_set.shape
    for i in range(n):
        y = test_set[:, i]
        trueClass = 1 + int(i / nbFacesTest)
        classif = RSC_identif(dico, y)
        print "Class " + str(trueClass) + " identified as " + str(classif)
        if classif == trueClass:
            good += 1
        tot += 1
    rate = good * 1.0 / (tot * 1.0)
    print "Recognition rate:", rate


def main():
    global classNum, nbFaces, dico, reductor, mean, dico_norm, nbFacesTest
    repo = "../lfw2/"
    nbFaces = 1
    nbFacesTest = 1
    dico, test_set, classNum, name_labels = create_dictionaries_from_db(repo, nbFaces, nbFacesTest)
    reductor = PCA_reductor(dico, nbDim)
    mean = mean_sample(dico)
    dico_norm = normalize_matrix(dico)
    testRecognizer(test_set)

nbDim = 120
nbIter = 2
param_c = 8.0
param_tau = 0.8
lmbda = 0.001
rel_tol = 0.001
eta = 1.0
silence = True

main()
