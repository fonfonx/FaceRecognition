# main file for face recognition
# use the paper about Robust Sparse Coding for Face Recognition

from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from math import *
import l1ls as L
import sys
from numpy import linalg as LA
import time
from os import listdir
from os.path import isfile, join
from scipy.fftpack import dct, fft
from numpy.fft import fft2

from pca import PCA_reductor, KEigen
from creation import *
from matrix import *
from preprocessing import *




############### DEBUG ############

def debug_diff_tab(diff_tab):
    print "debug diff_tab \n"
    for i in range(classNum):
        print str(i + 1) + ": " + str(diff_tab[i])


def debug_alpha(alpha):
    print "debug alpha \n"
    for i in range(classNum):
        for j in range(nbFaces):
            print str(i + 1) + ": " + str(alpha[i * nbFaces + j])


def debug_tab(tab):
    n = len(tab)
    for i in range(n):
        print tab[i]

###################################


def mean_sample(mat):
    n, m = mat.shape
    mean = np.array([sum(mat[i, :]) for i in range(n)]) / (1.0 * m)
    return mean


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
        # print diff_tab
        debug_diff_tab(diff_tab)
    return np.argmin(diff_tab) + 1


def myclassif(x, nbFaces):
    diff_tab = np.zeros(classNum)
    for c in range(classNum):
        diff_tab[c] = normColumn(x[nbFaces * c:nbFaces * (c + 1)])
    if not (silence):
        # print diff_tab
        debug_diff_tab(diff_tab)
    return np.argmax(diff_tab) + 1


def toDiag(before_exp):
    n = len(before_exp)
    rep = np.zeros(n)
    for i in range(n):
        if before_exp[i] <= 700.0:
            rep[i] = 1.0 / (1.0 + exp(before_exp[i]))
        else:
            rep[i] = 0.0
        #rep[i]=sqrt(rep[i])
    return rep


def l2_ls(D, y, lmbda):
    pr = D.transpose().dot(D)
    n, n = pr.shape
    toinv = pr + lmbda * np.identity(n)
    inv = LA.inv(toinv)
    reste = D.transpose().dot(y)
    rep=inv.dot(reste)
    return rep


def RSC_identif(TrainSet, Test):
    e = np.array((Test - mean).astype(float))
    norm_y = normColumn(Test)
    NTest = normalizeColumn(Test)
    for j in range(nbIter):
        delta = fdelta(e)
        mu = param_c / delta
        before_exp = mu * (e ** 2 - delta)
        todiag = toDiag(before_exp)

        # choice 1: create diagonal matrix
        # W = np.diag(todiag.flatten())
        # WTrain = normalizeMatrix(W.dot(TrainSet))
        # WTest = normalizeColumn(W.dot(Test))

        # choice 2: direct computation
        WTrain=normalizeMatrix(TrainSet*todiag[:,np.newaxis])
        WTest=normalizeColumn(todiag*Test)

        WTrainRed = dimReduct(WTrain, reductor)
        WTestRed = dimReduct(WTest, reductor)
        D = normalizeMatrix(WTrainRed)
        y = normalizeColumn(WTestRed)

        #[x, status, hist] = L.l1ls(D, y, lmbda, quiet=True)
        x = l2_ls(D, y, lmbda)

        if j == 0:
            alpha = x
        else:
            # eta=find_eta(y,D,alpha,x,mu,delta,nbDim)
            alpha = alpha + eta * (x - alpha)

        if not (silence):
            debug_alpha(alpha)

        e = norm_y * (NTest - dico_norm.dot(alpha))
    return classif(D, y, alpha, nbFaces)
    # return classif(TrainSet,Test,alpha,nbFaces)
    # return myclassif(alpha,nbFaces)


def test_class(man, nbr, nbMen):
    tot = 0
    good = 0
    if man:
        for j in range(nbFaces):
            k = 14 + j
            #k=8+j
            # nomImage = "M-" + fillStringNumber(nbr, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            nomImage = "m-" + fillStringNumber(nbr, 3) + "-" + str(k) + ".bmp"
            pathImage = database + nomImage
            y = columnFromImage(pathImage)
            classif = RSC_identif(dico, y)
            print "Class " + str(nbr) + " identified as " + str(classif)  # +" "+str(classif2)
            if classif == nbr:
                good += 1
            tot += 1
    else:
        for j in range(nbFaces):
            k = 14 + j
            #k=8+j
            # nomImage = "W-" + fillStringNumber(nbr, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            nomImage = "w-" + fillStringNumber(nbr, 3) + "-" + str(k) + ".bmp"
            pathImage = database + nomImage
            y = columnFromImage(pathImage)
            classif = RSC_identif(dico, y)
            print "Class " + str(nbr + nbMen) + " identified as " + str(classif)  # +" "+str(classif2)
            if classif == nbMen + nbr:
                good += 1
            tot += 1
    return tot, good


def test_recognizer():
    nbMen = 50
    nbWomen = 50
    tot = 0
    good = 0
    for i in range(1, nbMen + 2,1):
        tot_int, good_int = test_class(True, i, nbMen)
        tot += tot_int
        good += good_int
    errors_men=tot-good
    print "errors men",errors_men
    for i in range(1, nbWomen + 2,1):
        tot_int, good_int = test_class(False, i, nbMen)
        tot += tot_int
        good += good_int
    errors_women=tot-good-errors_men
    print "errors women",errors_women
    rate = good * 1.0 / (tot * 1.0)
    print "Recognition rate:", rate


def testRecognizer(testSet):
    tot = 0
    good = 0
    p, n = testSet.shape
    for i in range(n):
        y = testSet[:, i]
        trueClass = 1 + int(i / nbFacesTest)
        classif = RSC_identif(dico, y)
        print "Class " + str(trueClass) + " identified as " + str(classif)
        if classif == trueClass:
            good += 1
        tot += 1
    rate = good * 1.0 / (tot * 1.0)
    print "Recognition rate:", rate


def main(version):
    global classNum, nbFaces, dico, reductor, mean, dico_norm, nbFacesTest
    if version=='AR':
        classNum = 100
        nbFaces = 7
        dico = createTrainingDico(nbFaces, database)
        print dico.shape
        reductor = PCA_reductor(dico, nbDim)
        mean = mean_sample(dico)
        dico_norm = normalizeMatrix(dico)
        test_recognizer()
    if version=='other':
        db = database
        percent = 1.0
        #dico, testSet, classNum, nbFaces, nbFacesTest = createDicosFromDirectory(db, 0.5, percent)
        dico,labels,nameLabels, classNum=createDicoFromDirectory("../AR_DB_train/")
        testSet, labelstest, nameLabelstest, c=createDicoFromDirectory("../AR_DB_test/")
        nbFaces=7
        nbFacesTest=7
        reductor = PCA_reductor(dico, nbDim)
        mean = mean_sample(dico)
        dico_norm = normalizeMatrix(dico)
        testRecognizer(testSet)
    if version=='real':
        dirTrain = "../g8_images_train/"
        dirTest = "../g8_images_test/"
        dico = createDicoFromDirectory(dirTrain)
        reductor = PCA_reductor(dico, nbDim)
        mean = mean_sample(dico)
        dico_norm = normalizeMatrix(dico)
        testSet = createDicoFromDirectory(dirTest)
        classNum = 13
        nbFaces = 7
        nbFacesTest = 1
        testRecognizer(testSet)
    if version=='lfw':
        #repo="../LFW_big_train_resized/"
        #repo="../LFW_verybig/"
        repo="../LFW/"
        nbFaces = 7
        nbFacesTest = 3
        dico,testSet, classNum, nameLabels=createDicosFromDirectory_fixed(repo,nbFaces,nbFacesTest)
        reductor = PCA_reductor(dico, nbDim)
        mean = mean_sample(dico)
        dico_norm = normalizeMatrix(dico)
        testRecognizer(testSet)


### DATABASES ###

# database = "../AR_crop/"
database = "../AR_matlab/"
database = "../AR_DB/"
ATT_DB = "../../databases/ATT/"
Yale_DB = "../../databases/CroppedYale/"

nbDim = 120
nbIter = 2
param_c = 8.0
param_tau = 0.8
lmbda = 0.001
rel_tol = 0.001
eta = 1.0
silence = True

version='other'
main(version)
