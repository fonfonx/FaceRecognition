# main file for face recognition
# use the paper about Robust Sparse Coding for Face Recognition

from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from math import *
import l1ls as L
import sys
from pca import PCA_reductor, KEigen
from numpy import linalg as LA
import time
from os import listdir
from os.path import isfile, join
from scipy.fftpack import dct, fft

### DATABASES ###

# database = "../AR_crop/"
# database="../AR_DB/"
database = "../AR_matlab/"
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

if database == "../AR_DB/":
    debMen = "m-"
    debWomen = "w-"
else:
    debMen = "M-"
    debWomen = "W-"


# represent a number with a string of 'tot' characters
# pad with 0 if the length is less than tot
def fillStringNumber(val, tot):
    valstr = str(val)
    while (len(valstr) < tot):
        valstr = "0" + valstr
    return valstr


def columnFromImage(img):
    im = Image.open(img)
    im = im.convert("L")
    im = np.asarray(im)
    # im=dct(im)
    return np.transpose(im).flatten()


# nbFaces: number of faces per training person
def createTrainingDico(nbFaces):
    nbMen = 50
    nbWomen = 50
    listImages = []
    for i in range(1, nbMen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = debMen + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            # nomImage = debMen + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
            pathImage = database + nomImage
            listImages.append(columnFromImage(pathImage))
    for i in range(1, nbWomen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = debWomen + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            # nomImage = debWomen + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
            pathImage = database + nomImage
            listImages.append(columnFromImage(pathImage))
    print "Creation of dictionary done"
    dico = (np.column_stack(listImages)).astype(float)
    return dico


# trainpart: percent of the repo used for training (rest -> testing)
# percent: percent of the repo used (globally)
# return trainSet, testSet, number of classes, number of training images and test images per class
def createDicosFromDirectory(repo, trainpart, percent=1.0):
    trainImages = []
    testImages = []
    directories = sorted(listdir(repo))
    nbClasses = len(directories)
    for d in directories:
        images = sorted(listdir(repo + d))
        n = int(percent * (len(images)))
        nb_train = 1
        train_max = int(trainpart * n)
        for i in np.random.permutation(range(n)):
            pathImage = repo + d + "/" + images[i]
            if nb_train <= train_max:
                nb_train += 1
                trainImages.append(columnFromImage(pathImage))
            else:
                testImages.append(columnFromImage(pathImage))
    nbFacesTrain = train_max
    nbFacesTest = n - train_max
    trainSet = (np.column_stack(trainImages)).astype(float)
    testSet = (np.column_stack(testImages)).astype(float)
    print "Training et Test sets have been created with success!"
    return trainSet, testSet, nbClasses, nbFacesTrain, nbFacesTest


def normColumn(col):
    return LA.norm(col)


def normalizeColumn(col):
    col = col.astype(float)
    sq = normColumn(col)
    ncol = col / sq
    return ncol


def normalizeMatrix(matrix):
    n, m = matrix.shape
    nmatrix = np.zeros((n, m))
    for j in range(m):
        nmatrix[:, j] = normalizeColumn(matrix[:, j])
    return nmatrix


def powerMatDiagSqrt(mat):
    n, m = mat.shape
    for i in range(n):
        mat[i, i] = sqrt(mat[i, i])
    return mat


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


######################################


# def pca(matrix, dim):
#     matrix = np.transpose(matrix)
#     pca_object = PCA(dim)
#     pca_object.fit(matrix)
#     return pca_object

# def pcaReduc(pca_object, matrix):
#     matrix = np.transpose(matrix)
#     reduct = pca_object.transform(matrix)
#     return np.transpose(reduct)

# def rho(ei,mu,delta):
#     a=1+exp(mu*delta-mu*ei**2)
#     b=1+exp(mu*delta)
#     return (-1.0/(2*mu))*(log(a)-log(b))
#
# def ei(y,dico,alpha,i):
#     return y[i]-dico[i,:].dot(alpha)
#
# def sum_rho(y,dico,alpha,mu,delta,n):
#     rep=0.0
#     for i in range(n):
#         rep+=rho(ei(y,dico,alpha,i),mu,delta)
#     return rep
#
# def find_eta(y,dico,alpha,x,mu,delta,n):
#     old=sum_rho(y,dico,alpha,mu,delta,n)
#     for eta in np.arange(1.0,0.0,-0.1):
#         new_alpha=alpha+eta*(x-alpha)
#         new=sum_rho(y,dico,new_alpha,mu,delta,n)
#         if new<old:
#             print eta
#             return eta
#     print 'none'
#     return 0.05

########################################################

def dimReduct(matrix, reductor):
    return reductor.transpose().dot(matrix)


def mean_sample(mat):
    n, m = mat.shape
    mean = np.array([sum(mat[i, :]) for i in range(n)]) / (1.0 * m)
    return mean


def fdelta(residual):
    n = len(residual)
    psi = residual ** 2
    psi = np.sort(psi)
    return psi[abs(param_tau * n)]


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
    return rep


def l2_ls(D, y, lmbda):
    pr = D.transpose().dot(D)
    n, n = pr.shape
    toinv = pr + lmbda * np.identity(n)
    inv = LA.inv(toinv)
    reste = D.transpose().dot(y)
    return inv.dot(reste)


def RSC_identif(TrainSet, Test):
    e = np.array((Test - mean).astype(float))
    norm_y = normColumn(Test)
    NTest = normalizeColumn(Test)
    for j in range(nbIter):
        delta = fdelta(e)
        mu = param_c / delta
        # delta=120.0
        # mu=0.1
        before_exp = mu * (e ** 2 - delta)
        todiag = toDiag(before_exp)
        W = np.diag(todiag.flatten())
        WTrain = normalizeMatrix(W.dot(TrainSet))
        WTest = normalizeColumn(W.dot(Test))
        WTrainRed = dimReduct(WTrain, reductor)
        WTestRed = dimReduct(WTest, reductor)
        D = normalizeMatrix(WTrainRed)
        y = normalizeColumn(WTestRed)

        # [x, status, hist] = L.l1ls(D, y, lmbda, quiet=True)
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
            # k=8+j
            nomImage = debMen + fillStringNumber(nbr, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            # nomImage = debMen + fillStringNumber(nbr, 3) + "-" + str(k) + ".bmp"
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
            # k=8+j
            nomImage = debWomen + fillStringNumber(nbr, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            # nomImage = debMen + fillStringNumber(nbr, 3) + "-" + str(k) + ".bmp"
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
    for i in range(1, nbMen + 1):
        tot_int, good_int = test_class(True, i, nbMen)
        tot += tot_int
        good += good_int
    for i in range(1, nbWomen + 1):
        tot_int, good_int = test_class(False, i, nbMen)
        tot += tot_int
        good += good_int
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


### V2: choose directory from training (ATT, Yale)

# db=ATT_DB
# percent=1.0
# dico,testSet,classNum,nbFaces, nbFacesTest=createDicosFromDirectory(db,0.5,percent)
# reductor = PCA_reductor(dico, nbDim)
# mean=mean_sample(dico)
# dico_norm=normalizeMatrix(dico)
#
# testRecognizer(testSet)
# print "fin"
# sys.exit()

### V1: explicit names from training (AR)

classNum = 100
nbFaces = 7

dico = createTrainingDico(nbFaces)
reductor = PCA_reductor(dico, nbDim)
mean = mean_sample(dico)
dico_norm = normalizeMatrix(dico)

test_recognizer()
print "fin"
