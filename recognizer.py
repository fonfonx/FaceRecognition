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

#database = "../AR_crop/"
#database="../AR_DB/"
database="../AR_matlab/"
nbDim = 120
nbIter = 2
param_c = 8.0
param_tau = 0.8
lmbda = 0.001
rel_tol = 0.001
classNum = 100
nbFaces = 7
eta = 0.8
silence = True

if database=="../AR_DB/":
    debMen="m-"
    debWomen="w-"
else:
    debMen="M-"
    debWomen="W-"


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
    return np.transpose(im).flatten()

# nbFaces: number of faces per training person
def createTrainingDico(nbFaces):
    nbMen = 50
    nbWomen = 50
    listImages = []
    for i in range(1, nbMen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = debMen + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            #nomImage = debMen + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
            pathImage = database + nomImage
            listImages.append(columnFromImage(pathImage))
    for i in range(1, nbWomen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = debWomen + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            #nomImage = debWomen + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
            pathImage = database + nomImage
            listImages.append(columnFromImage(pathImage))
    print "Creation of dictionary done"
    dico=(np.column_stack(listImages)).astype(float)
    return dico

def normColumn(col):
    return LA.norm(col)

def normalizeColumn(col):
    col = col.astype(float)
    sq = normColumn(col)
    col /= sq
    return col

def normalizeMatrix(matrix):
    n, m = matrix.shape
    for j in range(m):
        matrix[:, j] = normalizeColumn(matrix[:, j])
    return matrix

def powerMatDiagSqrt(mat):
    n, m = mat.shape
    for i in range(n):
        mat[i, i] = sqrt(mat[i, i])
    return mat


############### DEBUG ############

def debug_diff_tab(diff_tab):
    print "debug diff_tab \n"
    for i in range(classNum):
        print str(i+1)+": "+str(diff_tab[i])

def debug_alpha(alpha):
    print "debug alpha \n"
    for i in range(classNum):
        for j in range(nbFaces):
            print str(i+1)+": "+str(alpha[i*nbFaces+j])


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
        #print diff_tab
        debug_diff_tab(diff_tab)
    return np.argmin(diff_tab) + 1

def myclassif(x,nbFaces):
    diff_tab=np.zeros(classNum)
    for c in range(classNum):
        diff_tab[c]=normColumn(x[nbFaces*c:nbFaces*(c+1)])
    if not (silence):
        # print diff_tab
        debug_diff_tab(diff_tab)
    return np.argmax(diff_tab)+1

def toDiag(before_exp):
    n=len(before_exp)
    rep=np.zeros(n)
    for i in range(n):
        if before_exp[i]<=700.0:
            rep[i]=1.0/(1.0+exp(before_exp[i]))
        else:
            rep[i]=0.0
    return rep

def RSC_identif(TrainSet, Test):
    NTrainSet = normalizeMatrix(TrainSet)
    NTest = normalizeColumn(Test)
    test_norm = normColumn(Test)
    # TrainSet=NTrainSet
    # Test=NTest
    ini = mean_sample(TrainSet)
    y_actu=ini
    e = np.array((Test - y_actu).astype(float))
    for j in range(nbIter):
        delta = fdelta(e)
        mu = param_c / delta
        before_exp=mu * e ** 2 - mu * delta
        #todiag = (1.0 / (np.exp(before_exp) + 1.0))
        todiag=toDiag(before_exp)
        W = np.diag(todiag.flatten())
        W = powerMatDiagSqrt(W)
        D = normalizeMatrix(W.dot(TrainSet))
        y = normalizeColumn(W.dot(Test))

        [x, status, hist] = L.l1ls(D, y, lmbda, quiet=True)


        if j == 0:
            alpha = x
        else:
            #eta=find_eta(y,D,alpha,x,mu,delta,nbDim)
            alpha = alpha + eta * (x - alpha)

        if not (silence):
            #print alpha
            debug_alpha(alpha)

        e = test_norm * (NTest - NTrainSet.dot(alpha))
        #e = (Test-TrainSet.dot(alpha)).astype(float)
    return classif(TrainSet, Test, alpha, nbFaces)
    #return classif(D,y,alpha,nbFaces)#,classif(TrainSet, Test, alpha, nbFaces)
    #return myclassif(x,nbFaces)

def test_class(man, nbr, dico_red, reductor, nbMen):
    tot = 0
    good = 0
    if man:
        for j in range(nbFaces):
            k = 14 + j
            nomImage = debMen + fillStringNumber(nbr, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            #nomImage = debMen + fillStringNumber(nbr, 3) + "-" + str(k) + ".bmp"
            pathImage = database + nomImage
            y = columnFromImage(pathImage)
            y = reductor.transpose().dot(y)
            classif = RSC_identif(dico_red, y)
            print "Class " + str(nbr) + " identified as " + str(classif)#+" "+str(classif2)
            if classif == nbr:
                good += 1
            tot += 1
            # fichier=file('reponses.txt','a')
            # fichier.write(''+str(i)+' '+str(classif)+'\n')
            # fichier.close()
    else:
        for j in range(nbFaces):
            k = 14 + j
            nomImage = debWomen + fillStringNumber(nbr, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            #nomImage = debWomen + fillStringNumber(nbr, 3) + "-" + str(k) + ".bmp"
            pathImage = database + nomImage
            y = columnFromImage(pathImage)
            y = reductor.transpose().dot(y)
            classif = RSC_identif(dico_red, y)
            print "Class " + str(nbMen + nbr) + " identified as " + str(classif)#+" "+str(classif2)
            if classif == nbMen + nbr:
                good += 1
            tot += 1
            # fichier=file('reponses.txt','a')
            # fichier.write(''+str(nbMen+i)+' '+str(classif)+'\n')
            # fichier.close()
    return tot, good


def test_recognizer():
    # fichier=file('reponses.txt','w')
    # fichier.close()
    nbMen = 50
    nbWomen = 50
    tot = 0
    good = 0
    # dim reduction
    dico_red = reductor.transpose().dot(dico)
    print "PCA done"
    to_test_men = [2, 4, 6, 9, 10, 11, 22, 24, 25, 27, 28, 29, 30, 31, 33, 35, 36, 46, 47, 48]
    to_test_women = [1, 5, 8, 9, 10, 11, 12, 14, 15, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30, 32, 35, 37, 39, 40, 41,
                     42, 43, 44, 47, 48, 50]
    to_test_big_men = [2, 28, 31, 48]
    to_test_big_women = [1, 5, 14, 24, 28, 30, 39, 41, 42, 47, 48]
    to_test = {1, 2}
    for i in range(1,nbMen+1):
        tot_int, good_int = test_class(True, i, dico_red, reductor, nbMen)
        tot += tot_int
        good += good_int
    for i in range(1,nbWomen+1):
        tot_int, good_int = test_class(False, i, dico_red, reductor, nbMen)
        tot += tot_int
        good += good_int
    rate = good * 1.0 / (tot * 1.0)
    print "Recognition rate:", rate


dico = createTrainingDico(nbFaces)
reductor = PCA_reductor(dico, nbDim)

test_recognizer()
print "fin"

