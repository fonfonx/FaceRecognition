# this file contains functions used in a SIFT-based approach

import cv2
import numpy as np
from creation import fillStringNumber
from matrix import normColumn, normalizeColumn, normalizeMatrix, dimReduct
from recognizer import fdelta, toDiag, l2_ls, debug_alpha, debug_diff_tab, mean_sample
from pca import PCA_reductor
import l1ls as L

def test():
    img="../AR_matlab/M-006-06.bmp"
    #img="../cousins.jpg"
    img=cv2.imread(img)
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    s=cv2.SIFT(contrastThreshold=0.07)
    #sift = cv2.xfeatures2d.SIFT_create()
    kp = s.detect(gray, None)
    kp,des=s.compute(gray,kp)
    print des.shape
    print len(kp)
    #img = cv2.drawKeypoints(gray, kp)
    #cv2.imwrite('test.jpg', img)
    #cv2.imshow("image",img)
    #cv2.waitKey(0)


def createSIFTDico(nbFaces, database):
    nbMen = 50
    nbWomen = 50
    listSIFT = []
    nbLabels = []
    nbLabels.append(0)
    sift=cv2.SIFT(contrastThreshold=0.07)
    for i in range(1, nbMen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = "M-" + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            pathImage = database + nomImage
            im = cv2.imread(pathImage)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            kp=sift.detect(im,None)
            kp, des = sift.compute(im, kp)
            des=des.transpose()
            nb=des.shape[1]
            if (listSIFT==[]):
                listSIFT=des
            else:
                listSIFT=np.concatenate((listSIFT,des),axis=1)
            nbLabels.append(nb)
    for i in range(1, nbWomen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = "W-" + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            pathImage = database + nomImage
            im = cv2.imread(pathImage)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            kp = sift.detect(im, None)
            kp, des = sift.compute(im, kp)
            des = des.transpose()
            nb = des.shape[1]
            if (listSIFT == []):
                listSIFT = des
            else:
                listSIFT = np.concatenate((listSIFT, des), axis=1)
            nbLabels.append(nb)
    print "Creation of SIFT dictionary done"
    dico = (listSIFT).astype(float)
    print dico.shape
    return dico,nbLabels


def sift_identif(sift,dico):
    print "sift"
    e = np.array((sift - mean).astype(float))
    norm_y = normColumn(sift)
    NTest = normalizeColumn(sift)
    for j in range(nbIter):
        delta = fdelta(e)
        mu = param_c / delta
        before_exp = mu * (e ** 2 - delta)
        todiag = toDiag(before_exp)

        WTrain = normalizeMatrix(dico * todiag[:, np.newaxis])
        WTest = normalizeColumn(todiag * sift)

        WTrainRed = dimReduct(WTrain, reductor)
        WTestRed = dimReduct(WTest, reductor)
        D = normalizeMatrix(WTrainRed)
        y = normalizeColumn(WTestRed)

        [x, status, hist] = L.l1ls(D, y, lmbda, quiet=True)
        #x = l2_ls(D, y, lmbda)

        if j == 0:
            alpha = x
        else:
            # eta=find_eta(y,D,alpha,x,mu,delta,nbDim)
            alpha = alpha + eta * (x - alpha)

        if not (silence):
            debug_alpha(alpha)

        e = norm_y * (NTest - dico_norm.dot(alpha))
    return classif(D, y, alpha, nbLabels)

def classif(D, y, x, nbLabels):
    diff_tab = np.zeros(classNum)
    for c in range(classNum):
        xclass = x[nbLabels[c]:nbLabels[c+1]-1]
        Dclass = D[:, nbLabels[c]:nbLabels[c + 1] - 1]
        diff = y - Dclass.dot(xclass)
        diff_tab[c] = diff.dot(diff)
    if not (silence):
        # print diff_tab
        debug_diff_tab(diff_tab)
    return np.argmin(diff_tab) + 1


def image_identif(image,dico):
    img=cv2.imread(image)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT(contrastThreshold=0.07)
    kp = sift.detect(img, None)
    kp, des = sift.compute(img, kp)
    des = des.transpose()
    nb=des.shape[1]
    tab_res=np.zeros(classNum)
    for i in range(nb):
        tab_res[sift_identif(des[:,i],dico)-1]+=1
    return np.argmax(tab_res)+1


def test_recognizer(dico):
    nbMen=50
    nbWomen=50
    tot=0
    good=0
    for i in range(1, nbMen + 1):
        for j in range(nbFaces):
            k=14+j
            nomImage = "M-" + fillStringNumber(i, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            pathImage = database + nomImage
            class_identif=image_identif(pathImage,dico)
            if (class_identif==i):
                good+=1
            tot+=1
            print 'Class '+str(i)+' identified as '+str(class_identif)
    for i in range(1, nbWomen + 1):
        for j in range(nbFaces):
            k = 14 + j
            nomImage = "W-" + fillStringNumber(i, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            pathImage = database + nomImage
            class_identif = image_identif(pathImage, dico)
            if (class_identif == i+nbMen):
                good += 1
            tot += 1
            print 'Class ' + str(i+nbMen) + ' identified as ' + str(class_identif)
    rate=1.0*good/(1.0*tot)
    print "rate: ",rate






nbDim = 120
nbIter = 2
param_c = 8.0
param_tau = 0.8
lmbda = 0.001
rel_tol = 0.001
eta = 1.0
silence = False
classNum=100
nbFaces=7

database = "../AR_matlab/"
dico, nbLabels=createSIFTDico(7,database)
reductor = PCA_reductor(dico, nbDim)
mean = mean_sample(dico)
dico_norm = normalizeMatrix(dico)

test_recognizer(dico)