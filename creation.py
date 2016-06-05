# this file contains the main function to create the dictionaries

import numpy as np
import cv2
import dlib
from PIL import Image
from os import listdir
from os.path import isfile, join
from scipy.fftpack import dct, fft
from numpy.fft import fft2
import random
from random import shuffle

from preprocessing import *
from alignment import align, dist, meshAlign, preprocess, landmarks, detectFace
from config import *


imref=cv2.imread("../tete3.jpg")
#imref=cv2.imread("../AR_matlab/M-001-01.bmp")


def columnFromImage(img):
    print img
    im=cv2.imread(img)

    im=preprocess(im,imref)

    #im=detectFace(im)


    # cv2.imshow("al",im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    #im = cv2.resize(im, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    #im=cv2.resize(im,(30,30),interpolation=cv2.INTER_CUBIC)

    #im=im.astype(float)
    #im=preprocessing(im)
    rep= np.transpose(im).flatten()
    #print rep.shape
    return rep
    #return landmarkImage(im)
    #print RM
    #return positionImageRM(im,RM)


# create training and testing sets from a database with a fixed number of images in both sets
def createDicosFromDirectory_fixed(repo, trainSize, testSize):
    trainImages = []
    testImages = []
    nameLabels = {}
    directories = sorted(listdir(repo))
    label=0
    for d in directories:
        images = sorted(listdir(repo + d))
        shuffle(images)
        if len(images)>=trainSize+testSize:
            nb_img = 0
            i = 0
            while nb_img < trainSize+testSize and i<len(images):
                pathImage = repo + d + "/" + images[i]
                i+=1
                try:
                    if nb_img < trainSize:
                        trainImages.append(columnFromImage(pathImage))
                    else:
                        testImages.append(columnFromImage(pathImage))
                    nb_img+=1
                except (cv2.error,TypeError) as e:
                    print "error image "+pathImage
            if nb_img<trainSize+testSize:
                print "removing "+d
                if nb_img<=trainSize and nb_img>0:
                    del trainImages[-nb_img:]
                elif nb_img>0:
                    del trainImages[-trainSize:]
                    del testImages[-(nb_img-trainSize):]
            else:
                label+=1
                nameLabels[label]=d
    trainSet = (np.column_stack(trainImages)).astype(float)
    testSet = (np.column_stack(testImages)).astype(float)
    print "Training and Test sets have been created with success!"
    print "There are "+str(label)+" classes"
    return trainSet, testSet, label, nameLabels

# return a dictionary and an array containing the labels of the images
def createDicoFromDirectory(repo):
    imagesArray = []
    imagesLabels = []
    nameLabels ={}
    directories = sorted(listdir(repo))
    label=0
    for d in directories:
        label+=1
        nameLabels[label]=d
        images=sorted(listdir(repo+d))
        #shuffle(images)
        for image in images:
            pathImage=repo+d+"/"+image
            try:
                imagesArray.append(columnFromImage(pathImage))
                imagesLabels.append(label)
            except (cv2.error,TypeError) as e:
                print "exception for image "+image
                imagesArray.append(imagesArray[-1])
                imagesLabels.append(label)
    dico=(np.column_stack(imagesArray)).astype(float)
    print "dico created!"
    return dico, imagesLabels, nameLabels, label










########################################################################################################################
###################################### FUNCTIONS NOT USED ANY MORE #####################################################
########################################################################################################################

predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)

# represent a number with a string of 'tot' characters
# pad with 0 if the length is less than tot
def fillStringNumber(val, tot):
    valstr = str(val)
    while (len(valstr) < tot):
        valstr = "0" + valstr
    return valstr


def stack_complex_matrix(mat):
    return np.concatenate((mat.real,mat.imag))


def landmarkImage(img):
    lm=landmarks(img,True)
    rep=np.zeros(25*len(lm))
    l=0
    for (x,y) in lm:
        for i in range(-2,3):
            for j in range(-2,3):
                rep[i*5+j+25*l]=img[x+i,y+j]
        l+=1
    return rep
    #return np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])


# intermediate function for the next method
def fillDist(lm,center,norm):
    tupcenter=lm[center][0],lm[center][1]
    tupnorm=lm[norm][0],lm[norm][1]
    Znorm=dist(tupnorm,tupcenter)
    rep=np.zeros(67)
    for i in range(68):
        if i!=center:
            tup = lm[i][0], lm[i][1]
            rep[i-(i>center)]=dist(tup,tupcenter)/(1.0*Znorm)
    return rep


# the image is mapped to the vector of distances between nose and other landmarks
def positionImage(img):
    lm=landmarks(img,True)
    rep30=fillDist(lm,30,27)
    rep37=fillDist(lm,37,27)
    rep44=fillDist(lm,44,27)
    rep0=fillDist(lm,0,27)
    rep16=fillDist(lm,16,27)
    rep56=fillDist(lm,56,27)
    rep8=fillDist(lm,8,27)
    rep=np.concatenate((rep30,rep37,rep44,rep0,rep16,rep56,rep8))
    return rep

# random mapping
RM=np.zeros((500,2),dtype=np.int)

for i in range(500):
    RM[i][0]=int(random.randint(0,67))
    RM[i][1]=int(random.randint(0,67))

# amelioration of the preceding function with random mapping
def positionImageRM(img,RM):
    lm=landmarks(img,True)
    Znorm=dist((lm[27][0],lm[27][1]),(lm[30][0],lm[30][1]))
    rep=np.zeros(500)
    for i in range(500):
        a=RM[i][0].item()
        b=RM[i][1].item()
        tup1=lm[a][0],lm[a][1]
        tup2=lm[b][0],lm[b][1]
        rep[i]=dist(tup1,tup2)/(1.0*Znorm)
    return rep

# for AR db
# nbFaces: number of faces per training person
def createTrainingDico(nbFaces, database):
    nbMen = 50
    nbWomen = 50
    listImages = []
    for i in range(1, nbMen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = "M-" + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            #nomImage = "m-" + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
            pathImage = database + nomImage
            try:
                listImages.append(columnFromImage(pathImage))
            except (cv2.error, TypeError) as e:
                print "error image " + pathImage
                listImages.append(listImages[-1])
    for i in range(1, nbWomen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = "W-" + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            #nomImage = "w-" + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
            pathImage = database + nomImage
            try:
                listImages.append(columnFromImage(pathImage))
            except (cv2.error, TypeError) as e:
                print "error image " + pathImage
                listImages.append(listImages[-1])
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