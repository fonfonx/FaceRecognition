# this file contains the main function to create the dictionaries

import numpy as np
import cv2
import dlib
from PIL import Image
from os import listdir
from os.path import isfile, join
from scipy.fftpack import dct, fft
from numpy.fft import fft2

from preprocessing import *
from alignment import align, dist, meshAlign

# represent a number with a string of 'tot' characters
# pad with 0 if the length is less than tot
def fillStringNumber(val, tot):
    valstr = str(val)
    while (len(valstr) < tot):
        valstr = "0" + valstr
    return valstr


def stack_complex_matrix(mat):
    return np.concatenate((mat.real,mat.imag))


imref = cv2.imread("../LFW_verybig/George_W_Bush/George_W_Bush_0089.jpg")

def columnFromImage(img):
    # PIL
    #im = Image.open(img)
    #im = im.convert("L")
    #im = np.asarray(im)
    print img
    im=cv2.imread(img)

    im = meshAlign(im, imref)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)


    #imref = cv2.cvtColor(imref, cv2.COLOR_BGR2GRAY)
    #im=align(im)



    #im=im.astype(float)
    #im=dct(im)
    #im=stack_complex_matrix(fft2(im))
    #im=absMat(im)
    #im=preprocessing(im)
    #return pca(im,200)
    rep= np.transpose(im).flatten()
    return rep
    #return landmarkImage(im)


    #return positionImage(im)

def landmarkImage(img):
    #predictor_path = "/home/xavier/dlib-18.18/shape_predictor_68_face_landmarks.dat"
    predictor_path="/root/Programs/dlib-18.18/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    #cascade_path = '/home/xavier/opencv/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml'
    cascade_path="/root/Programs/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    rects = cascade.detectMultiScale(img, 1.3, 5)
    x, y, w, h = rects[0].astype(long)
    x = x.item()
    y = y.item()
    h = h.item()
    w = w.item()
    rect = dlib.rectangle(x, y, x + w, y + h)
    #rep=np.array([img[p.x,p.y] for p in predictor(img, rect).parts()])
    lm=predictor(img,rect).parts()
    rep=np.zeros(25*len(lm))
    l=0
    for p in lm:
        for i in range(-2,3):
            for j in range(-2,3):
                rep[i*5+j+25*l]=img[p.x+i,p.y+j]
        l+=1
    return rep
    #return np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])


# intermediate function for the next method
def fillDist(lm,center,norm):
    tupcenter=lm[center].x,lm[center].y
    tupnorm=lm[norm].x,lm[norm].y
    Znorm=dist(tupnorm,tupcenter)
    rep=np.zeros(67)
    for i in range(68):
        if i!=center:
            tup = lm[i].x, lm[i].y
            rep[i-(i>center)]=dist(tup,tupcenter)/(1.0*Znorm)
    return rep


# the image is mapped to the vector of distances between nose and other landmarks
def positionImage(img):
    predictor_path = "/home/xavier/dlib-18.18/shape_predictor_68_face_landmarks.dat"
    #predictor_path = "/root/Programs/dlib-18.18/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    cascade_path = '/home/xavier/opencv/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml'
    #cascade_path = "/root/Programs/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    rects = cascade.detectMultiScale(img, 1.3, 5)
    x, y, w, h = rects[0].astype(long)
    x=x.item()
    y=y.item()
    h=h.item()
    w=w.item()
    rect = dlib.rectangle(x, y, x + w, y + h)
    lm = predictor(img, rect).parts()
    rep30=fillDist(lm,30,27)
    rep37=fillDist(lm,37,27)
    rep44=fillDist(lm,44,27)
    rep0=fillDist(lm,0,27)
    rep16=fillDist(lm,16,27)
    rep56=fillDist(lm,56,27)
    rep8=fillDist(lm,8,27)
    rep=np.concatenate((rep30,rep37,rep44,rep0,rep16,rep56,rep8))
    return rep



# nbFaces: number of faces per training person
def createTrainingDico(nbFaces, database):
    nbMen = 50
    nbWomen = 50
    listImages = []
    for i in range(1, nbMen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = "M-" + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            # nomImage = "M-" + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
            pathImage = database + nomImage
            listImages.append(columnFromImage(pathImage))
    for i in range(1, nbWomen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = "W-" + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            # nomImage = "W-" + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
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

# create training and testing sets from a database with a fixed number of images in both sets
def createDicosFromDirectory_fixed(repo, trainSize, testSize):
    trainImages = []
    testImages = []
    directories = sorted(listdir(repo))
    nbClasses = len(directories)
    for d in directories:
        nb_train=1
        images = sorted(listdir(repo + d))
        #for i in np.random.permutation(range(n)):
        for i in range(trainSize+testSize):
            pathImage = repo + d + "/" + images[i]
            if nb_train <= trainSize:
                nb_train += 1
                trainImages.append(columnFromImage(pathImage))
            else:
                testImages.append(columnFromImage(pathImage))
    trainSet = (np.column_stack(trainImages)).astype(float)
    testSet = (np.column_stack(testImages)).astype(float)
    print "Training et Test sets have been created with success!"
    return trainSet, testSet, nbClasses


def createDicoFromDirectory(repo):
    imagesArray = []
    directories = sorted(listdir(repo))
    for d in directories:
        images=sorted(listdir(repo+d))
        for image in images:
            pathImage=repo+d+"/"+image
            #print str(columnFromImage(pathImage).shape)+pathImage
            imagesArray.append(columnFromImage(pathImage))
    dico=(np.column_stack(imagesArray)).astype(float)
    print "dico created!"
    return dico
