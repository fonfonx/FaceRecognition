# functions performing face alignment

import cv2
import dlib
import numpy as np
from math import *

# initial parameters for a 100x100 image
# will be updated later
NOSE_POS=(50,50)
EYES_SPACE=40
FACE_HEIGHT=60 #from nose to chin

PREDICTOR_PATH = "/home/xavier/dlib-18.18/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='/home/xavier/opencv/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

def dist(tupleA,tupleB):
    return sqrt((tupleB[0]-tupleA[0])**2+(tupleB[1]-tupleA[1])**2)

def landmarks(img):
    h,w=img.shape[:2]
    print h,w
    rects = cascade.detectMultiScale(img, 1.3, 5)
    x, y, w, h = rects[0].astype(long)
    rect = dlib.rectangle(x, y, x + w, y + h)
    print x,y,w,h
    return np.array([(p.x, p.y) for p in predictor(img, rect).parts()])

# returns the position of the nose, eyes and chin
def usefulPoints(img):
    lm=landmarks(img)
    nose=lm[30]
    left_eye=tuple((np.array(lm[37])+np.array(lm[38])+np.array(lm[40])+np.array(lm[41]))/4.0)
    right_eye = tuple((np.array(lm[43]) + np.array(lm[44]) + np.array(lm[46]) + np.array(lm[47])) / 4.0)
    chin=lm[8]
    mideye=lm[27]
    return nose,chin,left_eye,right_eye,mideye


def initializeParameters(listImages):
    n=len(listImages)
    nArray=np.zeros(2)
    cArray = np.zeros(2)
    lArray = np.zeros(2)
    rArray = np.zeros(2)
    mArray=np.zeros(2)
    for im in listImages:
        img=cv2.imread(im)
        nose,chin,le,re,me=usefulPoints(img)
        nArray+=np.array(nose)
        cArray += np.array(chin)
        lArray += np.array(le)
        rArray += np.array(re)
        mArray+=np.array(me)
    nArray=nArray/(1.0*n)
    cArray=cArray/(1.0*n)
    lArray=lArray/(1.0*n)
    rArray=rArray/(1.0*n)
    mArray=mArray/(1.0*n)
    print lArray
    print rArray
    NOSE_POS=tuple(nArray)
    EYES_SPACE=dist(tuple(lArray),tuple(rArray))
    FACE_HEIGHT=dist(tuple(mArray),tuple(cArray))
    print NOSE_POS,EYES_SPACE,FACE_HEIGHT

def align(img):
    nose,chin,le,re,me=usefulPoints(img)

im="../photomoi.jpg"
initializeParameters([im])

