# functions performing face alignment

import cv2
import dlib
import numpy as np
from math import *
from os import listdir

# initial parameters for a 59x43image
LEFT_EYE_POS=(14,19)
EYES_SPACE=18
FACE_HEIGHT=26 #from nose to chin
HEIGHT=59
WIDTH=43

PREDICTOR_PATH = "/home/xavier/dlib-18.18/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='/home/xavier/opencv/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

def dist(tupleA,tupleB):
    return sqrt((tupleB[0]-tupleA[0])**2+(tupleB[1]-tupleA[1])**2)


#detectface: boolean value indicating if we have to perform face detection on the picture or not
def landmarks(img,detectface):
    if detectface:
        rects = cascade.detectMultiScale(img, 1.1, 5)
        x, y, w, h = rects[0].astype(long)
        rect = dlib.rectangle(x, y, x + w, y + h)
        # print x,y,w,h
    else:
        h,w=img.shape[:2]
        #print h,w
        rect=dlib.rectangle(0,0,h,w)
    return np.array([(p.x, p.y) for p in predictor(img, rect).parts()])

# returns the position of the nose, eyes and chin
def usefulPoints(img,detectface):
    lm=landmarks(img,detectface)
    nose=lm[30]
    left_eye=tuple((np.array(lm[37])+np.array(lm[38])+np.array(lm[40])+np.array(lm[41]))/4.0)
    right_eye = tuple((np.array(lm[43]) + np.array(lm[44]) + np.array(lm[46]) + np.array(lm[47])) / 4.0)
    chin=lm[8]
    mideye=lm[27]
    return nose,chin,left_eye,right_eye,mideye


def rotate(img, angle, center):
    #image_center = tuple(np.array(img.shape[:2]) / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, (int(img.shape[1]*1.0),int(img.shape[0]*1.0)), flags=cv2.INTER_LINEAR)
    return result

def translation(img, vec):
    tx=vec[0]
    ty=vec[1]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def initializeParameters(repo):
    listImages=sorted(listdir(repo))
    n=len(listImages)
    nArray=np.zeros(2)
    cArray = np.zeros(2)
    lArray = np.zeros(2)
    rArray = np.zeros(2)
    mArray=np.zeros(2)
    for im in listImages:
        img=cv2.imread(repo+im)
        nose,chin,le,re,me=usefulPoints(img,False)
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
    LEFT_EYE_POS=tuple(lArray)
    EYES_SPACE=dist(tuple(lArray),tuple(rArray))
    FACE_HEIGHT=dist(tuple(mArray),tuple(cArray))
    print LEFT_EYE_POS,EYES_SPACE,FACE_HEIGHT

def align(img):
    # initial parameters for a 59x43image
    LEFT_EYE_POS = (12, 19)
    EYES_SPACE = 18
    FACE_HEIGHT = 26  # from nose to chin
    HEIGHT = 59
    WIDTH = 43

    PREDICTOR_PATH = "/home/xavier/dlib-18.18/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    cascade_path = '/home/xavier/opencv/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)




    cv2.imshow("img",img)
    nose,chin,le,re,me=usefulPoints(img,True)
    #1st step
    #rotation around left eye
    eye_vector=np.array(re)-np.array(le)
    theta=atan2(eye_vector[1],eye_vector[0])
    img_rot=rotate(img,theta*180.0/pi,le)
    # resizing
    eye_space=dist(le,re)
    face_height=dist(chin,me)
    #print "eye space",eye_space
    #print "face height",face_height
    x_factor = EYES_SPACE / eye_space
    y_factor=FACE_HEIGHT/face_height
    factor=(x_factor+y_factor)/2.0
    #print "factors:",x_factor,y_factor
    img_res = cv2.resize(img_rot, None, fx=x_factor, fy=x_factor, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("res",img_res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #2nd step
    img=img_res
    #translation
    nose, chin, le, re, me = usefulPoints(img,True)
    transl_vec=tuple(np.array(LEFT_EYE_POS)-np.array(le))
    img_t=translation(img,transl_vec)
    #crop
    crop_img=img_t[0:HEIGHT,0:WIDTH]
    img=crop_img

    #3rd step: verification
    # img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #nose, chin, le, re, me = usefulPoints(img, False)
    #print le,re

    # cv2.imshow("res",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

#im="../photomoi.jpg"
# im="../LFW_verybig/Abdullah_Gul/Abdullah_Gul_0005.jpg"
# img=cv2.imread(im)
# align(img)
#repo="../AR_matlab/"
#initializeParameters(repo)
