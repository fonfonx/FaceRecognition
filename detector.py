# file for face detection example
# we use the opencv face detector

import numpy as np
import cv2
from os import listdir
import sys

path='/home/xavier/opencv/opencv-2.4.10/data/haarcascades/'

def faceDetector(image, newName="image", saveFolder="", display=True, save=False):
    path = '/home/xavier/opencv/opencv-2.4.10/data/haarcascades/'
    img=cv2.imread(image)
    # img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(path+'haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    nbr=0
    for (x,y,w,h) in faces:
        nbr+=2
        crop=img[y:y+h, x:x+w]
        crop_gray=gray[y:y+h, x:x+w]
        if save:
            cv2.imwrite(saveFolder+newName+"_"+str(nbr)+".jpg",crop)
        if display:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # eyes = eye_cascade.detectMultiScale(crop_gray)
            # for (ex,ey,ew,eh) in eyes:
            #    cv2.rectangle(crop,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    if display:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print "detection done!"

if len(sys.argv)>=2:
    image=sys.argv[1]

faceDetector(image)

