# file for face detection example
# we use the opencv face detector

import numpy as np
import cv2
from os import listdir
import sys
import os
import shutil

path='/home/xavier/opencv/opencv-2.4.10/data/haarcascades/'

def faceDetector(image, newName="image", saveFolder="", display=True, save=False):
    path = '/home/xavier/opencv/opencv-2.4.10/data/haarcascades/'
    img=cv2.imread(image)
    #img = cv2.resize(img,None,fx=1.5, fy=1.5, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(path+'haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#, minSize=(100,100))
    print faces
    nbr=0
    minwidth=1000
    for (x,y,w,h) in faces:
        if w<=minwidth:
            minwidth=w
        nbr+=1
        crop=img[y:y+h, x:x+w]
        crop_gray=gray[y:y+h, x:x+w]
        if save:
            pathname=saveFolder+"/"+str(nbr)+newName
            cv2.imwrite(pathname,crop)
        if display:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # eyes = eye_cascade.detectMultiScale(crop_gray)
            # for (ex,ey,ew,eh) in eyes:
            #    cv2.rectangle(crop,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    if display:
        #cv2.imwrite(pathname,img)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print "detection done!"+image
    return minwidth

# create repository dir if it does not exist
def createRepo(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)


def preprocessing(folder):
    trainName=folder+"_train/"
    createRepo(trainName)
    people=sorted(listdir(folder))
    nbClasses=len(people)
    minwidth=1000
    for perso in people:
        people_repo=trainName+perso
        createRepo(people_repo)
        images=sorted(listdir(folder+"/"+perso))
        for image in images:
            imagepath=folder+"/"+perso+"/"+image
            mw=faceDetector(imagepath,image,people_repo, False, True)
            if mw<=minwidth:
                minwidth=mw
    print "Preprocessing done!",minwidth
    return nbClasses,minwidth



if len(sys.argv)>=2:
    image=sys.argv[1]
else:
    image="../g8.jpg"

g8_images="../g8_images"

lfw="../LFW_big"

#preprocessing(lfw)

#faceDetector(image,"g8_detect.jpg",".",True,True)
#faceDetector(image,"politic.jpg","../g8_images_test/test",False,True)

#preprocessing(g8_images)

im="../LFW_verybig/Abdullah_Gul/Abdullah_Gul_0001.jpg"
faceDetector(im,"qf","j",True, False)
