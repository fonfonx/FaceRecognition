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
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    nbr=0
    for (x,y,w,h) in faces:
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
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print "detection done!"+image

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
    for perso in people:
        people_repo=trainName+perso
        createRepo(people_repo)
        images=sorted(listdir(folder+"/"+perso))
        for image in images:
            imagepath=folder+"/"+perso+"/"+image
            faceDetector(imagepath,image,people_repo, False, True)
    print "Preprocessing done!"
    return nbClasses



if len(sys.argv)>=2:
    image=sys.argv[1]
else:
    image="../g8_autre2.jpg"

g8_images="../g8_images"

#faceDetector(image)

preprocessing(g8_images)
