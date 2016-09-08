import cv2
import dlib
import numpy as np

from config import *

predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)

def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.2,5)
    print len(rects)
    print rects
    if len(rects)==0:
        x=0
        y=0
        w,h=im.shape[:2]
    else:
        rects = rects[np.argsort(rects[:, 3])[::-1]]
        x,y,w,h =rects[0].astype(long)
    rect=dlib.rectangle(x,y,x+w,y+h)
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        #print idx, point
        pos = (point[0, 0], point[0, 1])
        #cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        cv2.circle(im, pos, 2, color=(255, 0, 0))
        cv2.circle(im, pos, 1, color=(255, 0, 0))
        cv2.circle(im, pos, 0, color=(255, 0, 0))
    return im


def test(im):
    x,y=im.shape[:2]
    rect=dlib.rectangle(0,0,x,y)
    lm=[]
    for p in predictor(im,rect).parts():
        print p.x,p.y
        #lm.append(im[p.x,p.y])
    print lm
    return np.matrix([[p.x,p.y] for p in predictor(im,rect).parts()])



im=cv2.imread('../tete2.png')
im = cv2.resize(im, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_CUBIC)
im_land=annotate_landmarks(im,get_landmarks(im))
cv2.imshow('Result',im_land)
#cv2.imshow('Result',annotate_landmarks(im,test(im)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("federer_landmarks.jpg",im_land)

