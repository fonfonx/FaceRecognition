# functions performing face alignment

import cv2
import dlib
import numpy as np
from math import *
from os import listdir
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# initial parameters for a 59x43image
LEFT_EYE_POS=(12,19)
EYES_SPACE=18
FACE_HEIGHT=26 #from nose to chin
HEIGHT=35
WIDTH=35

predictor_path = "/home/xavier/dlib-18.18/shape_predictor_68_face_landmarks.dat"
#predictor_path = "/root/Programs/dlib-18.18/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
cascade_path='/home/xavier/opencv/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml'
#cascade_path="/root/Programs/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

def dist(tupleA,tupleB):
    return sqrt((tupleB[0]-tupleA[0])**2+(tupleB[1]-tupleA[1])**2)


#detectface: boolean value indicating if we have to perform face detection on the picture or not
def landmarks(img,detectface):
    if detectface:
        rects = cascade.detectMultiScale(img, 1.1, 5)
        x, y, w, h = rects[0].astype(long)
        x=x.item()
        y=y.item()
        w=w.item()
        h=h.item()
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
    LEFT_EYE_POS = (5, 7)
    EYES_SPACE = 18
    FACE_HEIGHT = 26  # from nose to chin
    HEIGHT = 30
    WIDTH = 35

    # #predictor_path = "/home/xavier/dlib-18.18/shape_predictor_68_face_landmarks.dat"
    # predictor_path = "/root/Programs/dlib-18.18/shape_predictor_68_face_landmarks.dat"
    # predictor = dlib.shape_predictor(predictor_path)
    # #cascade_path = '/home/xavier/opencv/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml'
    # cascade_path = "/root/Programs/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml"
    # cascade = cv2.CascadeClassifier(cascade_path)




    nose,chin,le,re,me=usefulPoints(img,False)
    print le,re
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
    print "factors:",x_factor,y_factor
    img_res = cv2.resize(img_rot, None, fx=x_factor, fy=y_factor, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("gul1step.jpg",img_res)
    # cv2.imshow("res",img_res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #2nd step
    img=img_res

    #translation
    #nose, chin, le, re, me = usefulPoints(img,True)
    #print le,re

    #detection does not always work
    # we compute le manually
    le=(le[0]*x_factor,le[1]*y_factor)

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


################################################################
################ IMAGE WARPING VERSION #########################
################################################################

def processImage(img):
    # landmark extraction
    lm=landmarks(img,True)
    # rectangle around face
    ymax=lm[8][1]
    xmin=lm[0][0]
    xmax=lm[16][0]
    ymin=min(lm[19][1],lm[24][1])
    xr=xmax-xmin
    yr=ymax-ymin
    epsilon=0.08
    xmin_rect=int(xmin-epsilon*xr)
    xmax_rect=int(xmax+epsilon*xr)
    ymin_rect=int(ymin-epsilon*yr)
    ymax_rect=int(ymax+epsilon*yr)
    coord=(xmin_rect,xmax_rect,ymin_rect,ymax_rect)
    # new landmarks (on the rectange sides)
    top_points=np.array([[x,ymin_rect] for x in np.linspace(xmin_rect,xmax_rect,15)])
    bottom_points=np.array([[x,ymax_rect] for x in np.linspace(xmin_rect,xmax_rect,20)])
    side_points=np.linspace(int(ymin_rect+yr*1.1/12.0),int(ymax_rect-yr*1.1/12.0),11)
    left_points=np.array([[xmin_rect,y] for y in side_points])
    right_points=np.array([[xmax_rect,y] for y in side_points])
    # all points for the triangulation
    lm_points=np.array([[x,y] for (x,y) in lm])
    all_points=np.concatenate((lm_points,top_points,right_points,bottom_points,left_points))
    #print all_points
    return all_points, coord

def delaunayTriangulation(points):
    tri=Delaunay(points)
    # plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # plt.show()
    return tri.simplices

# base_points: coordinates of the landmark points of reference image
# triangulation: Delaunay triangulation of the base points
def warpImage(img, triangulation, base_points,coord):
    all_points,co=processImage(img)
    img_out = 255 * np.ones(img.shape, dtype=img.dtype)
    for t in triangulation:
        # triangles to map one another
        src_tri=np.array([[all_points[x][0],all_points[x][1]] for x in t]).astype(np.float32)
        dest_tri=np.array([[base_points[x][0],base_points[x][1]] for x in t]).astype(np.float32)
        # bounding boxes
        src_rect=cv2.boundingRect(np.array([src_tri]))
        dest_rect=cv2.boundingRect(np.array([dest_tri]))
        # crop images
        src_crop_tri=np.zeros((3,2),dtype=np.float32)
        dest_crop_tri=np.zeros((3,2))
        for k in range(0,3):
            for dim in range(0,2):
                src_crop_tri[k][dim]=src_tri[k][dim]-src_rect[dim]
                dest_crop_tri[k][dim]=dest_tri[k][dim]-dest_rect[dim]
        src_crop_img=img[src_rect[1]:src_rect[1]+src_rect[3],src_rect[0]:src_rect[0]+src_rect[2]]
        # affine transformation estimation
        mat=cv2.getAffineTransform(np.float32(src_crop_tri),np.float32(dest_crop_tri))
        dest_crop_img = cv2.warpAffine(src_crop_img, mat, (dest_rect[2], dest_rect[3]), None, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
        # use a mask to keep only the triangle pixels
        # Get mask by filling triangle
        mask = np.zeros((dest_rect[3], dest_rect[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dest_crop_tri), (1.0, 1.0, 1.0), 16, 0)

        # Apply mask to cropped region
        dest_crop_img = dest_crop_img * mask

        # Copy triangular region of the rectangular patch to the output image
        img_out[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = \
            img_out[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] * ((1.0, 1.0, 1.0) - mask)

        img_out[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = \
            img_out[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] + dest_crop_img
    return img_out[coord[2]:coord[3],coord[0]:coord[1]]

def detectFace(img):
    path = '/home/xavier/opencv/opencv-2.4.10/data/haarcascades/'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    x,y,w,h=faces[0]
    face_img=img[y:y+h, x:x+w]
    return face_img


def meshAlign(img,imgref):
    bp,coord=processImage(imgref)
    tr=delaunayTriangulation(bp)
    img_out=warpImage(img,tr,bp,coord)
    print img_out.shape
    return img_out









# #im="../photomoi.jpg"
# #im="../LFW_verybig/Bill_Clinton/Bill_Clinton_0019.jpg"
# im="testgulechec.jpg"
# im2="../LFW_verybig/Bill_Clinton/Bill_Clinton_0002.jpg"
# img=cv2.imread(im)
# img2=cv2.imread(im2)
# align(img)
# # repo="../AR_matlab/"
# # initializeParameters(repo)
# # bp,coord=processImage(img)
# # tr=delaunayTriangulation(bp)
# # img_out=warpImage(img2,tr,bp,coord)
# # cv2.imshow("wrap",img_out)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
