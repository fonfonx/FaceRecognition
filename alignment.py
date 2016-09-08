# functions performing face alignment (ie preprocessing tasks)
# face alignment is done in two parts: mesh align with triangulation and manual alignment

import cv2
import dlib
import numpy as np
from math import *
from os import listdir
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from config import *

predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)


def dist(tupleA, tupleB):
    return sqrt((tupleB[0] - tupleA[0]) ** 2 + (tupleB[1] - tupleA[1]) ** 2)


# return the landmarks of a face (array of tuples)
# detectface: boolean value indicating if we have to perform face detection on the picture or not
def landmarks(img, detectface):
    if detectface:
        rects = cascade.detectMultiScale(img, 1.3, 5)
        rects = rects[np.argsort(rects[:, 3])[::-1]]
        x, y, w, h = rects[0].astype(long)
        x = x.item()
        y = y.item()
        w = w.item()
        h = h.item()
        rect = dlib.rectangle(x, y, x + w, y + h)
        # print x,y,w,h
    else:
        h, w = img.shape[:2]
        # print h,w
        rect = dlib.rectangle(0, 0, h, w)
    return np.array([(p.x, p.y) for p in predictor(img, rect).parts()])

def detectFace(img):
    rects = cascade.detectMultiScale(img, 1.3, 5)
    rects = rects[np.argsort(rects[:, 3])[::-1]]
    x,y,w,h=rects[0]
    return img[y:y+h,x:x+w]


########################################################################################################################
######################################### MANUAL ALIGNMENT #############################################################
########################################################################################################################

# returns the position of the nose, eyes and chin
def usefulPoints(img, detectface):
    lm = landmarks(img, detectface)
    nose = lm[30]
    left_eye = tuple((np.array(lm[37]) + np.array(lm[38]) + np.array(lm[40]) + np.array(lm[41])) / 4.0)
    right_eye = tuple((np.array(lm[43]) + np.array(lm[44]) + np.array(lm[46]) + np.array(lm[47])) / 4.0)
    chin = lm[8]
    mideye = lm[27]
    # mouth
    mouth_array=np.zeros(2)
    for i in range(48,68):
        mouth_array+=np.array(lm[i])
    mouth=tuple(mouth_array/20.0)
    return nose, chin, left_eye, right_eye, mideye, mouth


# rotation of the image
def rotate(img, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, (int(img.shape[1] * 1.0), int(img.shape[0] * 1.0)), flags=cv2.INTER_LINEAR)
    return result


# translation of the image
def translation(img, vec):
    tx = vec[0]
    ty = vec[1]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


# manual align function
def align(img):
    nose, chin, le, re, me, mouth = usefulPoints(img, False) #should be false except for myface

    # 1st step
    # rotation around left eye
    eye_vector = np.array(re) - np.array(le)
    theta = atan2(eye_vector[1], eye_vector[0])
    img_rot = rotate(img, theta * 180.0 / pi, le)
    # resizing
    eye_space = dist(le, re)
    face_height = dist(chin, me)
    #eye_mouth=dist(me,mouth)
    x_factor = EYES_SPACE / eye_space
    #y_factor = EYE_MOUTH / eye_mouth
    y_factor = FACE_HEIGHT / face_height
    factor = (x_factor + y_factor) / 2.0
    img_res = cv2.resize(img_rot, None, fx=x_factor, fy=y_factor, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("me_rot.jpg",img_res)
    # 2nd step
    img = img_res

    ### in order to print/save intermediate result
    # cv2.imshow("inter",img_res)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imwrite("me_rot.jpg",img_res)

    # translation
    # nose, chin, le, re, me, mouth = usefulPoints(img,False)
    # detection does not always work
    # we compute le manually
    le = (le[0] * x_factor, le[1] * y_factor)
    transl_vec = tuple(np.array(LEFT_EYE_POS) - np.array(le))
    img_t = translation(img, transl_vec)
    # crop
    crop_img = img_t[0:HEIGHT, 0:WIDTH]
    img = crop_img
    cv2.imwrite("me_crop.jpg",crop_img)
    # 3rd step: verification
    # img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # nose, chin, le, re, me = usefulPoints(img, False)
    # print le,re

    # cv2.imshow("res",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


########################################################################################################################
####################################### IMAGE WARPING VERSION ##########################################################
########################################################################################################################

# preprocessing of the image
# --> return all the points that will be used for the triangulation and the coordinates of the rectangle around the face
def processImage(img):
    # landmark extraction
    lm = landmarks(img, True) # change if not LFW (True for LFW)
    # rectangle around face
    ymax = lm[8][1]
    xmin = lm[0][0]
    xmax = lm[16][0]
    ymin = min(lm[19][1], lm[24][1])
    xr = xmax - xmin
    yr = ymax - ymin
    epsilon = 0.08
    xmin_rect = int(xmin - epsilon * xr)
    xmax_rect = int(xmax + epsilon * xr)
    ymin_rect = int(ymin - epsilon * yr)
    ymax_rect = int(ymax + epsilon * yr)
    coord = (xmin_rect, xmax_rect, ymin_rect, ymax_rect)
    # new landmarks (on the rectangle sides)
    top_points = np.array([[x, ymin_rect] for x in np.linspace(xmin_rect, xmax_rect, 15)])
    bottom_points = np.array([[x, ymax_rect] for x in np.linspace(xmin_rect, xmax_rect, 20)])
    side_points = np.linspace(int(ymin_rect + yr * 1.1 / 12.0), int(ymax_rect - yr * 1.1 / 12.0), 11)
    left_points = np.array([[xmin_rect, y] for y in side_points])
    right_points = np.array([[xmax_rect, y] for y in side_points])
    # all points for the triangulation
    lm_points = np.array([[x, y] for (x, y) in lm])
    all_points = np.concatenate((lm_points, top_points, right_points, bottom_points, left_points))
    # print all_points
    return all_points, coord


# Delaunay triangulation of the points
def delaunayTriangulation(points):
    tri = Delaunay(points)
    # plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # plt.show()
    return tri.simplices

def boundingRect(triangle,xmax,ymax):
    x=floor(np.amin(triangle[:,0]))
    y=floor(np.amin(triangle[:,1]))
    xx=min(xmax,ceil(np.amax(triangle[:,0])))
    yy=min(ymax,ceil(np.amax(triangle[:,1])))
    return (int(x),int(y),int(xx-x),int(yy-y))

# base_points: coordinates of the landmark points of reference image
# triangulation: Delaunay triangulation of the base points
def warpImage(img, triangulation, base_points, coord):
    all_points, co = processImage(img)
    img_out = 255 * np.ones(img.shape, dtype=img.dtype)
    for t in triangulation:
        # triangles to map one another
        src_tri = np.array([[all_points[x][0], all_points[x][1]] for x in t]).astype(np.float32)
        dest_tri = np.array([[base_points[x][0], base_points[x][1]] for x in t]).astype(np.float32)
        # bounding boxes
        src_rect = cv2.boundingRect(np.array([src_tri]))
        dest_rect = cv2.boundingRect(np.array([dest_tri]))
        #src_rect=boundingRect(src_tri)
        #dest_rect=boundingRect(dest_tri)

        # crop images
        src_crop_tri = np.zeros((3, 2), dtype=np.float32)
        dest_crop_tri = np.zeros((3, 2))
        for k in range(0, 3):
            for dim in range(0, 2):
                src_crop_tri[k][dim] = src_tri[k][dim] - src_rect[dim]
                dest_crop_tri[k][dim] = dest_tri[k][dim] - dest_rect[dim]
        src_crop_img = img[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
        # affine transformation estimation
        mat = cv2.getAffineTransform(np.float32(src_crop_tri), np.float32(dest_crop_tri))
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
            img_out[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] * (
                (1.0, 1.0, 1.0) - mask)

        img_out[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = \
            img_out[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] + dest_crop_img

    return img_out[coord[2]:coord[3], coord[0]:coord[1]]


# meshAlign function --> maps all the triangles of img to the triangles of imgref
def meshAlign(img, imgref):
    bp, coord = processImage(imgref)
    tr = delaunayTriangulation(bp)
    img_out = warpImage(img, tr, bp, coord)
    # print img_out.shape
    return img_out


########################################################################################################################
########################################### TOTAL ALIGN FUNCTION #######################################################
########################################################################################################################

def preprocess(img, imgref):
    im = meshAlign(img, imgref)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    im = align(im)
    return im



########################################################################################################################
################################### FUNCTIONS NOT USED ANY MORE ########################################################
########################################################################################################################


# NOT USED (in the align part)
# to find the good values for the constants LEFT_EYE_POS, etc
def initializeParameters(repo):
    listImages = sorted(listdir(repo))
    n = len(listImages)
    nArray = np.zeros(2)
    cArray = np.zeros(2)
    lArray = np.zeros(2)
    rArray = np.zeros(2)
    mArray = np.zeros(2)
    mouthArray = np.zeros(2)
    for im in listImages:
        print im
        img = cv2.imread(repo + im)
        nose, chin, le, re, me, mouth = usefulPoints(img, False)
        nArray += np.array(nose)
        cArray += np.array(chin)
        lArray += np.array(le)
        rArray += np.array(re)
        mArray += np.array(me)
        mouthArray += np.array(mouth)
    nArray = nArray / (1.0 * n)
    cArray = cArray / (1.0 * n)
    lArray = lArray / (1.0 * n)
    rArray = rArray / (1.0 * n)
    mArray = mArray / (1.0 * n)
    mouthArray = mouthArray / (1.0 * n)
    LEFT_EYE_POS = tuple(lArray)
    EYES_SPACE = dist(tuple(lArray), tuple(rArray))
    FACE_HEIGHT = dist(tuple(mArray), tuple(cArray))
    FACE_HEIGHT2 = dist(tuple(mArray), tuple(mouthArray))
    print LEFT_EYE_POS, EYES_SPACE, FACE_HEIGHT, FACE_HEIGHT2

def draw_triangulation(im, tri, bp):
    img = im.copy()
    for t in tri:
        n1=t[0]
        n2=t[1]
        n3=t[2]
        pt1=(int(bp[n1][0]),int(bp[n1][1]))
        pt2 = (int(bp[n2][0]), int(bp[n2][1]))
        pt3 = (int(bp[n3][0]), int(bp[n3][1]))
        cv2.line(img,pt1,pt2,(0,0,255))
        cv2.line(img,pt3,pt2,(0,0,255))
        cv2.line(img, pt1, pt3, (0, 0, 255))
    return img

### print/save my photo cropped
# im="../photomoi.jpg"
# img = cv2.imread(im)
# img_align=align(img)
# cv2.imshow("align",img_align)
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imwrite("me_crop.jpg",img_align)


### print/save warp image
# im = "../federer.jpg"
# imref = "../tete6.jpg"
# img = cv2.imread(im)
# img_ref = cv2.imread(imref)
# img_warp = meshAlign(img, img_ref)
# cv2.imshow("warp", img_warp)
# cv2.waitKey()
# cv2.destroyAllWindows()

### print/save triangulation
# save image as before
# allpoints, co = processImage(img)
# bp, coord=processImage(img_ref)
# tr = delaunayTriangulation(bp)
# img_tri = draw_triangulation(img, tr, allpoints)
# cv2.imshow("tri", img_tri)
# cv2.waitKey()
# cv2.destroyAllWindows()

### images to test
# im="../federer.jpg"
# im="../LFW_verybig/David_Beckham/David_Beckham_0009.jpg"
# im="../LFW_verybig/Gordon_Brown/Gordon_Brown_0009.jpg"
# im="../LFW_verybig/Recep_Tayyip_Erdogan/Recep_Tayyip_Erdogan_0002.jpg"
# im="../LFW_verybig/Angelina_Jolie/Angelina_Jolie_0009.jpg"
# im="../LFW_verybig/Hillary_Clinton/Hillary_Clinton_0004.jpg"
# im="../LFW_verybig/Queen_Elizabeth_II/Queen_Elizabeth_II_0013.jpg"
# im="../LFW_verybig/George_W_Bush/George_W_Bush_0036.jpg"
# im="testgulechec.jpg"
# im='../tete.jpg'
# im="../LFW_verybig/Bill_Clinton/Bill_Clinton_0002.jpg"


