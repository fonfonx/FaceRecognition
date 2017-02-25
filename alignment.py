"""
Functions performing face alignment (ie preprocessing tasks)
Face alignment is done in two parts: mesh align with triangulation and manual alignment
"""

from math import *
from os import listdir
from scipy.spatial import Delaunay
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np

from config import *
from landmark import get_landmarks

predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)


def dist(tupleA, tupleB):
    """ Distance between two points in R^2 """
    return sqrt((tupleB[0] - tupleA[0]) ** 2 + (tupleB[1] - tupleA[1]) ** 2)


def detect_face(img):
    """ Function detecting the biggest face in an image """
    rects = cascade.detectMultiScale(img, 1.3, 5)
    rects = rects[np.argsort(rects[:, 3])[::-1]]
    x, y, w, h = rects[0]
    return img[y:y + h, x:x + w]


##############################
# Manual Alignment Functions #
##############################

def useful_points_on_face(img, detect_face):
    """ Return the position of the nose, eyes and chin """
    lm = get_landmarks(img, detect_face)

    nose = lm[30]
    left_eye = tuple((np.array(lm[37]) + np.array(lm[38]) + np.array(lm[40]) + np.array(lm[41])) / 4.0)
    right_eye = tuple((np.array(lm[43]) + np.array(lm[44]) + np.array(lm[46]) + np.array(lm[47])) / 4.0)
    chin = lm[8]
    mideye = lm[27]
    # mouth
    mouth_array = np.zeros(2)
    for i in range(48, 68):
        mouth_array += np.array(lm[i])
    mouth = tuple(mouth_array / 20.0)

    return nose, chin, left_eye, right_eye, mideye, mouth


def rotate(img, angle, center):
    """ Rotate the image around center of angle """
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, (int(img.shape[1] * 1.0), int(img.shape[0] * 1.0)), flags=cv2.INTER_LINEAR)

    return result


def translation(img, vec):
    """ Translate the image according to the corresponding vector """
    tx = vec[0]
    ty = vec[1]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def align(img, display=False, save=False):
    """ Manually align the image """
    nose, chin, left_eye, right_eye, mideye, mouth = useful_points_on_face(img, False)  # False except for myface

    # 1st step
    # rotation around left eye
    eye_vector = np.array(right_eye) - np.array(left_eye)
    theta = atan2(eye_vector[1], eye_vector[0])
    img_rot = rotate(img, theta * 180.0 / pi, left_eye)
    # resizing
    eye_space = dist(left_eye, right_eye)
    face_height = dist(chin, mideye)
    # eye_mouth=dist(me,mouth)
    x_factor = EYES_SPACE / eye_space
    # y_factor = EYE_MOUTH / eye_mouth
    y_factor = FACE_HEIGHT / face_height
    factor = (x_factor + y_factor) / 2.0
    img_res = cv2.resize(img_rot, None, fx=x_factor, fy=y_factor, interpolation=cv2.INTER_CUBIC)

    if display:
        cv2.imshow("First alignment", img_res)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite("first_alignment.jpg", img_res)

    # 2nd step
    img = img_res
    # translation
    left_eye = (left_eye[0] * x_factor, left_eye[1] * y_factor)
    transl_vec = tuple(np.array(LEFT_EYE_POS) - np.array(left_eye))
    img_t = translation(img, transl_vec)
    # crop
    crop_img = img_t[0:HEIGHT, 0:WIDTH]
    img = crop_img

    if display:
        cv2.imshow("Manual Alignment", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite("manual_alignment.jpg", img)

    return img


###########################
# IMAGE WARPING FUNCTIONS #
###########################

def preprocess_image_before_triangulation(img):
    """
    Perform preprocessing of the image
    Return all the points that will be used for the triangulation and the coordinates of the rectangle around the face
    """
    # landmark extraction
    lm = get_landmarks(img, True)  # change if not LFW (True for LFW)

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

    return all_points, coord


def delaunay_triangulation(points, plot=False):
    """ Extract a Delaunay's triangulation from the points """
    tri = Delaunay(points)
    if plot:
        plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
        plt.plot(points[:, 0], points[:, 1], 'o')
        plt.show()
    return tri.simplices


# base_points: coordinates of the landmark points of reference image
# triangulation: Delaunay triangulation of the base points
def warpImage(img, triangulation, base_points, coord):
    all_points, co = preprocess_image_before_triangulation(img)
    img_out = 255 * np.ones(img.shape, dtype=img.dtype)
    for t in triangulation:
        # triangles to map one another
        src_tri = np.array([[all_points[x][0], all_points[x][1]] for x in t]).astype(np.float32)
        dest_tri = np.array([[base_points[x][0], base_points[x][1]] for x in t]).astype(np.float32)
        # bounding boxes
        src_rect = cv2.boundingRect(np.array([src_tri]))
        dest_rect = cv2.boundingRect(np.array([dest_tri]))
        # src_rect=boundingRect(src_tri)
        # dest_rect=boundingRect(dest_tri)

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
    bp, coord = preprocess_image_before_triangulation(imgref)
    tr = delaunay_triangulation(bp)
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

def draw_triangulation(im, tri, bp):
    img = im.copy()
    for t in tri:
        n1 = t[0]
        n2 = t[1]
        n3 = t[2]
        pt1 = (int(bp[n1][0]), int(bp[n1][1]))
        pt2 = (int(bp[n2][0]), int(bp[n2][1]))
        pt3 = (int(bp[n3][0]), int(bp[n3][1]))
        cv2.line(img, pt1, pt2, (0, 0, 255))
        cv2.line(img, pt3, pt2, (0, 0, 255))
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
# im = "../GWBush36.jpg"
# imref = "../tete6.jpg"
# img = cv2.imread(im)
# img_ref = cv2.imread(imref)
# img_warp = meshAlign(img, img_ref)
# cv2.imshow("warp", img_warp)
# cv2.imwrite("bush_warp.jpg", img_warp)
# cv2.waitKey()
# cv2.destroyAllWindows()

# preprocess(img, img_ref)

### print/save triangulation
## save image as before
# allpoints, co = preprocess_image_before_triangulation(img)
# bp, coord=preprocess_image_before_triangulation(img_ref)
# tr = delaunay_triangulation(bp)
# img_tri = draw_triangulation(img, tr, allpoints)
# cv2.imshow("tri", img_tri)
# cv2.imwrite("tete6_tri.jpg", img_tri)
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
