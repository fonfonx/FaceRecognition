import cv2
import dlib
import numpy as np

from config import *

predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)


def get_landmarks(img, detect_face=True):
    """ Return the landmarks of the image """
    x = 0
    y = 0
    w, h = img.shape[:2]
    if detect_face:
        rects = cascade.detectMultiScale(img, 1.3, 5)
        if len(rects) >= 1:
            rects = rects[np.argsort(rects[:, 3])[::-1]]
            x, y, w, h = rects[0].astype(long)
            x = x.item()
            y = y.item()
            w = w.item()
            h = h.item()
    rect = dlib.rectangle(x, y, x + w, y + h)
    return np.array([(p.x, p.y) for p in predictor(img, rect).parts()])


def annotate_landmarks(im, landmarks):
    """ Draw red points corresponding to the landmarks on the image """
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.circle(im, pos, 2, color=(255, 0, 0))
        cv2.circle(im, pos, 1, color=(255, 0, 0))
        cv2.circle(im, pos, 0, color=(255, 0, 0))
    return im


def get_landmarks_image(image_path, image_path_to_save="./landmarks.jpg", save=False):
    """ Show and return an image with the facial landmarks and optionally save it"""
    im = cv2.imread(image_path)
    im = cv2.resize(im, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_CUBIC)
    im_land = annotate_landmarks(im, get_landmarks(im))
    cv2.imshow('Image with landmarks', im_land)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save:
        cv2.imwrite(image_path_to_save, im_land)

if __name__ == '__main__':
    im_path = '../lfw2/Angelina_Jolie/Angelina_Jolie_0001.jpg'
    get_landmarks_image(im_path)
