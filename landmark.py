import cv2
import dlib
import numpy as np

from config import *

predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)


def convert_to_long(x, y, w, h):
    """ Put x, y, w, h to the good format """
    x = long(x)
    y = long(y)
    w = long(w)
    h = long(h)
    return x, y, w, h


def get_landmarks(img, detect_face=True):
    """ Return the landmarks of the image """
    w, h = img.shape[:2]

    # default values
    x = 0.08 * w
    y = 0.08 * h
    w = 0.84 * w
    h = 0.84 * h
    x, y, w, h = convert_to_long(x, y, w, h)

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
    """ Draw blue points corresponding to the landmarks on the image """
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.circle(im, pos, 3, color=(255, 0, 0))
        cv2.circle(im, pos, 2, color=(255, 0, 0))
        cv2.circle(im, pos, 1, color=(255, 0, 0))
        cv2.circle(im, pos, 0, color=(255, 0, 0))
    return im


def get_landmarks_image(image_path, save=False, image_path_to_save="./landmarks.jpg"):
    """ Show and return an image with the facial landmarks and optionally save it """
    im = cv2.imread(image_path)
    im = cv2.resize(im, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_CUBIC)
    im_land = annotate_landmarks(im, get_landmarks(im))
    cv2.imshow('Image with landmarks', im_land)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save:
        cv2.imwrite(image_path_to_save, im_land)

if __name__ == '__main__':
    im_path = 'federer_image.jpg'
    get_landmarks_image(im_path, True, "federer_landmarks.jpg")
