# config file containing some global variables and paths

# paths for landmarks and cascades
PREDICTOR_PATH = "/home/xavier/dlib-18.18/shape_predictor_68_face_landmarks.dat"
# PREDICTOR_PATH = "/root/Programs/dlib-18.18/shape_predictor_68_face_landmarks.dat"

CASCADE_PATH="/home/xavier/opencv/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml"
# CASCADE_PATH="/root/Programs/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml"


# constants for face alignment
LEFT_EYE_POS = (6, 6)
EYES_SPACE = 18
FACE_HEIGHT = 32  # from eyes to chin
EYE_MOUTH = 16
HEIGHT = 30
WIDTH = 30