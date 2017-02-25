""" config file containing some global variables and paths """

# paths for landmarks and cascades
PREDICTOR_PATH = "/home/xavier/HeavyPrograms/dlib/shape_predictor_68_face_landmarks.dat"

CASCADE_PATH = "/home/xavier/HeavyPrograms/opencv-2.4.10/data/haarcascades/haarcascade_frontalface_default.xml"

# constants for face alignment
LEFT_EYE_POS = (6, 6)
EYES_SPACE = 18
FACE_HEIGHT = 32  # from eyes to chin
EYE_MOUTH = 16
HEIGHT = 30
WIDTH = 30

# reference face image
IMREF_PATH = "../tete6.jpg"

# database paths
DATABASE_PATH = "../lfw2/"

# Parameters
TRAINING_FACES = 1
TEST_FACES = 1
NB_DIM = 120

# RSC algorithm params
NB_ITER = 2
PARAM_C = 8.0
PARAM_TAU = 0.8
LAMBDA = 0.001

REG_METHOD = 'l2'  # default (other possible value: l1)
