# this file contains the main function to create the dictionaries

import numpy as np
from PIL import Image
from os import listdir

# represent a number with a string of 'tot' characters
# pad with 0 if the length is less than tot
def fillStringNumber(val, tot):
    valstr = str(val)
    while (len(valstr) < tot):
        valstr = "0" + valstr
    return valstr


def columnFromImage(img):
    im = Image.open(img)
    im = im.convert("L")
    im = np.asarray(im)
    # im=dct(im)
    #im=fft2(im)
    #im=absMat(im)
    return np.transpose(im).flatten()


# nbFaces: number of faces per training person
def createTrainingDico(nbFaces, database):
    nbMen = 50
    nbWomen = 50
    listImages = []
    for i in range(1, nbMen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = "M-" + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            # nomImage = "M-" + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
            pathImage = database + nomImage
            listImages.append(columnFromImage(pathImage))
    for i in range(1, nbWomen + 1):
        for j in range(1, nbFaces + 1):
            nomImage = "W-" + fillStringNumber(i, 3) + "-" + fillStringNumber(j, 2) + ".bmp"
            # nomImage = "W-" + fillStringNumber(i, 3) + "-" + str(j) + ".bmp"
            pathImage = database + nomImage
            listImages.append(columnFromImage(pathImage))
    print "Creation of dictionary done"
    dico = (np.column_stack(listImages)).astype(float)
    return dico


# trainpart: percent of the repo used for training (rest -> testing)
# percent: percent of the repo used (globally)
# return trainSet, testSet, number of classes, number of training images and test images per class
def createDicosFromDirectory(repo, trainpart, percent=1.0):
    trainImages = []
    testImages = []
    directories = sorted(listdir(repo))
    nbClasses = len(directories)
    for d in directories:
        images = sorted(listdir(repo + d))
        n = int(percent * (len(images)))
        nb_train = 1
        train_max = int(trainpart * n)
        for i in np.random.permutation(range(n)):
            pathImage = repo + d + "/" + images[i]
            if nb_train <= train_max:
                nb_train += 1
                trainImages.append(columnFromImage(pathImage))
            else:
                testImages.append(columnFromImage(pathImage))
    nbFacesTrain = train_max
    nbFacesTest = n - train_max
    trainSet = (np.column_stack(trainImages)).astype(float)
    testSet = (np.column_stack(testImages)).astype(float)
    print "Training et Test sets have been created with success!"
    return trainSet, testSet, nbClasses, nbFacesTrain, nbFacesTest


def createDicoFromDirectory(repo):
    imagesArray = []
    directories = sorted(listdir(repo))
    for d in directories:
        images=sorted(listdir(repo+d))
        for image in images:
            pathImage=repo+d+"/"+image
            #print str(columnFromImage(pathImage).shape)+pathImage
            imagesArray.append(columnFromImage(pathImage))
    dico=(np.column_stack(imagesArray)).astype(float)
    print "dico created!"
    return dico