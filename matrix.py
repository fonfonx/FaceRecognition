# this file contains all function operating on matrices

import numpy as np
from numpy import linalg as LA
from math import *

def normColumn(col):
    return LA.norm(col)


def normalizeColumn(col):
    col = col.astype(float)
    sq = normColumn(col)
    ncol = col / sq
    return ncol


def normalizeMatrix(matrix):
    n, m = matrix.shape
    nmatrix = np.zeros((n, m))
    for j in range(m):
        nmatrix[:, j] = normalizeColumn(matrix[:, j])
    return nmatrix


def powerMatDiagSqrt(mat):
    n, m = mat.shape
    for i in range(n):
        mat[i, i] = sqrt(mat[i, i])
    return mat

# returns the module of each cell (if complex matrix)
def absMat(mat):
    n,m=mat.shape
    rep=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            rep[i,j]=abs(mat[i,j])
    return rep

def dimReduct(matrix, reductor):
    return reductor.transpose().dot(matrix)
    #return matrix
