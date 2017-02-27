""" Utility functions operating on matrices """

import numpy as np
from numpy import linalg as LA
from math import *


def norm_column(col):
    """ Norm of a column vector """
    return LA.norm(col)


def normalize_column(col):
    """ Normalize a column vector """
    col = col.astype(float)
    sq = norm_column(col)
    ncol = col / sq
    return ncol


def normalize_matrix(matrix):
    """ Normalize a matrix """
    n, m = matrix.shape
    nmatrix = np.zeros((n, m))
    for j in range(m):
        nmatrix[:, j] = normalize_column(matrix[:, j])
    return nmatrix


def dim_reduct(matrix, reductor, enable_dim_reduction=False):
    """ Perform dimensionality reduction given a reductor matrix """
    if enable_dim_reduction:
        return reductor.transpose().dot(matrix)
    else:
        return matrix


def mean_sample(mat):
    """ Return the mean column of a matrix """
    n, m = mat.shape
    mean = np.array([sum(mat[i, :]) for i in range(n)]) / (1.0 * m)
    return mean
