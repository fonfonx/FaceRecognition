import numpy as np
from math import *
from numpy import linalg as lg
import sys


def mean_row(matrix):
    """ Return the mean row of a matrix """
    matrix = matrix.astype(float)
    n, m = matrix.shape
    mean = np.zeros(n)
    for i in range(n):
        mean[i] = (sum(matrix[i, :])) / m
    return mean


def zeromean_row(matrix):
    """ Return a zero mean row matrix from the given matrix """
    n, m = matrix.shape
    meanR = mean_row(matrix)
    zmatrix = np.zeros((n, m))
    for i in range(n):
        zmatrix[i, :] = matrix[i, :] - meanR[i]
    return zmatrix


def biggest_eigen_k(matrix, K):
    """ Return the K eigenvector/eigenvalue pairs associated to the K greatest eigenvalues """
    eig_val, eig_vec = lg.eig(matrix)
    order = eig_val.argsort()[::-1]
    eig_val = eig_val[order]
    eig_vec = eig_vec[:, order]
    return eig_vec[:, :K], eig_val[:K]


def pca_reductor(Y, K):
    """
    Return the matrix W of the SVD

    To reduce the matrix M (p rows -> features, n columns -> test samples) we will only have to return W.transpose()*M
    """
    p, n = Y.shape
    Y = zeromean_row(Y)
    if p <= n:
        matrix = Y.dot(Y.transpose()) / (n - 1)
        eig_vec, eig_val = biggest_eigen_k(matrix, K)
        return eig_vec
    else:
        matrix = Y.transpose().dot(Y) / (n - 1)
        eig_vec, eig_val = biggest_eigen_k(matrix, K)
        eig_vec2 = np.zeros((p, K))
        Y = Y / sqrt(n - 1)
        for i in range(K):
            eig_vec2[:, i] = (1 / sqrt(eig_val[i])) * Y.dot(eig_vec[:, i])
        return eig_vec2
