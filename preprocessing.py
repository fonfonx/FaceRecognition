# this file contains preprocessing function  (like Local Binary Patterns)

import numpy as np


# binary comparison
def isBigger(a, b):
    if a >= b:
        return 1
    else:
        return 0


def listToInt(tab):
    rep = 0
    for a in tab:
        rep *= 2
        rep += a
    return rep


def listToInts(tab, offset):
    rep = 0
    n = len(tab)
    for i in range(n):
        rep *= 2
        rep += tab[(i + offset) % n]
    return rep


def listToMoy(tab):
    tot = 0
    for i in range(8):
        tot += listToInts(tab, i)
    return tot


def listToMax(tab):
    rep = listToInts(tab, 0)
    for i in range(1, 8):
        a = listToInts(tab, i)
        if a > rep:
            rep = a
    return rep


# auxiliary function for local binary patterns
# return the new value for pixel (i,j)
def newpixel_lbp(matrix, i, j):
    ref = matrix[i, j]
    tab = []
    for k in range(-1, 2):
        for l in range(-1, 2):
            if (k, l) != (0, 0):
                tab.append(isBigger(matrix[i + k, j + l], ref))
    return listToInt(tab)

def newpixel_lbp_mult(matrix,i,j,offset):
    ref = matrix[i, j]
    tab = []
    for k in range(-1, 2):
        for l in range(-1, 2):
            if (k, l) != (0, 0):
                tab.append(isBigger(matrix[i + k, j + l], ref))
    return listToInts(tab,offset)

# Local Binary Patterns
def LBP(matrix):
    n, m = matrix.shape
    lpb_matrix = np.zeros((n - 2, m - 2))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            lpb_matrix[i - 1, j - 1] = newpixel_lbp(matrix, i, j)
    return lpb_matrix


def LBP_multiple(matrix):
    n, m = matrix.shape
    lpb_matrix = np.zeros((n - 2, 8*(m - 2)))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            for k in range(8):
                lpb_matrix[i - 1, 8*(j - 1)+k] = newpixel_lbp_mult(matrix, i, j, k)
    return lpb_matrix


def preprocessing(matrix):
    return matrix
    #return LBP(matrix)
    #return LBP_multiple(matrix)
