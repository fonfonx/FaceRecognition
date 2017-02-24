# this file contains preprocessing function  (like Local Binary Patterns)

import numpy as np
import random
from math import *
from pca import KEigen


# binary comparison
def isBigger(a, b):
    if a >= b:
        return 1
    else:
        return 0

####################################################################################
######################### Local Binary Patterns ####################################
####################################################################################


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


def newpixel_lbp_mult(matrix, i, j, offset):
    ref = matrix[i, j]
    tab = []
    for k in range(-1, 2):
        for l in range(-1, 2):
            if (k, l) != (0, 0):
                tab.append(isBigger(matrix[i + k, j + l], ref))
    return listToInts(tab, offset)


def newpixel_lbp_extend(matrix, sl, i, j, offset):
    ref = matrix[i, j]
    tab = []
    for (u, v) in sl:
        tab.append(isBigger(matrix[u, v], ref))
    return listToInts(tab, offset)


def newpixel_lbp_patch(matrix, sl, i, j, offset, patch_size):
    ref=patch_value(matrix,i,j,patch_size, patch_size)
    tab=[]
    for (u,v) in sl:
        tab.append(isBigger(patch_value(matrix,u,v,patch_size, patch_size),ref))
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
    lpb_matrix = np.zeros((n - 2, 8 * (m - 2)))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            for k in range(8):
                lpb_matrix[i - 1, 8 * (j - 1) + k] = newpixel_lbp_mult(matrix, i, j, k)
    return lpb_matrix


def LBP_multiple2(matrix):
    n, m = matrix.shape
    lbp_matrix = np.zeros((n - 2, (m - 2),8))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            for k in range(8):
                lbp_matrix[i - 1, (j - 1) ,k] = newpixel_lbp_mult(matrix, i, j, k)
    return np.concatenate((lbp_matrix[:,:,0],lbp_matrix[:,:,1],lbp_matrix[:,:,2],lbp_matrix[:,:,3],lbp_matrix[:,:,4],lbp_matrix[:,:,5],lbp_matrix[:,:,6],lbp_matrix[:,:,7]),axis=0)


def random_lists(n,m,minus=1):
    rep=[]
    for i in range(n*m):
        sublist=[]
        for k in range(8):
            sublist.append((random.randint(0,n-minus),random.randint(0,m-minus)))
        rep.append(sublist)
    return rep

def LBP_extend(matrix):
    n,m=matrix.shape
    lbp_matrix=np.zeros((n,m,8))
    i=0
    j=0
    for sl in RL:
        for k in range(8):
            lbp_matrix[i,j,k]=newpixel_lbp_extend(matrix,sl,i,j,k)
        j+=1
        if (j==m):
            j=0
            i+=1
    return np.concatenate((lbp_matrix[:,:,0],lbp_matrix[:,:,1],lbp_matrix[:,:,2],lbp_matrix[:,:,3],lbp_matrix[:,:,4],lbp_matrix[:,:,5],lbp_matrix[:,:,6],lbp_matrix[:,:,7]),axis=0)


def LBP_patch(matrix, patch_size, step):
    n,m=matrix.shape
    new_n=n/step
    new_m=m/step
    lbp_matrix=np.zeros((new_n,new_m,8))
    for i in range(new_n-patch_size/step):
        for j in range(new_m-patch_size/step):
            for k in range(8):
                lbp_matrix[i,j,k]=newpixel_lbp_patch(matrix,RL[i*new_m+j],i*step,j*step,k,patch_size)
    return np.concatenate((lbp_matrix[:,:,0],lbp_matrix[:,:,1],lbp_matrix[:,:,2],lbp_matrix[:,:,3],lbp_matrix[:,:,4],lbp_matrix[:,:,5],lbp_matrix[:,:,6],lbp_matrix[:,:,7]),axis=0)



############################################################################
############################ PATCH #########################################
############################################################################

# value of a rectangular patch with dimensions h*l
def patch_value(matrix, i, j, h, l):
    rep=0
    for x in range(i, i + h):
        for y in range(j, j + l):
            rep += matrix[x, y]
    return rep*1.0/(h*l*1.0)


# tuple and dim are tuples
def diff_patch(matrix, tuple1, tuple2, dim1, dim2):
    return patch_value(matrix,tuple1[0],tuple1[1],dim1[0],dim1[1])-patch_value(matrix,tuple2[0],tuple2[1],dim2[0],dim2[1])
    #return isBigger(patch_value(matrix, tuple1[0], tuple1[1], dim1[0], dim1[1]),patch_value(matrix, tuple2[0], tuple2[1], dim2[0], dim2[1]))


def random_pairing(n, x, y, c):
    abs = range(0,x,c)
    ord = range(0,y,c)
    pairs1 = []
    pairs2 = []
    for i in range(n):
        pairs1.append((random.choice(abs), random.choice(ord)))
        pairs2.append((random.choice(abs), random.choice(ord)))
    return pairs1, pairs2


def patch_matrix(matrix, nb):
    n, m = matrix.shape
    rep = np.zeros((1, nb))
    for i in range(nb):
        rep[0, i] = diff_patch(matrix, pairs1[i], pairs2[i], (c, c), (c, c))
    return rep


#############################################################################
c = 4
ww=59
hh=43
#ww=51
#hh=51
pairs1, pairs2 = random_pairing(5 * ww*hh, ww-c, hh-c,c)
RL=random_lists(ww,hh,2)

#############################################################################

#############################################################################
############################# GRADIENT ######################################
#############################################################################

def gradx(matrix, i, j):
    return (float(matrix[i + 1, j]) - float(matrix[i - 1, j])) / 2.0


def grady(matrix, i, j):
    return (float(matrix[i, j + 1]) - float(matrix[i, j - 1])) / 2.0


def gradmat_x(matrix):
    n, m = matrix.shape
    grx = np.zeros((n - 2, m - 2))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            grx[i - 1, j - 1] = gradx(matrix, i, j)
    return grx


def gradmat_y(matrix):
    n, m = matrix.shape
    gry = np.zeros((n - 2, m - 2))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            gry[i - 1, j - 1] = gradx(matrix, i, j)
    return gry


def gradmat_ch(matrix):
    return np.concatenate((gradmat_x(matrix), gradmat_y(matrix)), axis=0)


def gradmat_norm(matrix):
    n, m = matrix.shape
    gradmat = np.zeros((n - 2, m - 2))
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            gradmat[i - 1, j - 1] = sqrt(gradx(matrix, i, j) ** 2 + grady(matrix, i, j) ** 2)
            # print gradmat[i-1,j-1]
    return gradmat


#############################################################################
############################## PCA ##########################################
#############################################################################

def meanColVec(X):
    n=len(X)
    return np.sum(X)*1.0/(1.0*n)

def pca(matrix,nbdim):
    X=np.transpose(matrix).flatten()
    X=X-meanColVec(X)
    X=X.reshape(len(X),1)
    mat=X.dot(X.transpose())
    eigVec,eigVal=KEigen(mat,nbdim)
    red=(X.transpose().dot(eigVec)).transpose()
    return red.reshape(nbdim)
    #return eigVec.transpose().dot(X)

#############################################################################

def preprocessing(matrix):
    n, m = matrix.shape
    return matrix
    # return gradmat_ch(matrix)
    # return LBP(matrix)
    # return LBP_multiple2(matrix)
    # return patch_matrix(matrix,5*n*m)
    # return LBP_extend(matrix)
    # return LBP_patch(matrix,3,3)
