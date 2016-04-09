# file used for PCA and Eigenface creation

import numpy as np
from math import *
from numpy import linalg as lg
import sys

# column vector containing the mean value of each row of the matrix
def meanRow(matrix):
	matrix=matrix.astype(float)
	n,m=matrix.shape
	mean=np.zeros(n)
	for i in range(n):
		mean[i]=(sum(matrix[i,:]))/m
	return mean

# return matrix-meanRow(matrix)
def zeroMeanRow(matrix):
	n,m=matrix.shape
	meanR=meanRow(matrix)
	zmatrix=np.zeros((n,m))
	for i in range(n):
		zmatrix[i,:]=matrix[i,:]-meanR[i]
	return zmatrix

# return K eigenvector/eigenvalue pairs associated
# to the K greatest eigenvalues
def KEigen(matrix,K):
	eigVal,eigVec=lg.eig(matrix)
	order=eigVal.argsort()[::-1]
	eigVal=eigVal[order]
	eigVec=eigVec[:,order]
	return eigVec[:,:K],eigVal[:K]

# return the matrix W of the SVD
# to reduce the matrix M (p rows -> features, n columns -> test samples)
# we only have to return W.transpose()*M

def PCA_reductor(Y, K):
	p,n=Y.shape
	Y=zeroMeanRow(Y)
	if p<=n:
		matrix=Y.dot(Y.transpose())/(n-1)
		eigVec,eigVal=KEigen(matrix,K)
		return eigVec
	else:
		matrix=Y.transpose().dot(Y)/(n-1)
		eigVec,eigVal=KEigen(matrix,K)
		eigVec2=np.zeros((p,K))
		Y=Y/sqrt(n-1)
		for i in range(K):
			eigVec2[:,i]=(1/sqrt(eigVal[i]))*Y.dot(eigVec[:,i])
		return eigVec2
