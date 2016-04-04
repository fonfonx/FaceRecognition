# file used for PCA and Eigenface creation

import numpy as np
from math import *
from numpy import linalg as lg

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
	if p<=n:
		matrix=Y.dot(Y.transpose())
		eigVec,eigVal=KEigen(matrix,K)
		return eigVec
	else:
		matrix=Y.transpose().dot(Y)
		eigVec,eigVal=KEigen(matrix,K)
		eigVec2=np.zeros((p,K))
		for i in range(K):
			eigVec2[:,i]=Y.dot(eigVec[:,i])
		return eigVec2
