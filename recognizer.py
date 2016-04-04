# main file for face recognition
# use the paper about Robust Sparse Coding for Face Recognition

from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from math import *
import l1ls as L
import sys
from pca import PCA_reductor, KEigen 
from numpy import linalg as LA
import time

database="../AR_crop/"
nbDim=120
nbIter=2
param_c=8.0
param_tau=0.8
lmbda=0.001
rel_tol=0.01
classNum=100
nbFaces=7
eta=0.2
silence=False

#represent a number with a string of 'tot' characters
#pad with 0 if the length is less than tot
def fillStringNumber(val,tot):
    valstr=str(val)
    while(len(valstr)<tot):
        valstr="0"+valstr
    return valstr

def columnFromImage(img):
    im=Image.open(img)
    im=im.convert("L")
    im=np.asarray(im)
    return np.transpose(im).flatten()


#nbFaces: number of faces per training person
def createTrainingDico(nbFaces):
    nbMen=50
    nbWomen=50
    listImages=[]
    for i in range(1,nbMen+1):
        for j in range(1,nbFaces+1):
            nomImage="M-"+fillStringNumber(i,3)+"-"+fillStringNumber(j,2)+".bmp"
            pathImage=database+nomImage
            listImages.append(columnFromImage(pathImage))
    for i in range(1,nbWomen+1):
        for j in range(1,nbFaces+1):
            nomImage="W-"+fillStringNumber(i,3)+"-"+fillStringNumber(j,2)+".bmp"
            pathImage=database+nomImage
            listImages.append(columnFromImage(pathImage))
    print "Creation of dictionary done"
    return (np.column_stack(listImages)).astype(float)

def normColumn(col):
    return LA.norm(col)

def normalizeColumn(col):
    col=col.astype(float)
    sq=normColumn(col)
    col/=sq
    return col

def normalizeMatrix(matrix):
    n,m=matrix.shape
    #matrix=matrix.astype(float)
    for j in range(m):
        matrix[:,j]=normalizeColumn(matrix[:,j])
    return matrix

def powerMatDiagSqrt(mat):
    n,m=mat.shape
    for i in range(n):
        mat[i,i]=sqrt(mat[i,i])
    return mat

def pca(matrix, dim):
    matrix=np.transpose(matrix)
    pca_object=PCA(dim)
    pca_object.fit(matrix)
    return pca_object

def pcaReduc(pca_object,matrix):   
    matrix=np.transpose(matrix)
    reduct= pca_object.transform(matrix)
    return np.transpose(reduct)

def mean_sample(mat):
    n,m=mat.shape
    mean=np.array([sum(mat[i,:]) for i in range(n)])/(1.0*m)
    return mean

def fdelta(residual):
    n=len(residual)
    psi=residual**2
    psi=np.sort(psi)
    return psi[abs(param_tau*n)]

def classif(D,y,x,nbFaces):
    diff_tab=np.zeros(classNum)
    for c in range(classNum):
        xclass=x[nbFaces*c:nbFaces*(c+1)]
        Dclass=D[:,nbFaces*c:nbFaces*(c+1)]
        diff=y-Dclass.dot(xclass)
        diff_tab[c]=diff.dot(diff)
    if not(silence):
        print diff_tab
    return np.argmin(diff_tab)+1


def RSC_identif(TrainSet,TestNR,reductor):
    #TrainSet=reductor.transpose().dot(TrainSet)
    Test=reductor.transpose().dot(TestNR)


    NTrainSet=normalizeMatrix(TrainSet)
    ini=mean_sample(TrainSet)
    e=np.array((Test-ini).astype(float))
    delta=fdelta(e)
    mu=param_c/delta
    todiag=(1.0/(np.exp(mu*e**2 -mu*delta)+1.0))
    W=np.diag(todiag.flatten())
    #reductor=PCA_reductor(TrainSet,nbDim)
    for j in range(nbIter):
        W=powerMatDiagSqrt(W)
        #sys.exit("stop")
        WTrainSet=normalizeMatrix(W.dot(TrainSet))
        WTest=normalizeColumn(W.dot(Test))
        #D=normalizeMatrix(pcaReduc(reductor,WTrainSet))
        #y=normalizeColumn(pcaReduc(reductor,WTest))

        ###D=normalizeMatrix(reductor.transpose().dot(WTrainSet))
        ###y=normalizeColumn(reductor.transpose().dot(WTest))
        D=normalizeMatrix(WTrainSet)
        y=normalizeColumn(WTest)

        [x,status,hist]=L.l1ls(D,y,lmbda,quiet=True)
        NTest=normalizeColumn(Test)
        test_norm=normColumn(Test)
        
        if j==0:
            alpha=x
        else:
            alpha=alpha+eta*(x-alpha)
        
        if not(silence):
            print alpha
        
        e=test_norm*(NTest-NTrainSet.dot(alpha))
        delta=fdelta(e)
        mu=param_c/delta
        todiag=(1.0/(np.exp(mu*e*e-mu*delta)+1.0))
        W=np.diag(todiag.flatten())
    return classif(TrainSet,Test,alpha,nbFaces)  

def testdico(dico):
    n,m=dico.shape
    val=0
    for i in range(m):
        val+=dico[0,i]
    return val*1.0/(m*1.0)

def test_class(man,nbr,dico_red,reductor,nbMen):
    tot=0
    good=0
    if man:
        for j in range(nbFaces):
            k = 14 + j
            nomImage = "M-" + fillStringNumber(nbr, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            pathImage = database + nomImage
            y = columnFromImage(pathImage)
            classif = RSC_identif(dico_red, y, reductor)
            print "Class " + str(nbr) + " identified as " + str(classif)
            if classif == nbr:
                good += 1
            tot += 1
            # fichier=file('reponses.txt','a')
            # fichier.write(''+str(i)+' '+str(classif)+'\n')
            # fichier.close()
    else:
        for j in range(nbFaces):
            k = 14 + j
            nomImage = "W-" + fillStringNumber(nbr, 3) + "-" + fillStringNumber(k, 2) + ".bmp"
            pathImage = database + nomImage
            y = columnFromImage(pathImage)
            classif = RSC_identif(dico_red, y, reductor)
            print "Class " + str(nbMen + nbr) + " identified as " + str(classif)
            if classif == nbMen + nbr:
                good += 1
            tot += 1
            # fichier=file('reponses.txt','a')
            # fichier.write(''+str(nbMen+i)+' '+str(classif)+'\n')
            # fichier.close()
    return tot,good



def test_recognizer():
    #fichier=file('reponses.txt','w')
    #fichier.close()
    nbMen=50
    nbWomen=50
    tot=0
    good=0
    reductor=PCA_reductor(dico,nbDim)
    # dim reduction
    dico_red=reductor.transpose().dot(dico)
    print "PCA done"
    to_test_men={2,4,6,9,10,11,22,24,25,27,28,29,30,31,33,35,36,46,47,48}
    to_test_women={1,5,8,9,10,11,12,14,15,18,19,20,22,23,24,25,26,27,28,30,32,35,37,39,40,41,42,43,44,47,48,50}
    to_test_big_men={2,28,31,48}
    to_test_big_women={1,5,14,24,28,30,39,41,42,47,48}
    to_test={1,2}
    for i in to_test:#range(1,nbMen+1):
        tot_int,good_int=test_class(True,i,dico_red,reductor,nbMen)
        tot+=tot_int
        good+=good_int
    for i in to_test_big_women:#range(1,nbWomen+1):
        tot_int, good_int = test_class(False, i, dico_red, reductor, nbMen)
        tot += tot_int
        good += good_int
    rate=good*1.0/(tot*1.0)
    print "Recognition rate:",rate




dico=createTrainingDico(nbFaces)
#reductor=PCA_reductor(dico,nbDim)
#print "PCA done"
#y=columnFromImage(database+"M-001-15.bmp")

#print RSC_identif(dico,y,reductor)
test_recognizer()
print "fin"

