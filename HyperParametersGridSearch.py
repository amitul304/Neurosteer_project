import sys, os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import getopt
import pickle
from scipy.sparse.linalg import eigs
from scipy.signal import correlate, find_peaks
from sklearn.metrics import mean_squared_error
from utils import *

def find_period_time(x):
    minCycle4Sleep = 500
    maxCycle4Sleep = 1500
    s = correlate(x,x)[x.shape[0]+minCycle4Sleep:x.shape[0]+maxCycle4Sleep]/sum(x**2)
    pks,pks_data = find_peaks(s,prominence=1e-6)
    optPeak = np.argmax(s[pks])
    cycleTime = pks[optPeak]+minCycle4Sleep
    peakCorr = s[pks[optPeak]]
    return peakCorr, cycleTime

def manifold_score(U):
    resPerDim = [find_period_time(U[:,i]) for i in range(1,U.shape[1])]
    peakScore = np.array([x[0] for x in resPerDim])
    optDim = np.argmax(peakScore)
    optScore = peakScore[optDim]
    return optScore, optDim+1

# def graph_laplacian(X, sigma, alpha):
#     K = np.exp(-squareform(pdist(X))/(2*sigma**2))
#     Rho = np.sum(K, axis=1)
#     if alpha==1:
#         K = (1/Rho[:,np.newaxis])*K*(1/Rho[np.newaxis,:])
#     elif alpha == 2:
#         K = (Rho[:,np.newaxis])**-0.5*K*(1/Rho[np.newaxis,:])**-0.5
#     return K/np.sum(K, axis=1)
#
# def diff_map(X, MAX_K = 50):
#     l, U =  eigs(X, MAX_K,maxiter=100,tol=0.01)
#     l = np.real(l)
#     U = np.real(U)
#     return l,U

def main(argv):
    inputPATH = ''
    outputPATH = ''
    featuresPATH = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:f:",["iPATH=","oPATH=","fPATH"])
    except getopt.GetoptError:
        print ('HyperParametersGridSearch.py -i <inputPATH> -o <outputPATH> -f <featuresPATH>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('HyperParametersGridSearch.py -i <inputPATH> -o <outputPATH> -f <featuresPATH>')
            sys.exit()
        elif opt in ("-i", "--iPATH"):
            inputPATH = arg
            outputPATH = arg
        elif opt in ("-o", "--oPATH"):
            outputPATH = arg
        elif opt in ("-f", "--fPATH"):
            featuresPATH = arg
    print ('Input PATH is ', inputPATH)
    print ('Output PATH is ', outputPATH)
    print ('Features PATH is ', featuresPATH)
    try:
        os.mkdir(outputPATH)
    except:
        pass
    #currFile = 'ML_dataset/DM/00a3b4810811-2018-07-29-21-07-50/00a3b4810811-2018-07-29-21-07-50.features.txt'
    pathSplit = inputPATH.split('/')
    d = pathSplit[-1]
    currFile = currFile = f'{inputPATH}/{d}.features.txt'
    # X = pd.read_csv(currFile,header=None ,usecols=range(122),skiprows=4,index_col=0)
    # #X_mat = nsos.read_csv(r'Sleep/Benchmark',curFile,header=None ,usecols=range(122),skiprows=4,index_col=0)
    # X_mat = X.values
    # x0,x1 = -5,0
    # X_mat[:,33:38] = X_mat[:,33:38]-[3.2,1,0.5,0.3,0]
    # X_mat_Transformed = np.minimum(np.ones(X_mat.shape),np.tanh(4*(X_mat-x0)/(x1-x0)-2))
    X_mat_Transformed = load_data(currFile)
    # plt.figure()
    # plt.imshow(X_mat_Transformed.T,aspect='auto',cmap='jet')
    # plt.show()
    try:
        with open(f'{featuresPATH}/ShortFeatures', 'rb') as f:
            ShortFeatures = pickle.load(f)
    except:
        ShortFeatures = np.arange(0,X_mat_Transformed.shape[1])
    ShortFeatures = ShortFeatures.flatten()
    #print(len(ShortFeatures))
    # print(ShortFeatures)
    # plt.figure()
    # plt.imshow(X_mat_Transformed[:,ShortFeatures].T,aspect='auto',cmap='jet')
    # plt.show()
    # t = 400
    # i = 0
    # X_mean_mat = []
    # for i in range(0,X_mat_Transformed.shape[0]-t,5):
    #     curr_mat = X_mat_Transformed[i:i+t,:]
    #     curr_mean = np.mean(curr_mat,axis=0)
    #     X_mean_mat.append(curr_mean)
    #
    # X_mean_mat = np.stack(X_mean_mat,0)

    X_mean_mat = calcMeanMat(X_mat_Transformed)

    sigmas = np.logspace(-2,1,num=50)
    #alphas = [0,1,2]
    density = np.zeros(sigmas.shape)
    for i, eps in enumerate(sigmas):
        print(f'{i},',end='')
        #K = make_kernelMatrix(distmatrix=d, eps=eps, alpha=0.5)
        tK = np.exp(-squareform(pdist(X_mean_mat[:, ShortFeatures]))/(2*eps**2))
        d1 = np.sum(tK)
        density[i] = np.log10(d1)
    print('')
#     density.append(np.sum(np.log(d1)*d1**(-1))/np.sum(d1**(-1)))
    # plt.figure()
    # plt.scatter(np.log10(sigmas),density)
    # plt.show()
    plt.figure()
    plt.plot(np.log10(np.logspace(-2,1,num=50)),density)
    plt.savefig(f'{outputPATH}/Density.jpg')

    plt.figure()
    plt.plot(np.gradient(density))
    plt.savefig(f'{outputPATH}/DensityGradient.jpg')

    linIdx = np.argwhere(np.gradient(density)>np.percentile(np.gradient(density),75)).flatten()
    x_linPart = np.log10(np.logspace(-2,1,num=50))[linIdx]
    y_linPart = density[linIdx]

    plt.figure()
    plt.plot(x_linPart,y_linPart)
    plt.savefig(f'{outputPATH}/LinearDensity.jpg')

    #print((y_linPart[-1]-y_linPart[0])/(x_linPart[-1]-x_linPart[0]))

    sigmas = np.logspace(x_linPart[0], x_linPart[-1],11)
    #print(sigmas)

    optScore=1
    optAlpha = 0
    optSigma = 0
    optDim = 20
    errors = []
    for alpha in [0,1,2]:

        for sigma in sigmas:
            print('.', end='')
            K = graph_laplacian(X_mean_mat[:,ShortFeatures], sigma, alpha)
            try:
                l, U = diff_map(K,20)
                # plt.figure()
                # plt.plot(np.log10(l),'.')
                # plt.show()
                # coptScore, coptDim = manifold_score(U)
                PSD,res,rank,s=np.linalg.lstsq(U,X_mean_mat[:,ShortFeatures],rcond=None)
                XrecSmall = np.matmul(U,PSD)
                PSD,res,rank,s=np.linalg.lstsq(U,X_mean_mat,rcond=None)
                XrecFull = np.matmul(U,PSD)
                # print(XrecSmall.shape)
                # print(XrecFull.shape)
                SmallError = mean_squared_error(XrecSmall, X_mean_mat[:,ShortFeatures])
                FullError = mean_squared_error(XrecFull, X_mean_mat)
                #print((SmallError+FullError) < optScore)
                if (SmallError+FullError) < optScore:
                    optScore = (SmallError+FullError)
                    optAlpha = alpha
                    optSigma = sigma
                errors.append(SmallError+FullError)
                #print(SmallError+FullError)
                # if coptScore>optScore:
                #     optScore = coptScore
                #     optAlpha = alpha
                #     optSigma = sigma
                #     optDim = coptDim
            except:
                pass
    print('')
    #print((optAlpha, optSigma, optScore,optDim))
    pickle_out = open(f'{outputPATH}/OptParams',"wb")
    pickle.dump((optAlpha, optSigma), pickle_out)
    pickle_out.close()

    K = graph_laplacian(X_mean_mat[:,ShortFeatures], optSigma, optAlpha)
    l,U = diff_map(K,20)
    PSD,res,rank,s=np.linalg.lstsq(U,X_mean_mat,rcond=None)
    Xrec = np.matmul(U,PSD)

    plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    plt.imshow(np.array(X_mean_mat).T,aspect='auto',vmin=-0.6,vmax=1,cmap='jet')
    plt.colorbar()
    plt.title(f'Moving Mean t=400')
    plt.xlabel('Window #')
    plt.ylabel('Feature')
    plt.subplot(2,1,2)
    plt.imshow(Xrec.T,aspect='auto',
               vmin=-0.6,
               vmax=1,cmap='jet')
    #plt.imshow(np.abs(np.matmul(u_np[:,:15],x)).T,aspect='auto')
    plt.colorbar()
    plt.title(f'Reconstructed')
    plt.xlabel('Window #')
    plt.ylabel('Feature')
    plt.savefig(f'{outputPATH}/Reconstruction.jpg')

if __name__ == "__main__":
    main(sys.argv[1:])
