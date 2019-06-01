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
    eps = 0.5
    alpha = 2.0
    try:
        opts, args = getopt.getopt(argv,"hi:o:f:e:a:",["iPATH=","oPATH=","fPATH","epsilon","alpha"])
    except getopt.GetoptError:
        print ('HyperParametersGridSearch.py -i <inputPATH> -o <outputPATH> -f <featuresPATH> -e <eps> -a <alpha>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('HyperParametersGridSearch.py -i <inputPATH> -o <outputPATH> -f <featuresPATH> -e <eps> -a <alpha>')
            sys.exit()
        elif opt in ("-i", "--iPATH"):
            inputPATH = arg
            outputPATH = arg
        elif opt in ("-o", "--oPATH"):
            outputPATH = arg
        elif opt in ("-f", "--fPATH"):
            featuresPATH = arg
        elif opt in ("-e", "--epsilon"):
            eps = float(arg)
        elif opt in ("-a", "--alpha"):
            alpha = float(arg)
    print ('Input PATH is ', inputPATH)
    print ('Output PATH is ', outputPATH)
    print ('Features PATH is ', featuresPATH)

    try:
        os.mkdir(outputPATH)
    except:
        pass
    try:
        with open(f'{inputPATH}/OptParams', 'rb') as f:
            alpha,eps = pickle.load(f)
            #print(alpha,eps)
    except:
        pass
    print('eps = ', eps)
    print('alpha = ', alpha)

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

    K = graph_laplacian(X_mean_mat[:,ShortFeatures], eps, alpha)
    l,U = diff_map(K,50)
    # print(X_mean_mat.shape)
    # print(U.shape)
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(np.array(X_mean_mat).T,aspect='auto',vmin=-0.6,vmax=1,cmap='jet')
    plt.colorbar()
    plt.title(f'Moving Mean t=400')
    plt.xlabel('Window #')
    plt.ylabel('Feature')
    plt.subplot(2,1,2)
    plt.imshow(U.T,aspect='auto',cmap='jet',vmin=np.percentile(U,5),vmax=np.percentile(U,95))
    plt.colorbar()
    plt.title(f'Embedding')
    plt.xlabel('Window #')
    plt.ylabel('Feature')
    plt.savefig(f'{outputPATH}/Enbedding.jpg')

    ConcatMat = np.hstack((X_mean_mat[:,ShortFeatures],U))
    corrcoefs = np.corrcoef(ConcatMat.T)
    #print(corrcoefs.shape)
    # plt.figure()
    # plt.imshow(corrcoefs)
    # plt.show()

    EmbedToFeatCorr = corrcoefs[:len(ShortFeatures),len(ShortFeatures):]
    # print(EmbedToFeatCorr.shape)
    # plt.figure()
    # plt.imshow(EmbedToFeatCorr)
    # plt.show()

    FeatScore = np.mean(EmbedToFeatCorr,axis=0)
    ToShowFeats = np.argwhere(FeatScore>=0.1).flatten()
    # for feat in ToShowFeats:
    #     plt.figure()
    #     plt.subplot(2,1,1)
    #     plt.imshow(np.array(X_mean_mat).T,aspect='auto',vmin=-0.6,vmax=1,cmap='jet')
    #     plt.title(f'Moving Mean t=400')
    #     plt.xlabel('Window #')
    #     plt.ylabel('Feature')
    #     plt.subplot(2,1,2)
    #     plt.plot(U[:,feat])
    #     plt.title(f'Embedded feature {feat}')
    #     plt.xlabel('Window #')
    #     plt.show()
    plt.figure()
    plt.plot(FeatScore,'.')
    plt.title('Features Scores')
    plt.xlabel('Feature')
    plt.ylabel('Score')
    plt.savefig(f'{outputPATH}/EmbeddedFeatScores.jpg')

    pickle_out = open(f'{outputPATH}/EmbeddedFeatureScores',"wb")
    pickle.dump(FeatScore, pickle_out)
    pickle_out.close()

if __name__ == "__main__":
    main(sys.argv[1:])
