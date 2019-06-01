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

def graph_laplacian(X, sigma, alpha):
    K = np.exp(-squareform(pdist(X))/(2*sigma**2))
    Rho = np.sum(K, axis=1)
    if alpha==1:
        K = (1/Rho[:,np.newaxis])*K*(1/Rho[np.newaxis,:])
    elif alpha == 2:
        K = (Rho[:,np.newaxis])**-0.5*K*(1/Rho[np.newaxis,:])**-0.5
    return K/np.sum(K, axis=1)

def diff_map(X, MAX_K = 50):
    l, U =  eigs(X, MAX_K,maxiter=100,tol=0.01)
    l = np.real(l)
    U = np.real(U)
    return l,U

def load_data(currFile):
    X = pd.read_csv(currFile,header=None ,usecols=range(122),skiprows=4,index_col=0)
    #X_mat = nsos.read_csv(r'Sleep/Benchmark',curFile,header=None ,usecols=range(122),skiprows=4,index_col=0)
    X_mat = X.values
    x0,x1 = -5,0
    X_mat[:,33:38] = X_mat[:,33:38]-[3.2,1,0.5,0.3,0]
    X_mat_Transformed = np.minimum(np.ones(X_mat.shape),np.tanh(4*(X_mat-x0)/(x1-x0)-2))
    return X_mat_Transformed

def calcMeanMat(X_mat_Transformed,t=400,gap=5):
    X_mean_mat = []
    for i in range(0,X_mat_Transformed.shape[0]-t,gap):
        curr_mat = X_mat_Transformed[i:i+t,:]
        curr_mean = np.mean(curr_mat,axis=0)
        X_mean_mat.append(curr_mean)

    X_mean_mat = np.stack(X_mean_mat,0)
    return X_mean_mat
