import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import os
from tqdm import tqdm
import sys, getopt
import pickle
from scipy.signal import find_peaks
from utils import *

def calcTimeSeriesAutoCorr(V,k):
    alpha = np.mean(V)
    nomi = 0
    V1 = V[:-k]-alpha
    V2 = V[k:]-alpha
    nomi = np.sum(V1*V2)
    denomi = np.sum((V-alpha)**2)
    #print(nomi)
    return nomi/denomi

def main(argv):
    inputPATH = ''
    outputPATH = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["iPATH=","oPATH="])
    except getopt.GetoptError:
        print ('test.py -i <inputPATH> -o <outputPATH>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputPATH> -o <outputPATH>')
            sys.exit()
        elif opt in ("-i", "--iPATH"):
            inputPATH = arg
        elif opt in ("-o", "--oPATH"):
            outputPATH = arg
    print ('Input PATH is ', inputPATH)
    print ('Output PATH is ', outputPATH)
    try:
        os.mkdir(outputPATH)
    except:
        pass
    folderList = os.listdir(inputPATH)
    AutocorrelationMaps = []
    AutocorraltionTimes = []
    for d in tqdm(folderList):
        if len(d)==2:
            path = inputPATH + '/' + d
            fileList =  os.listdir(path)

            for f in tqdm(fileList):
                currFile = f'{inputPATH}/{d}/{f}/{f}.features.txt'
                try:

                    # X = pd.read_csv(currFile,header=None ,usecols=range(122),skiprows=4,index_col=0)
                    # x0,x1 = -5,0
                    # X_mat = X.values
                    # X_mat[:,33:38] = X_mat[:,33:38]-[3.2,1,0.5,0.3,0]
                    # X_mat_Transformed = np.minimum(np.ones(X_mat.shape),np.tanh(4*(X_mat-x0)/(x1-x0)-2))
                    X_mat_Transformed = load_data(currFile)
                    r_vecs = []
                    AutoCorrTimes = []
                    for i in range(X_mat_Transformed.shape[1]):
                        r = []
                        for k in np.arange(1,7200,10):
                            r.append(calcTimeSeriesAutoCorr(X_mat_Transformed[:,i],k))
                        r_max = max(r)
                        r_lower = np.argwhere(r<=(r_max/math.exp(1)))
                        temp_time = 10*r_lower[0]
                        r_vecs.append(r)
                        AutoCorrTimes.append(temp_time)
                    AutocorrelationMaps.append(r_vecs)
                    AutocorraltionTimes.append(AutoCorrTimes)
                    # print(currFile)
                except:
                    pass
    DecayTimeMat = [np.array(x).flatten() for x in AutocorraltionTimes]
    DecayTimeMat = np.array(DecayTimeMat)
    plt.figure()
    plt.hist(DecayTimeMat.flatten(),50)
    plt.title('Decay Time distribution')
    plt.xlabel('Time [s]')
    plt.savefig(f'{outputPATH}/DecayTimeAll.jpg')
    plt.figure()
    plt.hist(np.median(DecayTimeMat,axis=0),20)
    plt.title('Decay Time median')
    plt.ylabel('Time [s]')
    plt.xlabel('Feature')
    plt.savefig(f'{outputPATH}/DecayTimeMedian.jpg')
    C,E = np.histogram(np.median(DecayTimeMat,axis=0),20)
    plt.figure()
    plt.plot(np.cumsum(C))
    plt.plot(np.diff(np.cumsum(C)))
    plt.title('Decay Time cumsum')
    plt.savefig(f'{outputPATH}/DecayTimeHistsCumSum.jpg')

    pickle_out = open(f'{outputPATH}/DecayTimeMat',"wb")
    pickle.dump(DecayTimeMat, pickle_out)
    pickle_out.close()

    pickle_out = open(f'{outputPATH}/AutocorrelationMaps',"wb")
    pickle.dump(AutocorrelationMaps, pickle_out)
    pickle_out.close()

    # C = np.median(DecayTimeMat,axis=0)
    C,E = np.histogram(np.median(DecayTimeMat,axis=0),20)
    P,_ = find_peaks(np.diff(np.cumsum(C)))
    # print(P)
    # print(E)

    plt.figure(figsize=(21,10))
    plt.boxplot(DecayTimeMat)

    plt.xticks(rotation=90)
    plt.ylabel('Decay Time (s)')
    plt.xlabel('Feature number')
    plt.savefig(f'{outputPATH}/DecayTimeDist.jpg')

    DecayTimeMatHists = []
    for i in range(DecayTimeMat.shape[1]):
        tempHistC,_ = np.histogram(DecayTimeMat[:,i],bins=30,range=[np.min(DecayTimeMat),np.max(DecayTimeMat)])
        DecayTimeMatHists.append(tempHistC)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(DecayTimeMatHists))

    #
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(DecayTimeMat)
    plt.figure(figsize=(21,10))
    plt.subplot(2,1,1)
    plt.boxplot(DecayTimeMat[:,kmeans.labels_==0])
    plt.xticks(rotation=90)
    plt.ylabel('Decay Time (s)')
    plt.xlabel('Feature number')
    locs, _ = plt.xticks()
    labels = np.argwhere(kmeans.labels_==0)
    plt.xticks(locs, labels)
    plt.subplot(2,1,2)
    plt.boxplot(DecayTimeMat[:,kmeans.labels_==1])
    plt.xticks(rotation=90)
    plt.ylabel('Decay Time (s)')
    plt.xlabel('Feature number')
    locs, _ = plt.xticks()
    labels = np.argwhere(kmeans.labels_==1)
    plt.xticks(locs, labels)
    plt.savefig(f'{outputPATH}/DecayTimeDistClusters.jpg')
    #
    features_meds = np.array([np.median(DecayTimeMat[:,kmeans.labels_==0]),np.median(DecayTimeMat[:,kmeans.labels_==1])])
    shortFeatIdx = np.argmin(features_meds)
    ShortFeatures = np.argwhere(kmeans.labels_== shortFeatIdx)
    #print(ShortFeatures)

    pickle_out = open(f'{outputPATH}/ShortFeatures',"wb")
    pickle.dump(ShortFeatures, pickle_out)
    pickle_out.close()



if __name__ == "__main__":
    main(sys.argv[1:])
