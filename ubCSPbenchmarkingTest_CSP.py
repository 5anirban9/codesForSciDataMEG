from scipy.io import loadmat
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from userDefFunc import my_function
from userDefFunc import f_logVar
import pickle

from sklearn import svm
from sklearn.model_selection import cross_val_score

subList = ["P01", "P02", "P03", "P04","P05", "P06", "P07","P08", "P09", "P10"]

info = np.empty(10)

for sb in range(10):
    #   for subID in subList:
    subID=subList[sb]

    dataSRC = "C:\\Users\\User\\Dropbox\\ClinicalBCIChallengeWCCI-2020-FullDataset\\"
    dataSRC1="E:\\Learning_python_tkinter\\"
    fnameWithPath=dataSRC+"parsed_"+subID+"E.mat"
    fnameWithPath1=dataSRC1+"trainedModel"+subID

    print(fnameWithPath)
    print(fnameWithPath1)

    subDataFile = loadmat(fnameWithPath)
    RawEEGData = subDataFile["RawEEGData"]
    Labels = subDataFile["Labels"]
    sampRate = subDataFile["sampRate"]

    dim=RawEEGData.shape

    nOfTrials=dim[0] #getting the number of trials
    #print(nOfTrials)
    nOfChann=dim[1]  #getting the number of channels
    #print(nOfChann)
    nOfSampsInTrial=dim[2] #getting the number of samples per trial
    #print(nOfSampsInTrial)

    with open(fnameWithPath1, 'rb') as f1:
       rd = pickle.load(f1)
    #  print(rd[4])

    svmMdl = rd[0]
    wCSP_mu = rd[1]
    wCSP_beta = rd[2]
    muBand = rd[3]
    betaBand = rd[4]

    #print(svmMdl)
    #print(wCSP_mu)
    #print(wCSP_beta)
    #print(muBand)
    #print(betaBand)

    order = 4

    mu_B, mu_A = signal.butter(order, muBand, 'bandpass', analog=False)
    beta_B, beta_A = signal.butter(order, betaBand, 'bandpass', analog=False)

    muRawEEGData=np.empty((nOfTrials, nOfChann, nOfSampsInTrial))
    betaRawEEGData=np.empty((nOfTrials, nOfChann, nOfSampsInTrial))

    for trlIndex in range(nOfTrials):

        sig=RawEEGData[trlIndex,:,:]
    #print("Properties:")
    #print(type(sig))
    #print(sig.shape)
        mu_temp=signal.lfilter(mu_B, mu_A, sig,1)
        beta_temp=signal.lfilter(beta_B, beta_A, sig,1)

        muRawEEGData[trlIndex,:,:]= mu_temp #altering the dimensions
        betaRawEEGData[trlIndex,:,:]= beta_temp #altering the dimensions
        #print(trlIndex)
        #print(muRawEEGData.shape)
        #print(betaRawEEGData.shape)

    labelsCls1 = np.where(Labels == 1)
    labelsCls1=labelsCls1[0]
    labelsCls2 = np.where(Labels == 2)
    labelsCls2=labelsCls2[0]

    feat = np.empty((nOfTrials,4))

    for trlIndex in range(nOfTrials):

        muTemp=muRawEEGData[trlIndex,:,:]
        betaTemp=betaRawEEGData[trlIndex,:,:]

        mu_temp = np.matmul(wCSP_mu,muTemp);  #calculating the Z matrix for mu band
        beta_temp = np.matmul(wCSP_beta,betaTemp);   #calculating the Z matrix for beta band

        logVarMu=f_logVar(mu_temp);      #calculating the logvariance for Mu
        logVarBeta=f_logVar(beta_temp);  #calculating the logvariance for Beta

        feat[trlIndex, :]=[logVarMu[0],  logVarMu[len(logVarMu)-1],  logVarBeta[0],  logVarBeta[len(logVarBeta)-1]]

    Test_X = feat
    Test_Y = Labels[:,0]

    Train_X = Test_X
    labelsCls1 = np.where(Labels == 1)
    labelsCls1=labelsCls1[0]
    labelsCls2 = np.where(Labels == 2)
    labelsCls2=labelsCls2[0]


    Test_Y_predicted = svmMdl.predict(Test_X)

    testAcc = accuracy_score(Test_Y, Test_Y_predicted)
    print(testAcc)
    info[sb]=testAcc


    #fbf=0
    #sbf=1
    #x1 = Train_X[labelsCls1,fbf]
    #y1 = Train_X[labelsCls1,sbf]

    #x2 = Train_X[labelsCls2,fbf]
    #y2 = Train_X[labelsCls2,sbf]

    #fig=plt.figure()
    #ax=fig.add_axes([0,0,1,1])
    #ax.scatter(x1, y1, color='b')
    #ax.scatter(x2, y2, color='r')
    #ax.set_xlabel('Grades Range')
    #ax.set_ylabel('Grades Scored')
    #ax.set_title('scatter plot')
    #plt.show()

print("meanCVaccAllSub", np.mean(info))






