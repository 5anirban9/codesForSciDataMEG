from scipy.io import loadmat
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from userDefFunc import my_function
from userDefFunc import f_logVar

from sklearn import svm
from sklearn.model_selection import cross_val_score

subList = ["P01", "P02", "P03", "P04","P05", "P06", "P07","P08", "P09", "P10"]

info = np.empty(10)
for sb in range(10):
    #   for subID in subList:
    subID=subList[sb]

    dataSRC = "C:\\Users\\User\\Dropbox\\ClinicalBCIChallengeWCCI-2020-FullDataset\\" #Change the path according to your computer wbere the patient dataset is there.
    fnameWithPath=dataSRC+"parsed_"+subID+"T.mat"
    print(fnameWithPath)

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

    order=4
    muBand=([8, 12]/sampRate)*2
    muBand=muBand[0]
    betaBand=([16, 24]/sampRate)*2
    betaBand=betaBand[0]

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

    muRawEEGDataCls1=muRawEEGData[labelsCls1,:,:] #mu band data from class 1 trials
    muRawEEGDataCls2=muRawEEGData[labelsCls2,:,:] #mu band data from class 2 trials

    betaRawEEGDataCls1=betaRawEEGData[labelsCls1,:,:] #beta band data from class 1 trials
    betaRawEEGDataCls2=betaRawEEGData[labelsCls2,:,:] #beta band data from class 2 trials

    #print(muRawEEGDataCls1.shape)
    #print(muRawEEGDataCls2.shape)
    #print(betaRawEEGDataCls1.shape)
    #print(betaRawEEGDataCls2.shape)

    muRawEEGDataCls1=np.swapaxes(muRawEEGDataCls1,0,1)
    muRawEEGDataCls2=np.swapaxes(muRawEEGDataCls2,0,1)
    betaRawEEGDataCls1=np.swapaxes(betaRawEEGDataCls1,0,1)
    betaRawEEGDataCls2=np.swapaxes(betaRawEEGDataCls2,0,1)

    #print(muRawEEGDataCls1.shape)
    #print(muRawEEGDataCls2.shape)
    #print(betaRawEEGDataCls1.shape)
    #print(betaRawEEGDataCls2.shape)

    cspMuRawEEGDataCls1 = np.reshape(muRawEEGDataCls1, (nOfChann, int(nOfTrials/2)*nOfSampsInTrial))
    cspMuRawEEGDataCls2 = np.reshape(muRawEEGDataCls2, (nOfChann, int(nOfTrials/2)*nOfSampsInTrial))
    cspBetaRawEEGDataCls1 = np.reshape(betaRawEEGDataCls1, (nOfChann, int(nOfTrials/2)*nOfSampsInTrial))
    cspBetaRawEEGDataCls2 = np.reshape(betaRawEEGDataCls2, (nOfChann, int(nOfTrials/2)*nOfSampsInTrial))

    #print(cspMuRawEEGDataCls1.shape)
    #print(cspMuRawEEGDataCls2.shape)
    #print(cspBetaRawEEGDataCls1.shape)
    #print(cspBetaRawEEGDataCls2.shape)

    wCSP_mu = my_function(cspMuRawEEGDataCls1,cspMuRawEEGDataCls2)
    wCSP_beta = my_function(cspBetaRawEEGDataCls1,cspBetaRawEEGDataCls2)

    #print("wCSP_mu", wCSP_mu)
    #print("wCSP_beta", wCSP_beta)

    feat = np.empty((nOfTrials,4))

    for trlIndex in range(nOfTrials):

        muTemp=muRawEEGData[trlIndex,:,:]
        betaTemp=betaRawEEGData[trlIndex,:,:]

        mu_temp = np.matmul(wCSP_mu,muTemp);  #calculating the Z matrix for mu band
        beta_temp = np.matmul(wCSP_beta,betaTemp);   #calculating the Z matrix for beta band

        logVarMu=f_logVar(mu_temp);      #calculating the logvariance for Mu
        logVarBeta=f_logVar(beta_temp);  #calculating the logvariance for Beta

        feat[trlIndex, :]=[logVarMu[0],  logVarMu[len(logVarMu)-1],  logVarBeta[0],  logVarBeta[len(logVarBeta)-1]]

    Train_X = feat
    Train_Y = Labels[:,0]

    clf = svm.SVC(kernel='linear', C=1)

    svmMdl = clf.fit(Train_X, Train_Y)
    #print(svmMdl.get_params(deep=True))
    Train_Y_predicted = svmMdl.predict(Train_X)
    from sklearn.metrics import accuracy_score
    trainAcc = accuracy_score(Train_Y_predicted, Train_Y)

    scores = cross_val_score(clf, Train_X, Train_Y, cv=10)
    meanCVacc = np.mean(scores)
    #print(meanCVacc)
    #print(sb)
    info[sb]=meanCVacc

    import pickle

    outputFilename="trainedModel"+subID
    with open(outputFilename, 'wb') as f:
        pickle.dump([svmMdl, wCSP_mu, wCSP_beta, muBand, betaBand], f)

print(info)
print("meanCVaccAllSub", np.mean(info))
    #with open('trainedModelP01', 'rb') as f1:
    #   rd = pickle.load(f1)
    #  print(rd[4])



    #x1 = Train_X[labelsCls1,2]
    #y1 = Train_X[labelsCls1,3]

    #x2 = Train_X[labelsCls2,2]
    #y2 = Train_X[labelsCls2,3]

    #fig=plt.figure()
    #ax=fig.add_axes([0,0,1,1])
    #ax.scatter(x1, y1, color='b')
    #ax.scatter(x2, y2, color='r')
    #ax.set_xlabel('Grades Range')
    #ax.set_ylabel('Grades Scored')
    #ax.set_title('scatter plot')
    #plt.show()

