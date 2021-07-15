%Source of MEG data in my PC(01.07.2021+10:09AM):
%E:\UU-JAN-All-Documents-AUG(2019)\MEG_BRI2017-19\

%%Description of Pipeline1
%RawMEGdata(RMD) for S1 only=>RMD-MIonly(RMD-MI)=>RMD-MI-MotorCortex(RMD-MI-MC)=>
%separate from RMD-MI-MC the RMD-MI-MC-gradiometers(RMD-MI-MC-Grad)and RMD-MI-MC-magnetometers(RMD-MI-MC-mag)
%Then calculate CSP-W for all the three categories: RMD-MI-MC(cat-I), RMD-MI-MC-Grad(cat-II), RMD-MI-MC-mag(cat-III).
%Extract CSP features for cat-I, cat-II, and cat-III
%Do the 10-fold cross validation and compute accuracy (Acc) and kappa for each fold
%Average all the folds and get meanAcc and meanKappa
%Repeat the process for all the subjects and take the grand average

clc;
clear;
close all;

totSub=18;

sessID=2;
binClasses=[2 4];

for subID=1:totSub
tic

load(['TrainingAnalysisReport_Sub' num2str(subID) '_binClasses_' num2str(binClasses(1)) num2str(binClasses(2)) '.mat']);

disp('loading data....');
load(['E:\UU-JAN-All-Documents-AUG(2019)\MEG_BRI2017-19\TempData\ParsedMEGData_P' num2str(subID) '_S' num2str(sessID)]); %Load the MEG data
disp('data loaded.');

disp('extracting meg data');
rmd_ts=MEGdata.x;
allChannId_ts = MEGdata.ci;
allChann_ts = MEGdata.c; %Read all the channelcodes
labels_ts=MEGdata.y;
fs=MEGdata.s;

clear MEGdata 

rmd_ts=rmd_ts(:,commChannIndexes_ts,:);
disp('reorganizing dimensions');
rmd_ts = permute(rmd_ts,[2 1 3]);

allChann_ts=allChann_ts(commChannIndexes_ts);
chIndentity_ts = chIndentify(allChann_ts); %Map the channelcodes with cortex id: MotorCortex=1, LeftTemporalLobe=2, RightTemporalLobe=3, FrontalLobe=4, OccipitalLobe=5.

mcChannelIndexes_ts=find(chIndentity_ts(:,1)==1); %Separate the motor cortex channel indexes (including gradiometers and magnetometers).
mcMagChannelIndexes_ts=find(chIndentity_ts(:,1)==1 & chIndentity_ts(:,2)==1); %Separate the motor cortex channel indexes for magnetometers only.
mcGradChannelIndexes_ts=find(chIndentity_ts(:,1)==1 & (chIndentity_ts(:,2)==2 | chIndentity_ts(:,2)==3)); %Separate the motor cortex channel indexes for gradiometers only.

allchgrad_ts=find(chIndentity_ts(:,2)==2 | chIndentity_ts(:,2)==3);


%%%%%Replace the original values with zscores%%%%%%%%%
disp('zscore computation for raw data');
for trl=1:size(rmd_ts,3)
    rmd_ts(:,:,trl)=zscore(rmd_ts(:,:,trl)')';
end

%disp('extracting Motor cortex data');


miLabelIndexes_ts=find(labels_ts==binClasses(1) | labels_ts==binClasses(2));
labels_ts=labels_ts(miLabelIndexes_ts);

if(categ==1)
    rmdMC_ts=rmd_ts(mcChannelIndexes_ts,timeSpan(1)*fs+1:timeSpan(2)*fs,miLabelIndexes_ts);
elseif(categ==2)
    rmdMC_ts=rmd_ts(mcMagChannelIndexes_ts,timeSpan(1)*fs+1:timeSpan(2)*fs,miLabelIndexes_ts);
elseif(categ==3)
    rmdMC_ts=rmd_ts(mcGradChannelIndexes_ts,timeSpan(1)*fs+1:timeSpan(2)*fs,miLabelIndexes_ts);
else
    rmdMC_ts=rmd_ts(allchgrad_ts,timeSpan(1)*fs+1:timeSpan(2)*fs,miLabelIndexes_ts);
end

clear rmd_ts

disp('Doing temporal filtering');
band= fBank(1,:);
f_rmd_ts_1=temporalFiltering(rmdMC_ts,band,fs);
band= fBank(2,:);
f_rmd_ts_2=temporalFiltering(rmdMC_ts,band,fs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Preparing test data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear allChann_ts chIndentity_ts mcChannelIndexes_ts mcMagChannelIndexes_ts mcGradChannelIndexes_ts allchgrad_ts

%%%%Feature Extraction%%%%%%%
 for trl=1:size(f_rmd_ts_1,3)

    temp=f_rmd_ts_1(:,:,trl);
    
    Data_CSP_1=W_CSP_1*temp;
    Data_CSP_1=Data_CSP_1';
    Feat1=log(var(Data_CSP_1(:,:),1)./sum(var(Data_CSP_1(:,:),1))); 
    
    temp=f_rmd_ts_2(:,:,trl);
    Data_CSP_2=W_CSP_2*temp;
    Data_CSP_2=Data_CSP_2';
    Feat2=log(var(Data_CSP_2(:,:),1)./sum(var(Data_CSP_2(:,:),1))); 
    
    Test_X(trl,:)=[Feat1(1) Feat1(end) Feat2(1) Feat2(end)];

 end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
  
 
  Test_Y=labels_ts';
 
  
  TR_MDL=fitcsvm(Test_X,Test_Y,'KernelFunction','linear');
  Group = predict(TR_MDL,Test_X);
  acc(subID)=length(find(Group==Test_Y))/length(Test_Y);
  
% save(['TrainingAnalysisReport_Sub' num2str(subID) '_binClasses_' num2str(binClasses(1)) num2str(binClasses(2)) '.mat'],'TR_MDL','fBank','timeSpan','categ','binClasses', 'W_CSP_1', 'W_CSP_2', 'commChannIds', 'commChannIndexes_ts');

clearvars -except acc subID timeSpan fBank categ binClasses totSub sessID
subID
toc
end
%%

% for subID=1:totSub
%     
%     accTemp=squeeze(acc(subID,:,:));
%     accCat=mean(mean(accTemp,2));
%     meanAccGrid(subID,:)=[accCat*100];
%     
% end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%















