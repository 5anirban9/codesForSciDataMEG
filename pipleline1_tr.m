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

timeSpan=[2.5 5.5];
fBank=[8 12;14 30]; 
categ=4; %MC=1;MCmag=2;MCgrad=3;all=4;
binClasses=[2 4];
totSub=18;

for subID=1:totSub
tic

sessID=2;
disp('loading data....');
load(['E:\UU-JAN-All-Documents-AUG(2019)\MEG_BRI2017-19\TempData\ParsedMEGData_P' num2str(subID) '_S' num2str(sessID)]); %Load the MEG data
disp('data loaded.');
allChannId_ts = MEGdata.ci;
clear MEGdata

sessID=1;
disp('loading data....');
load(['E:\UU-JAN-All-Documents-AUG(2019)\MEG_BRI2017-19\TempData\ParsedMEGData_P' num2str(subID) '_S' num2str(sessID)]); %Load the MEG data
disp('data loaded.');

disp('extracting meg data');
rmd_tr=MEGdata.x;
allChannId_tr = MEGdata.ci;
allChann_tr = MEGdata.c; %Read all the channelcodes
labels_tr=MEGdata.y;
fs=MEGdata.s;

clear MEGdata 

[commChannIds, commChannIndexes_tr]=intersect(allChannId_tr,allChannId_ts);
[commChannIds, commChannIndexes_ts]=intersect(allChannId_ts,allChannId_tr); %need to store for future use.

rmd_tr=rmd_tr(:,commChannIndexes_tr,:);
disp('reorganizing dimensions');
rmd_tr = permute(rmd_tr,[2 1 3]);

allChann_tr=allChann_tr(commChannIndexes_tr);
chIndentity_tr = chIndentify(allChann_tr); %Map the channelcodes with cortex id: MotorCortex=1, LeftTemporalLobe=2, RightTemporalLobe=3, FrontalLobe=4, OccipitalLobe=5.

mcChannelIndexes_tr=find(chIndentity_tr(:,1)==1); %Separate the motor cortex channel indexes (including gradiometers and magnetometers).
mcMagChannelIndexes_tr=find(chIndentity_tr(:,1)==1 & chIndentity_tr(:,2)==1); %Separate the motor cortex channel indexes for magnetometers only.
mcGradChannelIndexes_tr=find(chIndentity_tr(:,1)==1 & (chIndentity_tr(:,2)==2 | chIndentity_tr(:,2)==3)); %Separate the motor cortex channel indexes for gradiometers only.

allchgrad_tr=find(chIndentity_tr(:,2)==2 | chIndentity_tr(:,2)==3);


%%%%%Replace the original values with zscores%%%%%%%%%
disp('zscore computation for raw data');
for trl=1:size(rmd_tr,3)
    rmd_tr(:,:,trl)=zscore(rmd_tr(:,:,trl)')';
end

%disp('extracting Motor cortex data');


miLabelIndexes_tr=find(labels_tr==binClasses(1) | labels_tr==binClasses(2));
labels_tr=labels_tr(miLabelIndexes_tr);

if(categ==1)
    rmdMC_tr=rmd_tr(mcChannelIndexes_tr,timeSpan(1)*fs+1:timeSpan(2)*fs,miLabelIndexes_tr);
elseif(categ==2)
    rmdMC_tr=rmd_tr(mcMagChannelIndexes_tr,timeSpan(1)*fs+1:timeSpan(2)*fs,miLabelIndexes_tr);
elseif(categ==3)
    rmdMC_tr=rmd_tr(mcGradChannelIndexes_tr,timeSpan(1)*fs+1:timeSpan(2)*fs,miLabelIndexes_tr);
else
    rmdMC_tr=rmd_tr(allchgrad_tr,timeSpan(1)*fs+1:timeSpan(2)*fs,miLabelIndexes_tr);
end

clear rmd_tr

disp('Doing temporal filtering');
band= fBank(1,:);
f_rmd_tr_1=temporalFiltering(rmdMC_tr,band,fs);
band= fBank(2,:);
f_rmd_tr_2=temporalFiltering(rmdMC_tr,band,fs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Preparing test data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear allChann_tr chIndentity_tr mcChannelIndexes_tr mcMagChannelIndexes_tr mcGradChannelIndexes_tr allchgrad_tr

disp('Doing spatial filtering');
[W_CSP_1] = spatialFiltering(f_rmd_tr_1,labels_tr,binClasses);
[W_CSP_2] = spatialFiltering(f_rmd_tr_2,labels_tr,binClasses);

%%%%Feature Extraction%%%%%%%
 for trl=1:size(f_rmd_tr_1,3)

    temp=f_rmd_tr_1(:,:,trl);
    
    Data_CSP_1=W_CSP_1*temp;
    Data_CSP_1=Data_CSP_1';
    Feat1=log(var(Data_CSP_1(:,:),1)./sum(var(Data_CSP_1(:,:),1))); 
    
    temp=f_rmd_tr_2(:,:,trl);
    Data_CSP_2=W_CSP_2*temp;
    Data_CSP_2=Data_CSP_2';
    Feat2=log(var(Data_CSP_2(:,:),1)./sum(var(Data_CSP_2(:,:),1))); 
    
    Train_X(trl,:)=[Feat1(1) Feat1(end) Feat2(1) Feat2(end)];

 end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
  
 
  Train_Y=labels_tr';
 
  
  TR_MDL=fitcsvm(Train_X,Train_Y,'KernelFunction','linear');
  Group = predict(TR_MDL,Train_X);
  acc(subID)=length(find(Group==Train_Y))/length(Train_Y);
  
save(['TrainingAnalysisReport_Sub' num2str(subID) '_binClasses_' num2str(binClasses(1)) num2str(binClasses(2)) '.mat'],'TR_MDL','fBank','timeSpan','categ','binClasses', 'W_CSP_1', 'W_CSP_2', 'commChannIds', 'commChannIndexes_ts');

clearvars -except acc subID timeSpan fBank categ binClasses totSub 
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















