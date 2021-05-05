% add the X+Y(:,L) in smote and select no of sample for each instance in
% minority label
clear
tic
addpath(genpath('.'));
datasets = [{'emotions'};{'datasets/genbase'};];
datasetNo =1;
filename1 = datasets{datasetNo};
load(filename1);
ttlFold = 5;
%% Test Result
maxSelfIter    = 5;
ttlEva         = 15;
maxIter        = 100;
Avg_LLSF_Self_Smote   = zeros(ttlEva,maxSelfIter);
Result_LLSF_SMOTE{ttlFold} = zeros(ttlEva,maxSelfIter);
optmParameter                   = struct;
optmParameter.maxIter           = maxIter;
optmParameter.minimumLossMargin = 0.0001;
optmParameter.bQuiet            = 1;
optmParameter.alpha             = 0.1;
optmParameter.beta              = 0.1;
optmParameter.gamma             = 0.1;
%%
no_fold = 5;
dataX = [X;Xt];
dataY = [Y;Yt];
N          = size(dataY,1);
rand_idx   = randperm(N);
partationData    = kfoldpartation( dataX, dataY, no_fold,rand_idx);

%% SELF LEARN
for runNo=1:ttlFold
    X   =   full(partationData{runNo}.X);
    Y   =   full(partationData{runNo}.Y);
    Xt  =   full(partationData{runNo}.Xt);
    Yt  =   full(partationData{runNo}.Yt);
   for selfIterNo = 1: maxSelfIter
        [IR_label, meanir]=Imbalance_ratio(Y);
    [~,numL]=size(Y);
    minorityL = IR_label>meanir;
        %% LLSF 
        Z = LLSF( X, Y, optmParameter);
        prdtnYtLLSF = (Xt*Z)';
        Result_LLSF_SMOTE{runNo}(:,selfIterNo) = EvaluationAll(prdtnYtLLSF>=0.5,Yt',prdtnYtLLSF);
        
       
        
        %% %% ADASYN IN SELF
        Xnew = [];
        numD      =  size(X,2);
        adasyn_beta                     = 1;
        adasyn_kDensity                 = 5;
        adasyn_kSMOTE                   = 5;
        adasyn_featuresAreNormalized    = true;
        for L=1:size(Y,2)
          if minorityL(L)          
                adasyn_features                 =[X Y(:,L)] ;
                adasyn_labels                   = Y(:,L);
                [adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);
               % adasyn_featuresSyn=adasyn_featuresSyn(:,1:numD);
                if (size(adasyn_featuresSyn,1)~=0)
                    Xnew          = [adasyn_featuresSyn(:,1:numD);Xnew];
                else
                    Xnew=X;
                end
          end       
        end
        %%
        Xnew = unique(Xnew,'rows');
        Ynew  = (Xnew*Z)>=0.5;
        %%
        
        newIDXToRetain =  sum(Ynew.* repmat(minorityL,size(Xnew,1),1),2) >0;
                %newIDXToRetain =  sum(Ynew.* repmat(ones(1,size(Ynew,2)),size(Xnew,1),1),2) >0;
                T_new_sample_min_present  =   sum(newIDXToRetain);
                X = [X;Xnew(newIDXToRetain,:)];
                Y = [Y;Ynew(newIDXToRetain,:)];

    end
    fprintf(" Run completed %d \n",runNo);
end


for runNo=1:ttlFold
    Avg_LLSF_Self_Smote = Avg_LLSF_Self_Smote + Result_LLSF_SMOTE{runNo};
end

Avg_LLSF_Self_Smote  = Avg_LLSF_Self_Smote ./ ttlFold;
toc
