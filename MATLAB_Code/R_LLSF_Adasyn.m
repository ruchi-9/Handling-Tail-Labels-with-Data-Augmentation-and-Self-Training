clear;
tic
addpath(genpath('.'));
rng('shuffle');
datasets={'image.mat';};%'emotions.mat';};
ttFold = 5;
ResultAll      = zeros(15,ttFold);
ResultAll_Ada   = zeros(15,ttFold);
maxIter        = 100;
optmParameter                   = struct;
optmParameter.maxIter           = maxIter;
optmParameter.minimumLossMargin = 0.0001;
optmParameter.bQuiet            = 1;
optmParameter.alpha             = 0.1;
optmParameter.beta              = 0.1;
optmParameter.gamma             = 0.1;
%%
load(datasets{1});
no_fold=5;

dataX=[X;Xt];
dataY=[Y;Yt];

N          = size(dataY,1);
rand_idx   =randperm(N);

partationData    = kfoldpartation( dataX, dataY, no_fold,rand_idx);

%for i=1:2
for foldNo=1:no_fold
    % [X,Xt,Y,Yt]=DATA2(batchsize,foldNo,N,dataX,dataY,rand_idx,fulldata);
    X   =   full(partationData{foldNo}.X);
    Y   =   full(partationData{foldNo}.Y);
    Xt  =   full(partationData{foldNo}.Xt);
    Yt  =   full(partationData{foldNo}.Yt);
    dataY = [Y;Yt];
    [~,numL]  =  size(Y); %  L=numLabels Labels in dataset
    numD      =  size(X,2);
    dataY     =  [Y;Yt];
    dataY(dataY==-1) = 0;
    
    [IR_label, meanir]  =  Imbalance_ratio(dataY);
    minorityL = IR_label>meanir;
    total_minlabels  =  sum(minorityL);
        
    Yt_PredO_LLSF= zeros(size(Yt));
    Yt_PredL_LLSF= zeros(size(Yt));
    Yt_PredO_LLSF_Ada= zeros(size(Yt));
    Yt_PredL_LLSF_Ada= zeros(size(Yt));
    
    for L = 1:numL
        
        % fprintf(fid,'no of data genertaed by label in round %d for label %d\n',foldNo,L);
        
        train_data      = X;
        train_target_CV = Y(:,L);
        train_target    = Y(:,L);
        test_target     = Yt(:,L);
        Z = LLSF( train_data, train_target, optmParameter);
        Yt_PredO_LLSF(:,L) = (Xt*Z)';
        Yt_PredL_LLSF(:,L)=Yt_PredO_LLSF(:,L)>=0.5;
        
        
        %%
        if minorityL(L)
            % ADASYN(in_features, in_labels, in_beta, in_kDensity, in_kSMOTE, in_featuresAreNormalized)
            adasyn_features                 =[train_data train_target];
            adasyn_labels                   = train_target;
            adasyn_beta                     = 1;
            adasyn_kDensity                 = 5;
            adasyn_kSMOTE                   = 5;
            adasyn_featuresAreNormalized    = true;
            
            [adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);
            train_target        = [adasyn_labelsSyn;train_target];
            train_data          = [adasyn_featuresSyn(:,1:numD);train_data];
            
            Z = LLSF( train_data, train_target, optmParameter);
        end
        Yt_PredO_LLSF_Ada(:,L) = (Xt*Z)';
        Yt_PredL_LLSF_Ada(:,L)=Yt_PredO_LLSF_Ada(:,L)>=0.5;
        %%
        
    end
    %%
    ResultAll(:,foldNo)     = EvaluationAll(Yt_PredL_LLSF',Yt',Yt_PredO_LLSF');
    ResultAll_Ada(:,foldNo)     = EvaluationAll(Yt_PredL_LLSF_Ada',Yt',Yt_PredO_LLSF_Ada');
    fprintf(" Run completed %d \n",foldNo);
end
AvgResultAll     = mean(ResultAll,2);
AvgResultAll_Ada     = mean(ResultAll_Ada,2);
a=[AvgResultAll AvgResultAll_Ada];
toc