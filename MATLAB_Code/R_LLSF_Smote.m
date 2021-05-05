clear
tic
addpath(genpath('.'));
datasets = [{'datasets/image'};{'datasets/rcv1-sample1'}; ];
datasetNo =1;

filename1 = datasets{datasetNo};
load(filename1);
ttlFold = 5;
ttlEva         = 15;
maxIter        = 100;

optmParameter                   = struct;
optmParameter.maxIter           = maxIter;
optmParameter.minimumLossMargin = 0.001;
optmParameter.bQuiet            = 1;
optmParameter.alpha             = 0.1;
optmParameter.beta              = 0.1;
optmParameter.gamma             = 0.1;
%%
Result_LLSF     = zeros(ttlEva,ttlFold);
Result_LLSF_MLSMOTE = zeros(ttlEva,ttlFold);

%%
no_fold = 5;
dataX = [X;Xt];
dataY = [Y;Yt];

N          = size(dataY,1);
rand_idx   = randperm(N);
partationData    = kfoldpartation( dataX, dataY, no_fold,rand_idx);
for runNo=1:ttlFold
    X   =   full(partationData{runNo}.X);
    Y   =   full(partationData{runNo}.Y);
    Xt  =   full(partationData{runNo}.Xt);
    Yt  =   full(partationData{runNo}.Yt);
    dataY = [Y;Yt];
    [IR_label, meanir]=Imbalance_ratio(Y);
    [~,numL]=size(Y);
    minorityL = IR_label>meanir;
   
   Yt_PredL_LLSF           = zeros(size(Yt));
    Yt_PredO_LLSF           = zeros(size(Yt));
    Yt_PredL_LLSF_MLSMOTE   = zeros(size(Yt));
    Yt_PredO_LLSF_MLSMOTE   = zeros(size(Yt));
   
    for L=1:size(Y,2)
        train_data      = X;
        train_target    = Y(:,L);
        test_target     = Yt(:,L);
        %% LLSF
        Z = LLSF( train_data, train_target, optmParameter);
        Yt_PredO_LLSF(:,L) = (Xt*Z)';
        Yt_PredL_LLSF(:,L)=Yt_PredO_LLSF(:,L)>=0.5;
        
        %% MLSMOTE
        if minorityL(L)
        [train_data1, train_target1] = SMOTE_FUN(train_data,Y(:,L));
        train_data=[train_data;train_data1];
        train_target=[train_target;train_target1];
        
        Z = LLSF( train_data, train_target, optmParameter);
        end
        Yt_PredO_LLSF_MLSMOTE(:,L) = (Xt*Z)';
        Yt_PredL_LLSF_MLSMOTE(:,L)=Yt_PredO_LLSF_MLSMOTE(:,L)>=0.5;
        
        
    end
     %% Result
    Result_LLSF(:,runNo) = EvaluationAll(Yt_PredL_LLSF',Yt',Yt_PredO_LLSF');
    Result_LLSF_MLSMOTE(:,runNo)     = EvaluationAll(Yt_PredL_LLSF_MLSMOTE',Yt',Yt_PredO_LLSF_MLSMOTE');
    fprintf("no of run completed %d\n", runNo);
end
Avg_Result_LLSF              = mean(Result_LLSF,2);
    Avg_Result_LLSF_MLSMOTE      = mean(Result_LLSF_MLSMOTE,2);
    a=[Avg_Result_LLSF Avg_Result_LLSF_MLSMOTE];
toc
%end
