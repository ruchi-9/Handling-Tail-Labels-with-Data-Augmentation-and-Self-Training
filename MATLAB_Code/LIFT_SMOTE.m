clear
tic
addpath(genpath('.'));
datasets = [{'datasets/ohsumed'};{'datasets/genbase'};];
datasetNo =1;
%birds 7 label image scene 3 
filename1 = datasets{datasetNo};
load(filename1);
ttlFold = 5;
ttlEva         = 15;
%%
Result_LIFT     = zeros(ttlEva,ttlFold);
Result_LIFT_MLSMOTE = zeros(ttlEva,ttlFold);

%% LIFT PARAMETER
ratio=0.1;
% Set the kernel type used by Libsvm
svm.type='Linear';
svm.para=[];
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
    [IR_label, meanir]=Imbalance_ratio(dataY);
    [~,numL]=size(dataY);
    minorityL = IR_label>meanir;
    Y_new=[];
    Yt_new     =[];
    for c=1:numL
        if minorityL(c)
            Y_new   =   [Y_new,Y(:,c)];
            Yt_new  =   [Yt_new,Yt(:,c)];
        end
    end
    Y=Y_new;
    Yt =Yt_new;
    clear Yt_new Y_new ;
    
    Yt_PredL_LIFT            = zeros(size(Yt));
    Yt_PredO_LIFT            = zeros(size(Yt));
    Yt_PredL_LIFT_MLSMOTE    = zeros(size(Yt));
    Yt_PredO_LIFT_MLSMOTE    = zeros(size(Yt));
    
    % Xnew  = [];
    for L=1:size(Y,2)
        train_data      = X;
        train_target    = Y(:,L);
        test_target     = Yt(:,L);
        %% LIFT
        [Outputs,Pre_Labels]  =  LIFT(train_data,train_target',Xt,test_target',ratio,svm);
        
        Yt_PredL_LIFT(:,L)      = Pre_Labels';
        Yt_PredO_LIFT(:,L)      = Outputs';
        
        %% MLSMOTE
        [train_data1, train_target1] = SMOTE_FUN(train_data,Y(:,L));
        train_data=[train_data;train_data1];
        train_target=[train_target;train_target1];
        
        [Outputs,Pre_Labels]  =  LIFT(train_data,train_target',Xt,test_target',ratio,svm);
        Yt_PredL_LIFT_MLSMOTE(:,L)      = Pre_Labels';
        Yt_PredO_LIFT_MLSMOTE(:,L)      = Outputs';
        
    end
    %% Result
     Result_LIFT(:,runNo)     = EvaluationAll(Yt_PredL_LIFT',Yt',Yt_PredO_LIFT');
    Result_LIFT_MLSMOTE(:,runNo) = EvaluationAll(Yt_PredL_LIFT_MLSMOTE',Yt',Yt_PredO_LIFT_MLSMOTE');
    fprintf("no of run completed %d\n", runNo);
end
Avg_Result_LIFT              = mean(Result_LIFT,2);
Avg_Result_LIFT_MLSMOTE      = mean(Result_LIFT_MLSMOTE,2);
a=[Avg_Result_LIFT Avg_Result_LIFT_MLSMOTE Avg_Result_LIFT_MLSMOTE-Avg_Result_LIFT];
toc