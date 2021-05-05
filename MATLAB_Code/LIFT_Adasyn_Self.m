% add the X+Y(:,L) in smote and select no of sample for each instance in
% minority label
clear
tic
addpath(genpath('.'));
datasets = [{'image'};{'datasets/emotions'};];
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
%% LIFT PARAMETER
ratio=0.1;
% Set the kernel type used by Libsvm
svm.type='Linear';
svm.para=[];
%%
no_fold = 5;
dataX = [X;Xt];
dataY = [Y;Yt];
    [IR_label, meanir]=Imbalance_ratio(dataY);
    [~,numL]=size(dataY);
    minorityL = IR_label>meanir;
    Y_new=[];
Yt_new     =[];
for c=1:numL
    if minorityL(c) && sum(dataY(:,c),1)>2
        Y_new   =   [Y_new,Y(:,c)];
        Yt_new  =   [Yt_new,Yt(:,c)];
    end
end
Y=Y_new;
Yt =Yt_new;

clear Yt_new Y_new ;
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
       %% LIFT
        [Outputs,Pre_Labels,Models,P_Centers,N_Centers]=LIFT_try(X,Y',Xt,Yt',ratio,svm);
        Yt_PredL_LIFT_SELF      = Pre_Labels;
        Yt_PredO_LIFT_SELF    = Outputs;
        ResultLIFT_Self_ADASYN{runNo}(:,selfIterNo) = EvaluationAll(Yt_PredL_LIFT_SELF,Yt',Yt_PredO_LIFT_SELF);
        
        %% %% ADASYN IN SELF
        Xnew = [];
        numD      =  size(X,2);
        adasyn_beta                     = 1;
        adasyn_kDensity                 = 5;
        adasyn_kSMOTE                   = 5;
        adasyn_featuresAreNormalized    = true;
        %z=ones(size(X_train_ADASYN,1),size(minorityL,2));
        for L=1:size(Y,2)
           %if sum(Y(:,L),1)>2
          
                adasyn_features                 =[X Y(:,L)] ;
                adasyn_labels                   = Y(:,L);
                [adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);
               % adasyn_featuresSyn=adasyn_featuresSyn(:,1:numD);
                if (size(adasyn_featuresSyn,1)~=0)
                    Xnew          = [adasyn_featuresSyn(:,1:numD);Xnew];
                    %Y_train_ADASYN        = [adasyn_labelsSyn;Y_train_ADASYN];
                else
                    % Y_train_ADASYN=Y_train_ADASYN;
                    Xnew=X;
                end
%                  Ynew = (adasyn_featuresSyn*Z)>=0.5;
%                 newIDXToRetain =  sum(Ynew(L,:).* ones(size(adasyn_featuresSyn,1),1),2) >0;
%                 T_new_sample_min_present  =   sum(newIDXToRetain);
                %Xnew = [adasyn_featuresSyn;Xnew];
          
            %end
        end
        %%
        Xnew = unique(Xnew,'rows');
        Y_new=zeros(size(Xnew,1),size(Y,2));
        %% predict lift output
        [~,Y_new]=lift_train(Models,P_Centers,N_Centers,Xnew,Y_new');
        Y_new=Y_new';
       
        %%
                newIDXToRetain =  sum(Y_new.* repmat(ones(1,size(Y_new,2)),size(Xnew,1),1),2) >0;
              %  newIDXToRetain =  sum(ynew.* repmat(ones(1,size(ynew,2)),size(xnew,1),1),2) >0;
        
                T_new_sample_min_present  =   sum(newIDXToRetain);
                X = [X;Xnew(newIDXToRetain,:)];
                Y = [Y;Y_new(newIDXToRetain,:)];
%         X=[X;Xnew];
%         Y=[Y;Ynew];
    end
    fprintf(" Run completed %d \n",runNo);
end


for runNo=1:ttlFold
    Avg_LLSF_Self_Smote = Avg_LLSF_Self_Smote + ResultLIFT_Self_ADASYN{runNo};
end

Avg_LLSF_Self_Smote  = Avg_LLSF_Self_Smote ./ ttlFold;
toc
