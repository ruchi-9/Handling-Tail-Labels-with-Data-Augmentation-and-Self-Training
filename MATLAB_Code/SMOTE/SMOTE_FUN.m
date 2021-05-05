function [train_data, train_target]=SMOTE_FUN(X,Y)
%% 
% train_data    =   [train_data; synthetic data];  
% train_target  =   [train_target; synthetic label];
% train target = N*1;  only for single labels;
%%
Data=[X,Y];
k=5;

% calculate at which row this label is 1 and take all the instance
% from X.
[r,~]=find(Y);
minBag=Data(r,:);  % get all instances of label line 7

for j=1:size(minBag,1)   %for each sample in minBag line 8
    distance= pdist2(minBag(j,:),minBag);  % calculate distance
    [distance_sort,Idx]=sort(distance,'ascend');  % sort Smaller To Largest Distance line 10
    %Neighbour set selection
    %neighbour=distance_sort(Idx(:,2:k+1));
    if size(Idx,2)>=k+1
        neighbour_IDX=Idx(:,2:k+1);
        % neighbour=full(minBag(Idx(:,2:k+1),:));
        neighbour=full(minBag(neighbour_IDX,:));
        refNeigh=neighbour(randi(length(neighbour_IDX)),:);
    else
        neighbour_IDX=Idx;
        neighbour=full(minBag(Idx,:));
        refNeigh=neighbour(randi(length(neighbour_IDX)),:);
    end
    % refNeigh=neighbour(randperm(size(neighbour,1),1),:); %randomly select neighbour
    %feature set and Label set selection
    
    synthSmpl_feature=zeros(1,size(X,2));
    %feature set assignment
    diff=refNeigh(1,1:size(X,2))-minBag(j,1:size(X,2));
    offset=diff*rand(1);
    value=minBag(j,1:size(X,2))+offset;
    synthSmpl_feature(1,1:size(X,2))=value;
    %label set assignment
    
    synthSmpl_label=ones(size(synthSmpl_feature,1),1);
    synthSmpl=[synthSmpl_feature,synthSmpl_label];
    Data=[Data;synthSmpl];
end
train_data=Data(:,1:size(X,2));
train_target=Data(:,(size(X,2)+1));
end