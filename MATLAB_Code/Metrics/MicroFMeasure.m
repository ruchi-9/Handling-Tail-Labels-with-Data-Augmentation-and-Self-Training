
function MicroF1Measure=MicroFMeasure(test_targets,predict_targets)
% syntax
%   MicroF1Measure=MicroFMeasure(test_targets,predict_targets)
%
% Input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   predict_targets     - L x num_test data matrix of predicted labels
%
% Output
%   MicroF1Measure
%{
    test_targets=double(test_targets==1);
    predict_targets=double(predict_targets==1);
    [L,num_test]=size(test_targets);
    groundtruth=reshape(test_targets,1,L*num_test);
    predict=reshape(predict_targets,1,L*num_test);
    intersection=groundtruth*predict';
    precision = intersection/sum(predict);
    recall = intersection/sum(groundtruth);
    MicroF1Measure=2*precision*recall/(precision+recall);
    
end
%}
Y=double(test_targets==1);
X=double(predict_targets==1);

XandY = X&Y;
if sum(XandY(:))==0
    %Precision=0;
    %Recall=0;
    MicroF1Measure=0;
else if  sum(X(:))==0
        %Precision=0;
        %Recall=0;
        MicroF1Measure=0;
    else if sum(Y(:))==0
            %Precision=0;
            %Recall=0;
            MicroF1Measure=0;
        else
            Precision=sum(XandY(:))/sum(X(:));
            
            Recall=sum(XandY(:))/sum(Y(:));
            MicroF1Measure=2*Precision*Recall/(Precision+Recall);
        end
    end
end
end