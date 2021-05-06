# Handling Tail Labels in Multi-Label Classification with Data Augmentation and Self Training 
Tail labels are one of the challenging aspect of Multi-label classification, which forces
the learning algorithms to learn from very few instances, thus crippling their prediction
accuracy. At the core, We have used oversampling and synthetic data generation
to conduct data augmentation for obtaining better results over base classifiers in use.
Overall, we experimented with three approaches – SMOTE-based, ADASYN based,
and Self-training based classification (a newly evolved semi-supervised machine learning
algorithm), all of which were found to be successful with different data-sets, setting
up a new record to beat for researchers.

## Contributions
- I carried out a thorough literature survey of the state-of-the-art methods for imbalanced data set classification.
- To overcome the tail label problem in multi-label classification, we have done
data augmentation using the concept of oversampling, instead of cloning the
instances.
- I have tried synthetic data generation using well-known techniques - SMOTE
and ADASYN.
- I have used the concept of self-training in multi-label classification for predicting
the label matrix for a instance.
- I have also used different evaluation metrics to find out the best result among
them in data science. 
#
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Tail%20labels.PNG)
#
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Datasets.PNG)
#
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Algo%201.PNG)
#
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Algo%202.PNG)
#
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/algo%203.PNG)
## Experimental Analysis
In my experiment, I have taken five-fold cross-validation approach for separating the training
and testing data. The data set was divided in the ratio of 80:20 for training and testing
respectively. Analysis of different values of K and Number of the sample
generated for each instance, shows that better result on selecting K as 5 in
SMOTE and self-iteration vary for each data set in Self training.
For finding out whether the approach helps in improving tail labels classification or
not, I have separated the tail labels from the data set and experiment on them.


![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%201.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%202.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%203.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%204.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%205.PNG)
