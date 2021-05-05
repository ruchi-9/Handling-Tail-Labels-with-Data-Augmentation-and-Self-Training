# Handling Tail Labels in Multi-Label Classification with Data Augmentation and Self Training 
Tail labels are one of the challenging aspect of Multi-label classification, which forces
the learning algorithms to learn from very few instances, thus crippling their prediction
accuracy. At the core, We have used oversampling and synthetic data generation
to conduct data augmentation for obtaining better results over base classifiers in use.
Overall, we experimented with three approaches – SMOTE-based, ADASYN based,
and Self-training based classification (a newly evolved semi-supervised machine learning
algorithm), all of which were found to be successful with different data-sets, setting
up a new record to beat for researchers.

# Contributions
1. I have study the state-of-the-art methods for imbalanced data set classification.
2. To overcome the tail label problem in multi-label classification, we have done
data augmentation using the concept of oversampling, instead of cloning the
instances.
3. I have tried synthetic data generation using well-known techniques - SMOTE
and ADASYN.
4. I have used the concept of self-training in multi-label classification for predicting
the label matrix for a instance.
5. I have also used different evaluation metrics to find out the better result among
them in data sets.
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Tail%20labels.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Datasets.PNG)

![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Algo%201.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Algo%202.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/algo%203.PNG)
# Experimental Analysis
In my experiment, i have took five-fold cross-validation approach for separating the training
and testing data. The data set divide in the ratio of 80:20 for training and testing
respectively along with analysis on a different value of K and Number of the sample
generated for each instance, finally concluding better result on selecting K as 5 in
SMOTE and self-iteration vary for each data set in Self training.
For finding out whether the approach helps in improving tail labels classification or
not, i have separated the tail labels from the data set and experiment on them.


![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%201.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%202.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%203.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%204.PNG)
![text](https://github.com/ruchi-9/Handling-Tail-labels-with-self-supervised-and-other-methods/blob/master/Screenshots/Result%205.PNG)
