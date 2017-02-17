#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (SVM) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
#sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

##features_train = features_train[:len(features_train)/100] 
##labels_train = labels_train[:len(labels_train)/100] 



#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="rbf")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
#########################################################
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print(accuracy)
