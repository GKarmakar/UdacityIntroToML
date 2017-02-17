"""
Faces recognization example using eigenfaces and SVMs

The dataset used in this example is a preprocessed excerpt of the "Labeled faces in the Wild:, aks LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw=funneled.tgz (233MB)
  ...-LFW: http://vis-www.cs.umass.edu/lfw/

"""

print _doc_
from time import time
import logging
import pylab as pl

from sklearn_cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

##Split into a training set and using a stratified k fold

# split into a training a test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

# Compute a PCA(eigenfaces) on the face dataset (treated as unlabeled dataset)
# unsupervised feature extraction/dimensioanlity reduction 

n_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0)

eigenfaces = pcs.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")

t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pcs.transform(X_test)
print("Done in %0.3fs" %(time() - t0)

# Train a SVM classification model

print("Fitting the classifier to the training set")

t0 = time()
param_grid = {
            'C': [1e3, 5e3, 1e4, 5e4, 1e5]
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
             }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" %(time() - t0)
print("Best estimator found by grid serach:")
print(clf.best_estimator_)

## Quantitative evaluation of the model quality on the test set

print("Predicting the people names on the testing set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
print("classification_report(y_test, y_pred, target_names=target_names)
print("confusion_matrix(y_test, y_pred, labesl=range(n_classes))







