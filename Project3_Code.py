# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:08:45 2020

@author: aaron
"""
# import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the data set
from sklearn import datasets

# 1.) import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :4]  # we only take the first two features.
y = iris.target

colors = ['r', 'b', 'y']
flower_feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
flower_target_names = ['Setosa', 'Versicolour', 'Virginica']

#Visualize the whole dataset
x_index = 2
y_index = 3
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()


# 2.) Using a Decision Tree to identify the two most important features

# training/validation/test split 
# test size is 20%; 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)

print('\n')
print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))


# decision tree cross-validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

d_array = np.arange(1, 7, 1)  #max_depth
best_score = 0.0

for tdepth in d_array:
    clf = DecisionTreeClassifier(max_depth = tdepth) 
    valid_score = cross_val_score(clf, X_train, y_train, cv=10)
    if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_d = tdepth

print("The best max-depth for decision trees is {:2d}".format(best_d))

# retrain the best model
clf = DecisionTreeClassifier(max_depth = best_d)
clf.fit(X_train, y_train)    # training
print('\n')
print('Accuracy of Decision tree classifier on the training set: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('Accuracy of Decision tree classifier on the test set: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))
y_pred = clf.predict(X_test)


# multi-class classification confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

import seaborn as sns
plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('Decision Tree test score:{0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()

# multi-class classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=flower_target_names))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))

#Plotting the decision tree
from sklearn.tree import plot_tree
plt.figure(figsize = (10,8))
plot_tree(clf, feature_names = flower_feature_names, class_names = flower_target_names, filled = True)

# feature importance
importances = clf.feature_importances_
fnames = np.array(flower_feature_names)
indices = np.argsort(importances)

plt.figure()
plt.title('Feature importances (max_depth={}) \n {}'.format(best_d, importances), size=15)
plt.barh(range(X.shape[1]), importances[indices], color="b")
plt.yticks(range(X.shape[1]), fnames[indices])
plt.ylim([-1, X.shape[1]])
plt.xlabel('Importance', size=15)
plt.ylabel('Features', size=15)
plt.show()

#Petal Length and Petal Width are the two most important features

#Setting up the classification again
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # getting only petal length and width as features
y = iris.target

colors = ['r', 'b', 'y']
flower_feature_names = ['Petal Length', 'Petal Width']
flower_target_names = ['Setosa', 'Versicolour', 'Virginica']

# 3.) Using LSVC to solve this multiclass classification

# training/validation/test split 
# test size is 20%; 
from sklearn.model_selection import train_test_split
X_train_, X_test_, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)

print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))

# Standardization and plot the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_)
X_test = scaler.transform(X_test_)

# LSVC classifier training and 5-fold cross-validation
C_array = np.arange(0.1, 5.1, 0.1)
best_score = 0.0

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
for C_lsvc in C_array:
    clf = LinearSVC(C=C_lsvc, max_iter=1000000) # initialization and configuration
    valid_score = cross_val_score(clf, X_train, y_train, cv=10)
    if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_C = C_lsvc

print("The best C is ", best_C)

# retrain the best model
clf = LinearSVC(C=best_C, max_iter=1000000)
clf.fit(X_train, y_train)    # training
print('\n')
print('The training score: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('The test score: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))

print("\n")


# decision regions: a contour plot
x1_min, x1_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
x2_min, x2_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

h = .02  
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', c=colors[yy-1], 
                label=flower_target_names[yy-1])
plt.title('LSVC decision regions and the test set', size=15)
plt.xlabel('flower feature Petal Length (x1)', size=15)
plt.ylabel('flower feature Petal Width (x2)', size=15)
plt.legend(fontsize='large')
plt.show()
print("\n")

# multi-class classification confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

import seaborn as sns
plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('LSVC Accuracy:{0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()

# multi-class classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=flower_target_names))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))

#Setting up the classification again
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # getting only petal length and width as features
y = iris.target

colors = ['r', 'b', 'y']
flower_feature_names = ['Petal Length', 'Petal Width']
flower_target_names = ['Setosa', 'Versicolour', 'Virginica']

# 3.) Using rbf SVC to solve this multiclass classification
# training/validation/test split 
# test size is 20%; 
from sklearn.model_selection import train_test_split
X_train_, X_test_, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)

print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))

# Standardization and plot the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_)
X_test = scaler.transform(X_test_)

# SVC with rbf kerenel classifier training and cross-validation
from sklearn.svm import SVC
best_score = 0.0
gamma_array = np.arange(0.1, 5.1, 0.1)
C_array = np.arange(0.1, 5.1, 0.1)

from sklearn.model_selection import cross_val_score
for gamma_svc in gamma_array:
    for C_svc in C_array:
        clf = SVC(kernel='rbf', random_state=0, gamma=gamma_svc, C=C_svc)
        valid_score = cross_val_score(clf, X_train, y_train, cv=10)
        if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_parameters = {'C':C_svc, 'gamma':gamma_svc}

print('best score: {:.3f}'.format(best_score))
print('best parameters: {}'.format(best_parameters))

# retrain the best model
clf = SVC(kernel='rbf', random_state=0, **best_parameters)
clf.fit(X_train, y_train)

print('Accuracy of rbf SVC classifier on the training set: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('Accuracy of rbf SVC classifier on the test set: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))
print('\n')

# decision regions: a contour plot
x1_min, x1_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
x2_min, x2_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5

plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

h = .02  
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', c=colors[yy-1], 
                label=flower_target_names[yy-1])
plt.title('rbf SVC decision regions and the test set', size=15)
plt.xlabel('flower feature Petal Length (x1)', size=15)
plt.ylabel('flower feature Petal Width (x2)', size=15)
plt.legend(fontsize='large')
plt.show()
print("\n")

# multi-class classification confusion matrix
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

import seaborn as sns
plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('rbf SVC Accuracy:{0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()

# multi-class classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=flower_target_names))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))

#Setting up the classification again
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # getting only petal length and width as features
y = iris.target

colors = ['r', 'b', 'y']
flower_feature_names = ['Petal Length', 'Petal Width']
flower_target_names = ['Setosa', 'Versicolour', 'Virginica']

# 3.) Using a Decision Tree to solve this multiclass classification

# training/validation/test split 
# test size is 20%; 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)

print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))

# decision tree cross-validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

d_array = np.arange(1, 7, 1)  #max_depth
best_score = 0.0

for tdepth in d_array:
    clf = DecisionTreeClassifier(max_depth = tdepth) 
    valid_score = cross_val_score(clf, X_train, y_train, cv=10)
    if valid_score.mean() > best_score:
            best_score = valid_score.mean()
            best_d = tdepth

print("The best max-depth for decision trees is {:2d}".format(best_d))

# retrain the best model
clf = DecisionTreeClassifier(max_depth = best_d)
clf.fit(X_train, y_train)    # training
print('\n')
print('Accuracy of Decision tree classifier on the training set: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('Accuracy of Decision tree classifier on the test set: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))
y_pred = clf.predict(X_test)

# decision regions: a contour plot
x1_min, x1_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
x2_min, x2_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5

plt.figure()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

h = .02  
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', c=colors[yy-1], 
                label=flower_target_names[yy-1])
plt.title('Decision Trees decision regions and the test set', size=15)
plt.xlabel('flower feature Petal Length (x1)', size=15)
plt.ylabel('flower feature Petal Width (x2)', size=15)
plt.legend(fontsize='large')
plt.show()
print("\n")

#Plotting the decision tree
from sklearn.tree import plot_tree
plt.figure(figsize = (10,8))
plot_tree(clf, feature_names = flower_feature_names, class_names = flower_target_names, filled = True)

# multi-class classification confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

import seaborn as sns
plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('Decision Tree test score:{0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()

# multi-class classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=flower_target_names))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))

#Setting up the classification again
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # getting only petal length and width as features
y = iris.target

colors = ['r', 'b', 'y']
flower_feature_names = ['Petal Length', 'Petal Width']
flower_target_names = ['Setosa', 'Versicolour', 'Virginica']

# 5.) Using a Enesemble to solve this multiclass classification
# training/validation/test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 4)

print('There are {:3d} samples in the training set.'.format(len(y_train)))
print('There are {:3d} samples in the test set.'.format(len(y_test)))

# base classifiers
from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier(max_depth = 4)
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators = 6, random_state = 2)
from sklearn.naive_bayes import GaussianNB
clf3 = GaussianNB()   

# voting classifier
from sklearn.ensemble import VotingClassifier
clf = VotingClassifier(estimators=[
        ('DT',clf1), ('kNN',clf2), ('gNB',clf3)], voting='soft')
clf = clf.fit(X_train, y_train)

print('\n')
print('Accuracy of voting classifier on training set: {:.2f}%'
     .format(clf.score(X_train, y_train)*100))
print('Accuracy of voting classifier on test set: {:.2f}%'
     .format(clf.score(X_test, y_test)*100))

# decision regions
plt.figure()
x1_min, x1_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
x2_min, x2_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
# plot the results as a contour plot
h = .02  # spacing between grid points
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for yy in range(1, 4):
    plt.scatter(X_test[y_test==yy, 0], X_test[y_test==yy, 1], marker='o', c=colors[yy-1], 
                label=flower_target_names[yy-1])
y_pred = clf.predict(X_test)
plt.title('Ensemble decision regions and the test set', size=15)
plt.xlabel('flower feature Petal Length (x1)', size=15)
plt.ylabel('flower feature Petal Width (x2)', size=15)
plt.legend(fontsize='large')
plt.show()
print("\n")

# multi-class classification confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mc = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confusion_mc, 
                     index = [i for i in range(1,4)], columns = [i for i in range(1,4)])

import seaborn as sns
plt.figure(figsize=(7.5,5))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Paired)
plt.title('Ensemble test score:{0:.2f}%'.format(clf.score(X_test, y_test)*100), size=15)
plt.ylabel('True label', size=15)
plt.xlabel('Predicted label', size=15)
plt.show()

# multi-class classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=flower_target_names))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Micro-averaged precision = {:.2f} (treat instances equally)'
      .format(precision_score(y_test, y_pred, average = 'micro')))
print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_pred, average = 'macro')))