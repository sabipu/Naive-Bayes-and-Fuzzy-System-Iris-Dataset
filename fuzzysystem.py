# -*- coding: utf-8 -*-
"""
@author: Bipu Bajgai & Sabharish Jayachandran
"""

# Author: Bipu Bajgai & Sabharish Jayachandran
# License: MIT
# To report a bug or contribute use following link
# https://github.com/sabipu/Naive-Bayes-and-Fuzzy-System-Iris-Dataset

#================
# LIBRARIES USED
#================
import pandas as pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

#================================
# SETTING SOME GLOBAL VARIABLES
#================================
trainingSize = 0.3
randomSeed = 7

#====================
# READING THE DATASET
#====================
dataset = pandas.read_csv('./iris.data', sep=',', names=["Sepal Length(cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)", "Category"])

#==============================================
# REPLACING THE SPECIES OR CATEGORY OF FLOWERS
#==============================================
dataset.Category.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [1, 2, 3], inplace=True)


#=================================
# STRING VALUES IN ARRAY, X AND Y
#=================================
dataArray = dataset.values
X = dataArray[:,0:4]
Y = dataArray[:,4]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=trainingSize, random_state=randomSeed)


#========================
# NAVÏE BAYES CLASSIFIER
#========================
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting test dataset result
predictionClassifier = classifier.predict(X_test)

# Creating confusion matrix
confusionMatrix = confusion_matrix(Y_test, predictionClassifier)

# Accuracy of model
accuracy = accuracy_score(Y_test, predictionClassifier)


print '\n'
print 'Following is the confusion matrix'
print '================================='
print ' '
print confusionMatrix
print '\n'
print '\n'
print 'Following is the accuracy of classifier'
print '======================================='
print accuracy
print '\n'


#========================
# CROSS VALIDATION
#========================
kFold = model_selection.KFold(n_splits=10, random_state=randomSeed)

validationResults = model_selection.cross_val_score(GaussianNB(), X_train, Y_train, cv=kFold, scoring='accuracy')



print 'Mean value of prediction'
print '========================'
print validationResults.mean()
print '\n'
print '\n'
print 'Standard Deviation of prediction'
print '================================'
print validationResults.std()


