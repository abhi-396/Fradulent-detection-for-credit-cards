# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:37:34 2019

@author: Abhishek Sharma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading data
dataset = pd.read_csv('creditcard.csv')

#Data Preprocessing
df = dataset.iloc[:, 1:]
df_train_all = df.iloc[:150000]
df_test_all = df.iloc[150000:]
df_train_fraud = df_train_all[df_train_all['Class'] == 1]
df_train_non_fraud = df_train_all[df_train_all['Class'] == 0]

""" Sampling training data(Random under Sampling) """
df_sample = df_train_non_fraud.sample(300)
df_train = df_train_fraud.append(df_sample)
df_train = df_train.sample(frac=1)
X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:, -1]
X_test = df_test_all.iloc[:, :-1]
y_test = df_test_all.iloc[:, -1]

# Fitting SVC model to sampled training data
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0, class_weight={0:0.60, 1:0.40})
classifier.fit(X_train, y_train)

# predicting test set
y_pred = classifier.predict(X_test)

# Finding confusion matrix and accuracy of model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print('Confusion matrix', cm)
print('Accuracy', accuracy)

# Visualizing training sample dataset through bargraphs
plt.bar([0, 1], [300, 293])
plt.xlabel('Class')
plt.ylabel('Amount')
plt.title('Amount vs Class')

# Visualizing training sample dataset through bargraphs
test_group = df_test_all.groupby(['Class']).groups
plt.bar([0, 1], [test_group[0].size, test_group[1].size])
plt.xlabel('Class')
plt.ylabel('Amount')
plt.title('Amount vs Class')
