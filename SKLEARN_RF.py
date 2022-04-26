# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 00:21:38 2020

@author: aina
"""
import pandas as pd
import numpy as np
import timeit
import scipy.stats as sps
# from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import sys


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

from pprint import pprint

start = timeit.default_timer()

start = timeit.default_timer()
#Import the dataset and define the feature as well as the target datasets / columns#
dataset = pd.read_csv('WBCDF1.data', #names=["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1","class",])
                      names=["ID","Clumpthickness","Uniformitycellsize","Uniformitycellshape","marginaladhesion","singleepithelialsize","Barenuclei","Chromatin","Nucleoli","Mitoses","class",])#Import all columns omitting the fist which consists the names of the animals

# We drop the animal names since this is not a good feature to split the data on
dataset = dataset.drop('ID', axis=1)

#######SKLEARN###########
for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])

X = dataset.drop(['class'],axis=1)
Y = dataset['class']


num_trees = 100
model = RandomForestClassifier(criterion="entropy", n_estimators=num_trees)
#Cross validation

accuracy = cross_validate(model,X,Y,cv=10)['test_score']
print('The accuracy based on SKLEARN is: ',sum(accuracy)/len(accuracy)*100,'%')
