#https://www.kaggle.com/c/house-prices-advanced-regression-techniques
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import *
from enum import Enum


trainfile = pd.read_csv("train.csv",index_col=0)
#print trainfile["MSSubClass"][1]
#print trainfile.head()

train_data = trainfile.as_matrix()
#print train_data[0]
#print train_data

train_features = train_data[:,:-1]
train_labels = train_data[:,-1:]
#print train_labels
'''
reg = LinearRegression()
reg.fit(train_features,train_labels)
coefs = reg.coef_
print coefs
'''

#print trainfile.info()

# print trainfile["MSSubClass"].head()

# MSSubClass_dummies = pd.get_dummies(trainfile["MSSubClass"])
# print MSSubClass_dummies

'''
		20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES
'''

'''
def MSSC(x):
	arr = [20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190]
	j=0
	for i in arr:
		if x==i:
			return j
		j+=1


mssc = np.zeros(16)
msscsum = np.zeros(16)
i=0
for x in train_features[:,0:1]:
	msscsum[MSSC(x[0])]+=train_labels[i,0]
	mssc[MSSC(x[0])]+=1
	i+=1
average =  msscsum/mssc
# print average.shape
# print train_features[:,0:1].shape
plt.scatter(['20','30','40','45','50','60','70','75','80','85','90','120','150','160','180','190'],average)
plt.show()
'''
