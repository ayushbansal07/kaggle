#https://www.kaggle.com/c/data-science-london-scikit-learn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB

trainfile = pd.read_csv("train.csv",header=None)
lblfile = pd.read_csv("trainLabels.csv",header=None)
testfile = pd.read_csv("test.csv",header=None)
#print trainfile.head()

train_features = trainfile.as_matrix()
train_labels = lblfile.as_matrix()
#test_features = testfile.as_matrix()
#print type(train_features)

# print train_features
# print train_labels
test_features = train_features[:900]
test_labels = train_labels[:900]
train_features = train_features[900:]
train_labels = train_labels[900:]
clf = svm.SVC(kernel='rbf')
clf.fit(train_features,train_labels)

print "SVM-> ",clf.score(test_features,test_labels)

clf2 = tree.DecisionTreeClassifier(min_samples_split=4)
clf2.fit(train_features,train_labels)
print "DT-> ",clf2.score(test_features,test_labels)

clf3 = RFC()
clf3.fit(train_features,train_labels)
print "RFC-> ",clf3.score(test_features,test_labels)

clf4 = GaussianNB()
clf4.fit(train_features,train_labels)
print "NB-> ",clf4.score(test_features,test_labels)
#print clf3.predict(test_features)

#print predictions
# data=[]
# for i in range(0,9000):
# 	data.append([i+1,predictions[i]])
# print data
# pd.DataFrame(data).to_csv("predictions.csv",index=False,header=['Id','Solution'])






