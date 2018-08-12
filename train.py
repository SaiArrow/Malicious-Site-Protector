import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn.externals import joblib
import time
import matplotlib.pylab as plt

import sys
import joblib

labels=[]
features=[]
file=open('Training Dataset.arff').read()
list=file.split('\n')
data=np.array(list)
data1=[i.split(',') for i in data]
data1=data1[0:-1]
for i in data1:
	labels.append(i[30])
data1=np.array(data1)
features=data1[:,:-1]
features=features[:,[0,1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,22,23,24,25,27,29]]
#print features
features=np.array(features).astype(np.float)

##### HAS TO BE CHANGED TO ALL ENTRIES OF THE DATASET
features_train=features[:10000]
labels_train=labels[:10000]
features_test=features[10000:]
labels_test=labels[10000:]


clf = ensemble.ExtraTreesClassifier(n_estimators=57,min_samples_split=2,random_state=40)
clf.fit(features, labels)


print("total=",clf.score(features_test,labels_test))

joblib.dump(clf,'classifier/extratree.pkl')



# print("\n\n ""Random Forest Algorithm Results"" ")
# clf4 = RandomForestClassifier(min_samples_split=7, verbose=True)
# clf4.fit(features_train, labels_train)
# importances = clf4.feature_importances_
# std = np.std([tree.feature_importances_ for tree in clf4.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
# # Print the feature ranking
# # print("Feature ranking:")
# # for f in range(features_train.shape[1]):
# #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
#
# pred4=clf4.predict(features_test)
# print(classification_report(labels_test, pred4))
# print('The accuracy is:', accuracy_score(labels_test, pred4))
# # print metrics.confusion_matrix(labels_test, pred4)
#
# #sys.setrecursionlimit(9999999)
# # joblib.dump(clf4, 'classifier/random_forest.pkl',compress=9)
#
# trees = range(100)
# accuracy = np.zeros(100)
# for x in range(len(trees)):
# 	clf = ensemble.ExtraTreesClassifier(n_estimators=57, min_samples_split=2,random_state=x)
# 	clf.fit(features_train, labels_train)
# 	result = clf.predict(features_test)
# 	accuracy[x] = metrics.accuracy_score(labels_test, result)
# 	print(x)
# plt.cla()
# plt.plot(trees, accuracy)
# plt.show()
