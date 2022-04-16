import scipy.io as scio
from sklearn.svm import SVC
import numpy as np
import random
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
random.seed(21)
confusion_matrix = np.zeros([3,3])
'''1是正类ASD，0是负类TD'''
total_histogram = scio.loadmat('./屏幕坐标点+头部姿态直方图存放/accumulative_histogram_C+P+Y+R.mat')
total_histogram = total_histogram['total_histogram']
label = total_histogram[:,0].astype(int)
C_P_Y_R = total_histogram[:,1:]
index = [i for i in range(405)]
random.shuffle(index)
score_list = []
for j in range(10):
    print('It is %dth turns'%j)
    if j == 0:
        label_test = label[index[0:40]]
        z = index[0:40]
        label_train = label[index[40:]]
        feature_test = C_P_Y_R[index[0:40]]
        feature_train = C_P_Y_R[index[40:]]
    elif j == 9:
        label_test = label[index[(40 * j):]]
        z = index[(40 * j):]
        label_train = label[index[0:(40 * j)]]
        feature_test = C_P_Y_R[index[(40 * j):]]
        feature_train = C_P_Y_R[index[0:(40 * j)]]
    else:
        label_test = label[index[(40*j):(40*(j+1))]]
        z = index[(40*j):(40*(j+1))]
        label_train = np.concatenate([label[index[0:(40*j)]],label[index[40*(j+1):]]])
        feature_test = C_P_Y_R[index[(40*j):(40*(j+1))]]
        feature_train = np.concatenate([C_P_Y_R[index[0:(40*j)]],C_P_Y_R[index[40*(j+1):]]])
    svc = XGBClassifier()
    svc.fit(feature_train,label_train)
    predict = svc.predict(feature_test)
    for x in range(len(label_test)):
        if label_test[x] == 0 and predict[x] == 0:
            confusion_matrix[0, 0] += 1
        elif label_test[x] == 0 and predict[x] == 1:
            confusion_matrix[0, 1] += 1
        elif label_test[x] == 0 and predict[x] == 2:
            confusion_matrix[0, 2] += 1
        elif label_test[x] == 1 and predict[x] == 0:
            confusion_matrix[1, 0] += 1
        elif label_test[x] == 1 and predict[x] == 1:
            confusion_matrix[1, 1] += 1
        elif label_test[x] == 1 and predict[x] == 2:
            confusion_matrix[1, 2] += 1
        elif label_test[x] == 2 and predict[x] == 0:
            confusion_matrix[2, 0] += 1
        elif label_test[x] == 2 and predict[x] == 1:
            confusion_matrix[2, 1] += 1
        elif label_test[x] == 2 and predict[x] == 2:
            confusion_matrix[2, 2] += 1
    print(confusion_matrix)
    score = svc.score(feature_test, label_test)
    score_list.append(score)
val_acc1 = (confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2])/405
print(val_acc1)
print(np.mean(score_list))