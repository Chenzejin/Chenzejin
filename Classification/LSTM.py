import scipy.io as scio
from sklearn import preprocessing
from sklearn.utils import class_weight
import numpy
from numpy import dot, eye, ones, zeros
import scipy.linalg
import random
import sklearn
import keras
import numpy as np
random.seed(21)
'''1是正类ASD，0是负类TD'''
total_histogram = scio.loadmat('./屏幕坐标点+头部姿态直方图存放/accumulative_histogram_C+P+Y+R.mat')
total_histogram = total_histogram['total_histogram']
label = total_histogram[:,0]
coordinate = total_histogram[:,1:326]
pitch = total_histogram[:,326:401]
yaw = total_histogram[:,401:476]
roll = total_histogram[:,476:]
coordinate_ts = np.reshape(coordinate,[405, 25, 13])
pitch_ts = np.reshape(pitch,[405, 25, 3])
yaw_ts = np.reshape(yaw,[405, 25, 3])
roll_ts = np.reshape(roll,[405, 25, 3])
C = coordinate_ts
C_P = np.concatenate((coordinate_ts,pitch_ts),-1)
C_Y = np.concatenate((coordinate_ts,yaw_ts),-1)
C_R = np.concatenate((coordinate_ts,roll_ts),-1)
C_P_Y_R = np.concatenate((coordinate_ts,pitch_ts,yaw_ts,roll_ts),-1)

index = [i for i in range(405)]
random.shuffle(index)
scio.savemat('index.mat',{'index':index})
print(index)
val = []
confusion_matrix = np.zeros([3,3])
best_model_file = "BestModel.h5"
    # Define several callbacks
best_model = keras.callbacks.ModelCheckpoint(best_model_file, monitor='val_acc',
                                verbose = 1, save_best_only = True)
# Fp = []
# Fn = []
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
#     # angle_train = angle_ts[index[0:218]]
#     # angle_test = angle_ts[index[218:]]
#     # displacement_train = displacement_ts[index[0:218]]
#     # displacement_test = displacement_ts[index[218:]]
    label_test_onehot = keras.utils.to_categorical(label_test)
    label_train_onehot = keras.utils.to_categorical(label_train)
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(25, 22)))
    model.add(keras.layers.LSTM(128,return_sequences=True))
    model.add(keras.layers.LSTM(128))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(3,activation='softmax'))
    class_weights = class_weight.compute_class_weight('balanced', np.unique(label_train), label_train)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=40, verbose=1)
    model.compile(loss='categorical_crossentropy',
                      optimizer='RMSprop',
                      metrics=['accuracy'])
    a = model.fit(feature_train, label_train_onehot,
                  batch_size=32, epochs=500,
                  validation_data=(feature_test, label_test_onehot),class_weight=class_weights,
                  callbacks=[best_model,early_stop])
    model1 = keras.models.load_model("BestModel.h5")
    predict = model1.predict_classes(feature_test)
#
    for x in range(len(label_test)):
        if label_test[x] == 0 and predict[x] == 0:
            confusion_matrix[0,0] += 1
        elif label_test[x] == 0  and predict[x] == 1:
            confusion_matrix[0,1] += 1
        elif label_test[x] == 0  and predict[x] == 2:
            confusion_matrix[0,2] += 1
        elif label_test[x] == 1 and predict[x] == 0:
            confusion_matrix[1,0] += 1
        elif label_test[x] == 1 and predict[x] == 1:
            confusion_matrix[1,1] += 1
        elif label_test[x] == 1 and predict[x] == 2:
            confusion_matrix[1,2] += 1
        elif label_test[x] == 2 and predict[x] == 0:
            confusion_matrix[2,0] += 1
        elif label_test[x] == 2 and predict[x] == 1:
            confusion_matrix[2,1] += 1
        elif label_test[x] == 2 and predict[x] == 2:
            confusion_matrix[2,2] += 1
    val_acc = a.history['val_acc']
    val_acc = max(val_acc)
    val.append(val_acc)
    print(confusion_matrix)
    # val_acc = a.history['val_acc']
#     val_acc = max(val_acc)

#     # print('序号:',z)
#     # print('Ture lable:',label_test)
#     # print('predict:',predict)
#     # print('FP:',Fp)
#     # print('Fn:',Fn)
val_acc1 = (confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2])/405
print(val_acc1)
print(val)
print(np.mean(val))