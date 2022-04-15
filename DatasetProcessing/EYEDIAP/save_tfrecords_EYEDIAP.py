import tensorflow as tf
import numpy as np
import scipy.io as sio
import os

SAVE_PATH = 'D:\EYEDIAP\\15Frame\TF_Eh'
path = 'D:\EYEDIAP\\15Frame\\Enhanced'
sess = tf.Session()
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
for files in os.listdir(path):
    files_name = path + '\\'+ files
    temp = files.split('.')
    save_name = SAVE_PATH+'\\'+temp[0]+'.tfrecords'
    writer = tf.python_io.TFRecordWriter(save_name)


    data = sio.loadmat(files_name)
    data = data['data']
    data = data[0,0]
    imgdata = data['img']
    label = data['gaze']
    print(label.shape,imgdata.shape)

    [m,_,_,_] = imgdata.shape
    for i in range(m):

        imagedata = imgdata[i,:,:,:]
        image = imagedata.tostring()
        l = label[i,:]
        example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(image),
                    'label': _float_feature(l),
                    }))
        writer.write(example.SerializeToString())
    writer.close()