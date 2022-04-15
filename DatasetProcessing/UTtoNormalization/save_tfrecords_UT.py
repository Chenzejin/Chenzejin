import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

SAVE_PATH = 'D:\\UT标准化双眼图像存放\\s30.tfrecords'
path = 'D:\\UT标准化双眼图像存放\\s30.mat'
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
writer = tf.python_io.TFRecordWriter(SAVE_PATH)
sess = tf.Session()

data = sio.loadmat(path)
data = data['data']
data = data[0,0]
imgdata = data['img']
gaze = data['gaze']


[m,_,_,_] = imgdata.shape
for i in range(m):

    imagedata = imgdata[i,:,:,:]
    image = imagedata.tostring()
    l = gaze[i,:]

    example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(image),
                'label': _float_feature(l),
                }))
    writer.write(example.SerializeToString())
writer.close()