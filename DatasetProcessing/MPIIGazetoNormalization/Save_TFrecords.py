
import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]='1'
SAVE_PATH = 'G:\\Normalized图像\\MPIIGaze_p14.tfrecords'

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
writer = tf.python_io.TFRecordWriter(SAVE_PATH)
sess = tf.Session()
path = 'G:\\Normalized图像\\MPIIGaze_p14.h5'
f = h5py.File(path,'r')
imgdata = f['/img'][:]
label = f['/label'][:]
f.close()
[m,_,_,_] = imgdata.shape
imgdata = imgdata.astype(np.uint8)
print(label.dtype)
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