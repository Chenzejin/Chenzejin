# 重新加载训练好的模型
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np
import gc
os.environ["CUDA_VISIBLE_DEVICES"]='1'

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
weight_decay = 0.1
epochs = 100
generations = 657
init_learning_rate = 0.001
batch_size = 32

model_path = 'E:\zyh\Cross_data\model_save\\model.ckpt-182736'

def batch_norm(inputs, training, data_format, name):

  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True, name=name)

def fixed_padding(inputs, kernel_size, data_format):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, name):
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format, name=name)
def residual_unit(inputs, filters, training, projection_shortcut,
                         strides, data_format, name):
  shortcut = inputs
  filters_out = filters*4
  inputs = batch_norm(inputs, training, data_format, name=name+"batch_normalization_1")
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not False:
    shortcut = conv2d_fixed_padding(
    inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
    data_format=data_format, name=name+"Residual_Con_pro")

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format, name=name+"Residual_Con_1")

  inputs = batch_norm(inputs, training, data_format, name=name+"batch_normalization_2")
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format, name=name+"Residual_Con_2")

  inputs = batch_norm(inputs, training, data_format, name=name+"batch_normalization_3")
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format, name=name+"Residual_Con_3")

  return inputs + shortcut


def AttentionModule(inputs, filters, p, t, r, stage,
                training, name, data_format):
  filters_out = filters * 4
  for i in range(p):
    inputs = residual_unit(
              inputs=inputs, filters=filters, projection_shortcut=True,
              strides=1, training=training,
              name=name+'Pre_{}'.format(i+1)+'_', data_format=data_format)

  trunk_inputs = inputs
  for j in range(t):
    trunk_inputs = residual_unit(
              inputs=trunk_inputs, filters=filters,projection_shortcut=False,
              strides=1, training=training,
              name=name+'Trunk{}_'.format(j+1), data_format=data_format)
  if stage == 1:
    mask_inputs = inputs
    mask_inputs = tf.layers.max_pooling2d(
              inputs=mask_inputs, pool_size=3, name=name+'DownSample_1',
              strides=2, padding='SAME', data_format=data_format)
    for k in range(r):
      mask_inputs = residual_unit(
          inputs=mask_inputs, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Mask{}_'.format(k + 1)+'_', data_format=data_format)
    mask_inputs_1 = mask_inputs
    mask_inputs = tf.layers.max_pooling2d(
              inputs=mask_inputs_1, pool_size=3, name=name + 'DownSample_2',
              strides=2, padding='SAME', data_format=data_format)
    for z in range(r):
      mask_inputs = residual_unit(
          inputs=mask_inputs, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Mask{}_'.format(z+r+1)+'_', data_format=data_format)
    mask_inputs_2 = mask_inputs
    mask_inputs = tf.layers.max_pooling2d(
        inputs=mask_inputs_2, pool_size=3, name=name + 'DownSample_3',
        strides=2, padding='SAME', data_format=data_format)
    for p in range(2*r):
      mask_inputs = residual_unit(
          inputs=mask_inputs, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Mask{}_'.format(2*r+p+1)+'_', data_format=data_format)
    weights1 = tf.get_variable(name+'weights_deconv1',
                            shape=[3, 3, filters_out, filters_out],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    mask_inputs = tf.nn.conv2d_transpose(
          mask_inputs, weights1, output_shape=tf.shape(mask_inputs_2),
          strides=[1, 2, 2, 1], padding='SAME')
    add2 = residual_unit(
          inputs=mask_inputs_2, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Add2_', data_format=data_format)
    mask_inputs = tf.add(add2,mask_inputs)
    for q in range(r):
      mask_inputs = residual_unit(
          inputs=mask_inputs, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Mask{}_'.format(4*r+q+1)+'_', data_format=data_format)
    weights2 = tf.get_variable(name+'weights_deconv2',
                             shape=[3, 3, filters_out, filters_out],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    mask_inputs = tf.nn.conv2d_transpose(
        mask_inputs, weights2, output_shape=tf.shape(mask_inputs_1),
        strides=[1, 2, 2, 1], padding='SAME')
    add1 = residual_unit(
          inputs=mask_inputs_1, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Add1_', data_format=data_format)
    mask_inputs = tf.add(add1,mask_inputs)

    for f in range(r):
      mask_inputs = residual_unit(
          inputs=mask_inputs, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Mask{}_'.format(5*r+f+1)+'_', data_format=data_format)
    weights2 = tf.get_variable(name+'weights_deconv3',
                             shape=[3, 3, filters_out, filters_out],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    mask_inputs = tf.nn.conv2d_transpose(
        mask_inputs, weights2, output_shape=tf.shape(inputs),
        strides=[1, 2, 2, 1], padding='SAME')
  # if stage == 1:
  #   add2 = residual_unit(
  #             inputs=mask_inputs_2, filters=filters,projection_shortcut=False,
  #             strides=1, training=training,
  #             name=name+'Add{}_'.format(stage+1), data_format=data_format)
  #   mask_inputs = tf.add(add2,mask_inputs)
  elif stage == 2:
    mask_inputs = inputs
    mask_inputs = tf.layers.max_pooling2d(
              inputs=mask_inputs, pool_size=3, name=name+'DownSample_1',
              strides=2, padding='SAME', data_format=data_format)
    for k in range(r):
      mask_inputs = residual_unit(
          inputs=mask_inputs, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Mask{}_'.format(k + 1)+'_', data_format=data_format)
    mask_inputs_2 = mask_inputs
    mask_inputs = tf.layers.max_pooling2d(
              inputs=mask_inputs_2, pool_size=3, name=name + 'DownSample_2',
              strides=2, padding='SAME', data_format=data_format)

    for p in range(2*r):
      mask_inputs = residual_unit(
          inputs=mask_inputs, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Mask{}_'.format(r+p+1)+'_', data_format=data_format)
    weights1 = tf.get_variable(name+'weights_deconv1',
                            shape=[3, 3, filters_out, filters_out],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    mask_inputs = tf.nn.conv2d_transpose(
          mask_inputs, weights1, output_shape=tf.shape(mask_inputs_2),
          strides=[1, 2, 2, 1], padding='SAME')
    add2 = residual_unit(
        inputs=mask_inputs_2, filters=filters, projection_shortcut=False,
        strides=1, training=training,
        name=name + 'Add2_', data_format=data_format)
    mask_inputs = tf.add(add2, mask_inputs)

    for q in range(r):
      mask_inputs = residual_unit(
          inputs=mask_inputs, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Mask{}_'.format(3*r+q+1)+'_', data_format=data_format)
    weights2 = tf.get_variable(name+'weights_deconv2',
                             shape=[3, 3, filters_out, filters_out],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    mask_inputs = tf.nn.conv2d_transpose(
        mask_inputs, weights2, output_shape=tf.shape(inputs),
        strides=[1, 2, 2, 1], padding='SAME')

  else:
    mask_inputs = inputs
    mask_inputs = tf.layers.max_pooling2d(
              inputs=mask_inputs, pool_size=3, name=name+'DownSample_1',
              strides=2, padding='SAME', data_format=data_format)
    for p in range(2*r):
      mask_inputs = residual_unit(
          inputs=mask_inputs, filters=filters, projection_shortcut=False,
          strides=1, training=training,
          name=name + 'Mask{}_'.format(p+1)+'_', data_format=data_format)
    weights1 = tf.get_variable(name+'weights_deconv1',
                            shape=[3, 3, filters_out, filters_out],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    mask_inputs = tf.nn.conv2d_transpose(
          mask_inputs, weights1, output_shape=tf.shape(inputs),
          strides=[1, 2, 2, 1], padding='SAME')


  mask_inputs = batch_norm(mask_inputs, training, data_format, name=name + "Linear_batch_normalization_1")
  mask_inputs = tf.nn.relu(mask_inputs)
  mask_inputs = conv2d_fixed_padding(
      inputs=mask_inputs, filters=filters * 4, kernel_size=1, strides=1,
      data_format=data_format, name=name + "Linear_Conv_1")
  mask_inputs = batch_norm(mask_inputs, training, data_format, name=name + "Linear_batch_normalization_2")
  mask_inputs = tf.nn.relu(mask_inputs)
  mask_inputs = conv2d_fixed_padding(
      inputs=mask_inputs, filters=filters * 4, kernel_size=1, strides=1,
      data_format=data_format, name=name + "Linear_Conv_2")
  mask_inputs = tf.nn.sigmoid(mask_inputs)
  outputs = tf.add(trunk_inputs, tf.multiply(trunk_inputs, mask_inputs))
  outputs = residual_unit(
              inputs=outputs, filters=filters,projection_shortcut=False,
              strides=1, training=training,
              name=name+'Residual_Con_fin_', data_format=data_format)
  return outputs

def AttentionResnet_model(img, training, data_format='channels_last'):
    img = conv2d_fixed_padding(
            inputs=img, filters=64, kernel_size=3,
            strides=2, data_format=data_format , name='Initial_Conv')
    img = batch_norm(img, training, data_format, name='Initial_batch_normalization')
    img = tf.nn.relu(img)
    img = tf.layers.max_pooling2d(inputs=img, pool_size=3, strides=2,
                                  padding='SAME', name='Initial_Pooling')
    img = residual_unit(
        inputs=img, filters=64, projection_shortcut=True,
        strides=1, training=training,
        name='Initial_Residual', data_format=data_format)
    img = AttentionModule(
            inputs=img, filters=64, stage=1,
            p=1, t=2, r=1, training=training, name='AttentionModule1_',data_format=data_format)
    img = residual_unit(
        inputs=img, filters=128, projection_shortcut=True,
        strides=2, training=training,
        name='Middle1_', data_format=data_format)
    img = AttentionModule(
        inputs=img, filters=128, stage=2,
        p=1, t=2, r=1, training=training, name='AttentionModule2_', data_format=data_format)
    img = residual_unit(
        inputs=img, filters=256, projection_shortcut=True,
        strides=2, training=training,
        name='Middle2_', data_format=data_format)
    img = AttentionModule(
        inputs=img, filters=256,stage=3,
        p=1, t=2, r=1, training=training, name='AttentionModule3_', data_format=data_format)
    img = residual_unit(
        inputs=img, filters=512, projection_shortcut=True,
        strides=2, training=training,
        name='Final_1', data_format=data_format)
    for i in range(0,2):
        img = residual_unit(
            inputs=img, filters=512, projection_shortcut=False,
            strides=1, training=training,
            name='Final_{}'.format(i+2), data_format=data_format)
    axes = [2, 3] if data_format == 'channels_first' else [1, 2]
    outputs = tf.reduce_mean(img, axes, keepdims=True)
    outputs = tf.identity(outputs, 'final_reduce_mean')

    outputs = tf.reshape(outputs, [-1, 2048])
    weights = tf.get_variable('weights_final',
                              shape=[2048, 2],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.001, dtype=tf.float32))
    biases = tf.get_variable('biases_final',
                             shape=[2],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0))
    outputs = tf.matmul(outputs, weights) + biases

    return outputs

path_evl = 'E:\zyh\标准化双眼图像mat文件'




# savename = 'E:\\1mat.mat'
# sio.savemat(savename,{'map':z})
save_path = 'G:\双眼预测屏幕坐标点\\'
sec_dir = os.listdir(path_evl)
for i in range(len(sec_dir)):
    sec_path = path_evl+'\\'+sec_dir[i+6]
    filename = os.listdir(sec_path)
    for j in range(len(filename)):
        images = tf.placeholder(tf.float32, shape=[None, 36, 224, 3])
        model_out = AttentionResnet_model(images, False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            file = sec_path+'\\'+filename[j]
            name = file[19:len(file)].split('\\')
            savename = save_path+name[1]+'_'+name[2]
            data = sio.loadmat(file)
            a = data['normdata']
            a = a[0, 0]
            img = a['img']
            [m, _, _, _] = img.shape
            half = np.floor(m/2)
            half = int(half)
            img1 = img[0:half,:,:,:]
            img2 = img[half:,:,:,:]
            img1 = tf.map_fn(lambda x: tf.image.per_image_standardization(x), img1, parallel_iterations=half)
            img2 = tf.map_fn(lambda x: tf.image.per_image_standardization(x), img2, parallel_iterations=m-half)
            image1 = sess.run(img1)
            image2 = sess.run(img2)
            pose = a['pose']
            # tensor_name_conv1_1 = "Sigmoid:0" #Sigmoid:0和Sigmoid_2:0
            z1 = sess.run(model_out, feed_dict={images: image1})
            z2 = sess.run(model_out,feed_dict={images: image2})
            z = np.concatenate((z1,z2),axis=0)
            sio.savemat(savename,{'gaze':z,'pose':pose})

            # del img ,img1, img2, image1, image2, pose, z, a,z1,z2,data,file,name
            # gc.collect()
            print(j+1)
            print(savename+'...Saved')
        tf.reset_default_graph()


# test_images = img_1




# tensor_name_conv1_1 = "model_definition/conv5_3/model_definition/conv5_3:0"



# saver = tf.train.import_meta_graph(graph_path)







