import tensorflow as tf
import numpy as np
import os
from model_utils import _pre_train,_pre_test,loss,angle_loss

os.environ["CUDA_VISIBLE_DEVICES"]='1'

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
weight_decay = 0.1
epochs = 100
generations = 657
init_learning_rate = 0.005

batch_size = 32
sess = tf.Session()
path_train = 'E:\Documents\图片和视频数据库\双眼MPIIGaze文件\\train'
path_test = 'E:\Documents\图片和视频数据库\双眼MPIIGaze文件\\test'
training_filenames = []
for train_file in os.listdir(path_train):
    training_filenames.append(path_train + '\\' + train_file)
print(training_filenames)
test_filenames = []
for test_file in os.listdir(path_test):
    test_filenames.append(path_test + '\\' + test_file)
checkpoint_dir = 'E:\zyh\跨MPIIGaze数据库模型保存\\'
reader = tf.TFRecordReader()
filenames = tf.placeholder(tf.string, shape=[None])
train_dataset = tf.data.TFRecordDataset(filenames)
train_dataset = train_dataset.map(_pre_train)
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.TFRecordDataset(filenames)
test_dataset = test_dataset.map(_pre_test)
test_dataset = test_dataset.shuffle(buffer_size=10000)
test_dataset = test_dataset.repeat()
test_dataset = test_dataset.batch(batch_size)
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)
next_element = iterator.get_next()

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
    # img = AttentionModule(
    #     inputs=img, filters=256,stage=3,
    #     p=1, t=2, r=1, training=training, name='AttentionModule3_', data_format=data_format)
    for i in range(0, 4):
        img = residual_unit(
            inputs=img, filters=256, projection_shortcut=False,
            strides=1, training=training,
            name='Final_1{}'.format(i + 2), data_format=data_format)
    img = residual_unit(
        inputs=img, filters=512, projection_shortcut=True,
        strides=2, training=training,
        name='Final_1', data_format=data_format)
    for i in range(0,2):
        img = residual_unit(
            inputs=img, filters=512, projection_shortcut=False,
            strides=1, training=training,
            name='Final_2{}'.format(i+2), data_format=data_format)
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

images = tf.placeholder(tf.float32,shape = [None, 36, 224, 3])
targets = tf.placeholder(tf.float32,shape = [None, 2])

model_output = AttentionResnet_model(images,training=TRAINING)
L2_loss = loss(model_output,targets,batch_size)
L2_loss_test = angle_loss(model_output,targets,batch_size)
rate = tf.placeholder(tf.float32)
# optimizer = tf.train.MomentumOptimizer(learning_rate=rate,momentum=momentum)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(L2_loss)
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./Model_Save')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('E:\Documents\\logs', sess.graph)
# summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    # epoch_learning_rate = init_learning_rate
learning_rate = init_learning_rate
for i in range(epochs):
    sess.run(train_init_op, feed_dict={filenames: training_filenames})
    if i % 40 == 0 or i % 60 == 0:
        learning_rate = learning_rate * weight_decay
    sess.run(tf.assign(TRAINING,True))
    for j in range(generations):
        train_image, train_label = sess.run(next_element)
        _, loss_value = sess.run([train, L2_loss],feed_dict={rate:learning_rate, images: train_image, targets: train_label})
        line = "epoch: %d generations: %d, train_loss: %.4f \n" % (
            i+1, j+1, loss_value)
        print(line)
        with open('E:\Documents\图片和视频数据库\loss\\UT_logs.txt', 'a') as f:
            f.write(line)
        # test_loss_value = sess.run(L2_loss_test)
    test = 0
    sess.run(test_init_op, feed_dict={filenames: test_filenames})
    sess.run(tf.assign(TRAINING,False))
    for m in range(10):
        test_image, test_label = sess.run(next_element)
        test_loss_value = sess.run(L2_loss_test, feed_dict={images: test_image, targets: test_label})
        test = test + test_loss_value
    test = test/10
    output2 = 'Epoch {}:  Test Loss = {:.5f} Learning rate: {:.8f}'.format((i + 1), test, learning_rate)
    with open('E:\Documents\图片和视频数据库\loss\\test_logs.txt', 'a') as f:
        f.write(output2)
    print(output2)
