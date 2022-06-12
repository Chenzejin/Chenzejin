import tensorflow as tf
import math

reader = tf.TFRecordReader()
def _pre_train(record):
    features ={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([2], tf.float32),
        }
    parsed = tf.parse_single_example(record, features)
    images_uint8 = tf.decode_raw(parsed['image'], tf.uint8)
    labels_two = tf.cast(parsed['label'], tf.float32)
    images = tf.cast(images_uint8, dtype=tf.float32)
    images = tf.reshape(images, [36, 224, 3])
    final_image = tf.image.random_brightness(images, max_delta=63)
    final_image = tf.image.random_contrast(final_image, lower=0.2, upper=1.8)
    final_image = tf.image.random_saturation(final_image, 0, 5)
    final_image = tf.image.per_image_standardization(final_image)
    labels = labels_two
    return final_image, labels
def _pre_test(record):
    features ={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([2], tf.float32),
        }
    parsed = tf.parse_single_example(record, features)
    images_uint8 = tf.decode_raw(parsed['image'], tf.uint8)
    labels_two = tf.cast(parsed['label'], tf.float32)
    images = tf.cast(images_uint8, dtype=tf.float32)
    images = tf.reshape(images, [36, 224, 3])
    final_image = tf.image.per_image_standardization(images)
    labels = labels_two
    return final_image, labels
def loss(logits,label_batches,batch_size):
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    loss_filter_fn =  exclude_batch_norm

    # Add weight decay to the loss.
    l2_re = 0.0002 * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    l2_loss = tf.square(logits-label_batches)
    # cost = l2_re+l2_loss
    cost = (tf.reduce_sum(l2_loss)+l2_re)/batch_size
    return cost
def angle_loss(logits,label):
    theta_pre = logits[:,0]
    theta_pre = tf.reshape(theta_pre,[-1,1])
    theta_tru = label[:,0]
    theta_tru = tf.reshape(theta_tru, [-1, 1])
    phi_pre = logits[:,1]
    phi_pre = tf.reshape(phi_pre, [-1, 1])
    phi_tru = label[:,1]
    phi_tru = tf.reshape(phi_tru, [-1, 1])
    x_pre = -tf.cos(theta_pre) * tf.sin(phi_pre)
    y_pre = -tf.sin(theta_pre)
    z_pre = -tf.cos(theta_pre) * tf.cos(phi_pre)
    x_tru = -tf.cos(theta_tru) * tf.sin(phi_tru)
    y_tru = -tf.sin(theta_tru)
    z_tru = -tf.cos(theta_tru) * tf.cos(phi_tru)
    pre = tf.concat([x_pre,y_pre,z_pre],axis=1)
    tru = tf.concat([x_tru,y_tru,z_tru],axis=1)
    temp_pre = tf.sqrt(tf.reduce_sum(tf.multiply(pre, pre),axis=1))
    temp_pre = tf.reshape(temp_pre,[-1,1])
    temp_tru = tf.sqrt(tf.reduce_sum(tf.multiply(tru, tru), axis=1))
    temp_tru = tf.reshape(temp_tru, [-1, 1])
    temp_l2 = tf.multiply(temp_pre, temp_tru)
    temp_dot = tf.reduce_sum(tf.multiply(pre, tru),axis=1)
    temp_dot = tf.reshape(temp_dot,[-1,1])
    angle = tf.abs(temp_dot/temp_l2)
    angle = tf.acos(angle)
    angle = angle*180/math.pi
    return angle

def learning_rate_with_decay(
        initial_learning_rate, boundary_epochs, decay_rates):
  initial_learning_rate = initial_learning_rate
  boundaries = [int(epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn
