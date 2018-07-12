'''
Face Feature Point Detection
'''
import tensorflow as tf
import cv2
import numpy as np
from util import load_image_label


# Weight initialization (Xavier's init)
def weight_xavier_init(shape, n_inputs, n_outputs, activefuncation='sigomd', uniform=True, variable_name=None):
    if activefuncation == 'sigomd':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.Variable(initial, name=variable_name)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs))
            initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
            return tf.Variable(initial, name=variable_name)
    elif activefuncation == 'relu':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * np.sqrt(2)
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.Variable(initial, name=variable_name)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * np.sqrt(2)
            initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
            return tf.Variable(initial, name=variable_name)
    elif activefuncation == 'tan':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * 4
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.Variable(initial, name=variable_name)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * 4
            initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
            return tf.Variable(initial, name=variable_name)


# Bias initialization
def bias_variable(shape, variable_name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=variable_name)


# 2D convolution
def conv2d(x, W, strides=1):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return conv_2d


# Max Pooling
def max_pool_2x2(x, Inception=False):
    if Inception:
        pool2d = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    else:
        pool2d = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool2d


# conv_bn_relu_drop
def conv_bn_relu_drop(x, kernalshape, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernalshape, n_inputs=kernalshape[0] * kernalshape[1] * kernalshape[2],
                               n_outputs=kernalshape[-1], activefuncation='relu', variable_name=scope + 'W')
        B = bias_variable([kernalshape[-1]], variable_name=scope + 'B')
        conv = conv2d(x, W) + B
        conv = tf.nn.relu(conv)
        return conv


# conv_bn_relu_drop
def fullconnected_bn_relu_drop(x, kernalshape, active=True, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernalshape, n_inputs=kernalshape[0] * kernalshape[1],
                               n_outputs=kernalshape[-1], activefuncation='relu', variable_name=scope + 'W')
        B = bias_variable([kernalshape[-1]], variable_name=scope + 'B')
        conv = tf.matmul(x, W) + B
        if active:
            conv = tf.nn.relu(conv)
        return conv


def _create_conv_net(X, image_width, image_height, image_channel, n_class=1):
    inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # layer1
    layer1 = conv_bn_relu_drop(inputX, kernalshape=[3, 3, image_channel, 32], scope='layer1')
    layer1 = conv_bn_relu_drop(layer1, kernalshape=[3, 3, 32, 32], scope='layer1_1')
    pool1 = max_pool_2x2(layer1)
    # layer2
    layer2 = conv_bn_relu_drop(pool1, kernalshape=[3, 3, 32, 64], scope='layer2')
    layer2 = conv_bn_relu_drop(layer2, kernalshape=[3, 3, 64, 64], scope='layer2_1')
    pool2 = max_pool_2x2(layer2)
    # layer3
    layer3 = conv_bn_relu_drop(pool2, kernalshape=[3, 3, 64, 128], scope='layer3')
    layer3 = conv_bn_relu_drop(layer3, kernalshape=[3, 3, 128, 128], scope='layer3_1')
    pool3 = max_pool_2x2(layer3)
    # layer4
    layer4 = conv_bn_relu_drop(pool3, kernalshape=[3, 3, 128, 256], scope='layer4')
    layer4 = conv_bn_relu_drop(layer4, kernalshape=[3, 3, 256, 256], scope='layer4_1')
    pool4 = max_pool_2x2(layer4)

    # Golble Average Pooling
    GAP = tf.reduce_mean(pool4, axis=(1, 2))
    # FC1
    GAP = tf.reshape(GAP, shape=[-1, 256])
    fc1 = fullconnected_bn_relu_drop(GAP, kernalshape=[256, 1024], scope='fc1')
    # FC2
    fc2 = fullconnected_bn_relu_drop(fc1, kernalshape=[1024, 500], scope='fc2')
    # FC3
    fc3 = fullconnected_bn_relu_drop(fc2, kernalshape=[500, n_class], active=False, scope='fc3')
    return fc3


def show_result(x, y, name=None):
    img = x.reshape(96, 96) * 255.
    img = np.clip(img, 0, 255).astype('uint8')
    for step in range(0, 30, 2):
        cv2.circle(img, center=(int(np.round(y[step] * 96)), int(np.round(y[step + 1] * 96))), radius=1,
                   color=(255, 0, 0))
    img = cv2.resize(img, dsize=(128, 128))
    cv2.imwrite(name, img)


image_size = 9216
image_width = image_height = 96
image_labels = 30

test_images, _ = load_image_label('test.csv')

# Create Input and Output
X = tf.placeholder('float', shape=[None, image_width * image_height])  # mnist data image of shape 96*96=9216
Y_gt = tf.placeholder('float', shape=[None, image_labels])  # 0-30 digits recognition => 10 classes

Y_pred = _create_conv_net(X=X, image_width=96, image_height=96, image_channel=1, n_class=image_labels)
'''
TensorFlow Session
'''
# start TensorFlow session
init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(init)
saver.restore(sess, 'model\my-model')
srcimg = cv2.imread('topson.jpg')
srcimg = cv2.resize(srcimg, (96, 96))
grayimg = cv2.cvtColor(srcimg, cv2.COLOR_RGB2GRAY)
grayimg = np.reshape(grayimg, (96 * 96)) / 255.
predict = Y_pred.eval(feed_dict={X: [grayimg]}, session=sess)
show_result(grayimg, predict[0], name=str(10) + '.jpg')
for step in range(8):
    predict = Y_pred.eval(feed_dict={X: [test_images[step]]}, session=sess)
    show_result(test_images[step], predict[0], name=str(step) + '.jpg')
sess.close()
