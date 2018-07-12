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


# Serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


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


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


def show_result(x, y, name):
    img = x.reshape(96, 96) * 255.
    img = np.clip(img, 0, 255).astype('uint8')
    for step in range(0, 30, 2):
        cv2.circle(img, center=(int(np.round(y[step] * 96)), int(np.round(y[step + 1] * 96))), radius=1,
                   color=(255, 0, 0))
    img = cv2.resize(img, dsize=(512, 512))
    cv2.imwrite(name, img)


LEARNING_RATE = 0.001
TRAINING_EPOCHS = 20000
BATCH_SIZE = 60
DISPLAY_STEP = 10
DROPOUT_CONV = 0.8
DROPOUT_HIDDEN = 0.8
VALIDATION_SIZE = 10

image_size = 9216
image_width = image_height = 96
image_labels = 30

train_images, train_labels = load_image_label('training.csv')

# Create Input and Output
X = tf.placeholder('float', shape=[None, image_width * image_height])  # mnist data image of shape 96*96=9216
Y_gt = tf.placeholder('float', shape=[None, image_labels])  # 0-30 digits recognition => 10 classes
Y_pred = _create_conv_net(X=X, image_width=96, image_height=96, image_channel=1, n_class=image_labels)
# Cost function and training
cost = tf.reduce_mean(tf.square(Y_pred - Y_gt))
# train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
'''
TensorFlow Session
'''
# start TensorFlow session
init = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())
tf.summary.scalar('loss', cost)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())
sess = tf.InteractiveSession()
sess.run(init)

DISPLAY_STEP = 1
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]
for i in range(TRAINING_EPOCHS):
    # get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)
    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i % DISPLAY_STEP == 0 or (i + 1) == TRAINING_EPOCHS:
        train_cost = cost.eval(feed_dict={X: batch_xs[0:BATCH_SIZE // VALIDATION_SIZE],
                                          Y_gt: batch_ys[0:BATCH_SIZE // VALIDATION_SIZE]})
        validation_cost = cost.eval(feed_dict={X: batch_xs[BATCH_SIZE // VALIDATION_SIZE:],
                                               Y_gt: batch_ys[BATCH_SIZE // VALIDATION_SIZE:]})
        print('epochs %d batch_train_cost => %.8f' % (i, train_cost))
        print('epochs %d batch_validation_cost => %.8f' % (i, validation_cost))
        # increase DISPLAY_STEP
        if i % (DISPLAY_STEP * 10) == 0 and i:
            DISPLAY_STEP *= 10
    # train on batch
    _, summary = sess.run([train_op, merged_summary_op], feed_dict={X: batch_xs, Y_gt: batch_ys})
    summary_writer.add_summary(summary, i)
save_path = saver.save(sess, 'model\my-model')
print("Model saved in file:", save_path)
sess.close()
