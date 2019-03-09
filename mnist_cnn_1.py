import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import time as time
from scipy import misc
# import mnist_data

batch_size = 256
test_size = 256
img_size = 28
num_classes = 10

def init_weights(shape):
    # tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

# mnist = mnist_data.read_data_sets("ata/")

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

trX, trY, teX, teY = mnist.train.images,\
                     mnist.train.labels,\
                     mnist.test.images,\
                     mnist.test.labels

# -1表示自动计算此维度
trX = trX.reshape(-1, img_size, img_size, 1)  # 28x28x1 input img
teX = teX.reshape(-1, img_size, img_size, 1)  # 28x28x1 input img

# None表示此维不确定
X = tf.placeholder("float", [None, img_size, img_size, 1])
Y = tf.placeholder("float", [None, num_classes])

w = init_weights([5, 5, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, num_classes])# FC 625 inputs, 10 outputs (labels)


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    # ‘SAME’字符串表示不够的列会填充0继续，VALID则直接丢弃
    # strides表示在各个维度上的移动步长，依次为batch，height，width，channels
    conv1 = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
    conv1_a = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1] ,strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)

    conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='SAME')
    conv2_a = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)

    conv3 = tf.nn.conv2d(conv2, w3, strides=[1, 1, 1, 1] ,padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    FC_layer = tf.reshape(conv3, [-1, w4.get_shape().as_list()[0]])    
    FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)

    output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)

    result = tf.matmul(output_layer, w_o)
    return result

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)

# 在张量的某个维度求平均值，没指定维度就是所有维度，这样最终输出是个标量
cost = tf.reduce_mean(Y_)

optimizer  = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

# 返回张量某维中最大值的下标
predict_op = tf.argmax(py_x, axis=1)

with tf.Session() as sess:
    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()

    #range(start, stop, step)返回一个列表
    for i in range(10):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(optimizer, feed_dict={X: trX[start:end],
                                          Y: trY[start:end],
                                          p_keep_conv: 0.8,
                                          p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX))# Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(('step %d, accuracy %f') % (i, 
           np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op,
                          feed_dict={X: teX[test_indices],
                                     Y: teY[test_indices], 
                                     p_keep_conv: 1.0,
                                     p_keep_hidden: 1.0}))))
    # 一张一张地测试模型的预测是否准确
    wrong_cnt = 0
    for c in range(10):
        i = rand.randint(0, len(teX) - 1)
        tex = teX[i].reshape(-1, img_size, img_size, 1)
        tey = teY[i].reshape(-1, 10)
        pre_num = sess.run(predict_op, feed_dict={X: tex,
                                                  Y: tey,
                                                  p_keep_conv: 1.0,
                                                  p_keep_hidden: 1.0})
        print(("count %d, predict number is %d, real is %d") % (c, pre_num, np.argmax(tey, axis=1)))
        tex = tex.reshape(img_size, img_size)
        plt.ion()
        tex = misc.imresize(tex, (1000, 1000))
        plt.imshow(tex, cmap = 'gray')
        plt.pause(1)
        if pre_num != np.argmax(tey, axis=1):
            wrong_cnt = wrong_cnt + 1
            print("predict wrong count %d" % wrong_cnt)
            time.sleep(5)
        plt.close()
print("Total predict wrong count: %d " % wrong_cnt)
