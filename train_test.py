import tensorflow as tf
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import datetime

from tensorflow.examples.tutorials.mnist import input_data
# from data.batch_data_manage import *
from data_process.batch_data_manage import *

# alias tf_test_utils as ttu
def print_var(var):
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        val = sess.run(var)
        return val


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('his', var)

# test_file_name = '../data/face_5w_data/1w_pic_pickle_test'
# test_file_name = '/home/kevin/data_set/1w_pic_pickle_test'
#
# f_test = open(test_file_name, 'rb')
# record_test = pickle.loads(f_test.read())
#
# data_test = record_test['data']
# label_test = record_test['lable']
# data_test = np.reshape(data_test, [-1, 24*24*3])
# mean = np.mean(data_test, axis=1).reshape(-1, 1)
# std = np.std(data_test, axis=1).reshape(-1, 1)
# data_test = (data_test - mean)/(std + 0.0001)
# data_test = np.reshape(data_test, [-1, 24, 24, 3])
#
# test_img_path = '/home/kevin/data_set/hand_negative_img'
# img_test = open(test_img_path, 'rb')
# img_record = pickle.loads(img_test.read())
#
# img_data = img_record['data']
# img_data = np.reshape(img_data, [-1, 24*24*3])
# mean = np.mean(img_data, axis=1).reshape(-1, 1)
# std = np.std(img_data, axis=1).reshape(-1, 1)
# img_data = (img_data - mean)/(std + 0.0001)
# img_data = np.reshape(img_data, [-1, 24, 24, 3])


# pic = data[62]
#
# pic = pic.reshape(24, 24, 3)
#
# img = pic[:, :, (2, 1, 0)]
#
# plt.imshow(img)
# plt.show()
#
# print(data[0])
#
# time.sleep(1000)

# mnist = input_data.read_data_sets('../data_set/mnist/', one_hot=True)


# define the output value dim
label_dim = 3

x = tf.placeholder("float", shape=[None, 36, 36, 3])
y_ = tf.placeholder("float", shape=[None, label_dim])

B1_A = tf.Variable(tf.random_normal([32], stddev=0.01))
B1_B = tf.Variable(tf.random_normal([32], stddev=0.01))

B2_A = tf.Variable(tf.random_normal([64], stddev=0.01))
B2_B = tf.Variable(tf.random_normal([64], stddev=0.01))

B3_A = tf.Variable(tf.random_normal([128], stddev=0.01))
B3_B = tf.Variable(tf.random_normal([128], stddev=0.01))

B4 = tf.Variable(tf.random_normal([label_dim], stddev=0.01))

SCAL1_A = tf.Variable(tf.random_normal([32], mean=0.01, stddev=0.01))
SCAL1_B = tf.Variable(tf.random_normal([32], mean=0.01, stddev=0.01))

SCAL2_A = tf.Variable(tf.random_normal([64], mean=0.01, stddev=0.01))
SCAL2_B = tf.Variable(tf.random_normal([64], mean=0.01, stddev=0.01))

SCAL3_A = tf.Variable(tf.random_normal([128], mean=0.01, stddev=0.01))
SCAL3_B = tf.Variable(tf.random_normal([128], mean=0.01, stddev=0.01))

SCAL4 = tf.Variable(tf.random_normal([label_dim], mean=0.01, stddev=0.01))

test_flag = tf.placeholder(dtype=tf.bool)

keep_prob = tf.placeholder("float")

W = tf.Variable(tf.zeros([36*36*3, 1]))
b = tf.Variable(tf.zeros([1]))

# W_fc2 = weight_variable([1024, 1])
# b_fc2 = bias_variable([1])


W_conv1_A = weight_variable([3, 3, 3, 32])
b_conv1_A = bias_variable([32])

W_conv1_B = weight_variable([3, 3, 32, 32])
b_conv1_B = bias_variable([32])

W_conv2_A = weight_variable([3, 3, 32, 64])
b_conv2_A = bias_variable([64])

W_conv2_B = weight_variable([3, 3, 64, 64])
b_conv2_B = bias_variable([64])


W_conv3_A = weight_variable([3, 3, 64, 128])
b_conv3_A = bias_variable([128])

W_conv3_B = weight_variable([3, 3, 128, 128])
b_conv3_B = bias_variable([128])

# vgg16_npy_path = './vgg16.npy'
#
# data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

# W_conv1_A = tf.Variable(data_dict['conv1_1'][0])
# b_conv1_A = tf.Variable(data_dict['conv1_1'][1])
#
# W_conv1_B = tf.Variable(data_dict['conv1_2'][0])
# b_conv1_B = tf.Variable(data_dict['conv1_2'][1])
#
# W_conv2_A = tf.Variable(data_dict['conv2_1'][0])
# b_conv2_A = tf.Variable(data_dict['conv2_1'][1])
#
# W_conv2_B = tf.Variable(data_dict['conv2_2'][0])
# b_conv2_B = tf.Variable(data_dict['conv2_2'][1])
#
# W_conv3_A = tf.Variable(data_dict['conv3_1'][0])
# b_conv3_A = tf.Variable(data_dict['conv3_1'][1])
#
# W_conv3_B = tf.Variable(data_dict['conv3_2'][0])
# b_conv3_B = tf.Variable(data_dict['conv3_2'][1])
# tf.Variable
#
# layer1 = data_dict['conv1_2']
#
# print(layer1[0].shape)
# print(layer1[1].shape)


W_conv4_A = weight_variable([5, 5, 128, 512])
W_conv4_B = weight_variable([1, 1, 512, 1024])
W_conv4_C = weight_variable([1, 1, 1024, label_dim])

# W_fc1 = weight_variable([3*3*256, 1024])
# b_fc1 = bias_variable([1024])

x_image = tf.reshape(x, [-1, 36, 36, 3])

bnepsilon = 1e-5

iter1 = tf.Variable(1, dtype=tf.float32)

iter_update = tf.assign(iter1, iter1 + 1.0)

with tf.control_dependencies([iter_update]):
    t1 = tf.nn.conv2d(x_image, W_conv1_A, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_A

with tf.name_scope('iter1'):
    variable_summaries(iter1)
m1, v1 = tf.nn.moments(t1, [0, 1, 2])
t1bn_A = tf.nn.batch_normalization(t1, m1, v1, B1_A, SCAL1_A, variance_epsilon=bnepsilon)

# t1bn_A = tf.nn.relu(t1bn_A)

t1bn = tf.nn.conv2d(t1bn_A, W_conv1_B, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_B
m1_B, v1_B = tf.nn.moments(t1bn, [0, 1, 2])
t1bn = tf.nn.batch_normalization(t1bn, m1_B, v1_B, B1_B, SCAL1_B, variance_epsilon=bnepsilon)

t1bn = tf.nn.relu(t1bn)

t1bn = max_pool_2x2(t1bn)


# with tf.name_scope('l1_out_put'):
#     variable_summaries(t1bn)
# with tf.name_scope('w_conv1'):
#     variable_summaries(W_conv1)
# with tf.name_scope('offset1'):
#     variable_summaries(B1)
# with tf.name_scope('scal1'):
#     variable_summaries(SCAL1)

t2 = tf.nn.conv2d(t1bn, W_conv2_A, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_A
m2, v2 = tf.nn.moments(t2, [0, 1, 2])
t2 = tf.nn.batch_normalization(t2, m2, v2, B2_A, SCAL2_A, variance_epsilon=bnepsilon)

# t2 = tf.nn.relu(t2)

t2bn = tf.nn.conv2d(t2, W_conv2_B, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_B
# m2_B, v2_B = tf.nn.moments(t2bn_B, [0, 1, 2])
# t2bn = tf.nn.batch_normalization(t2bn_B, m2_B, v2_B, B2_B, SCAL2_B, variance_epsilon=bnepsilon)

t2bn = tf.nn.relu(t2bn)

t2bn = max_pool_2x2(t2bn)

# with tf.name_scope('l2_out_put'):
#     variable_summaries(t2bn)
# with tf.name_scope('w_conv2'):
#     variable_summaries(W_conv2)
# with tf.name_scope('offset2'):
#     variable_summaries(B2)
# with tf.name_scope('scal2'):
#     variable_summaries(SCAL2)

t3_A = tf.nn.conv2d(t2bn, W_conv3_A, strides=[1, 1, 1, 1], padding='SAME') + b_conv3_A
m3_A, v3_A = tf.nn.moments(t3_A, [0, 1, 2])
t3_A = tf.nn.batch_normalization(t3_A, m3_A, v3_A, B3_A, SCAL3_A, variance_epsilon=bnepsilon)

# t3_A = tf.nn.relu(t3_A)

t3_B = tf.nn.conv2d(t3_A, W_conv3_B, strides=[1, 1, 1, 1], padding='SAME') + b_conv3_B
m3_B, v3_B = tf.nn.moments(t3_B, [0, 1, 2])
t3_B = tf.nn.batch_normalization(t3_B, m3_B, v3_B, B3_B, SCAL3_B, variance_epsilon=bnepsilon)

t3bn_B = tf.nn.relu(t3_B)

# h_pool3 = max_pool_2x2(t3bn)

t3bn = max_pool_2x2(t3bn_B)

# with tf.name_scope('l1_out_put'):
#     variable_summaries(t3bn)
# with tf.name_scope('w_conv3'):
#     variable_summaries(W_conv3)
# with tf.name_scope('offset3'):
#     variable_summaries(B3)
# with tf.name_scope('scal3'):
#     variable_summaries(SCAL3)


keep_prob = tf.placeholder("float")

t3bn = tf.nn.dropout(t3bn, keep_prob)

t4_A = tf.nn.conv2d(t3bn, W_conv4_A, strides=[1, 1, 1, 1], padding='VALID')

# t4_A = tf.nn.relu(t4_A)

t4 = tf.nn.conv2d(t4_A, W_conv4_B, strides=[1, 1, 1, 1], padding='VALID')


t4 = tf.nn.conv2d(t4, W_conv4_C, strides=[1, 1, 1, 1], padding='VALID')
# m4, v4 = tf.nn.moments(t4, [0, 1, 2])
#
# t4bn = tf.nn.batch_normalization(t4, m4, v4, B4, SCAL4, variance_epsilon=bnepsilon)

# h_pool2_flat = tf.reshape(t4bn, [-1, 10])
# h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

# t4bn = tf.nn.relu(t4bn)

# t4bn = tf.nn.relu(t4)

y_conv = tf.reshape(t4, [-1, label_dim])

# s

# y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# y_conv = tf.nn.softmax(final)

# sess = tf.InteractiveSession()

# rand = tf.random_normal([100, 6], mean=0.0, stddev=2.0)
# # rand = tf.random_normal([100, 10], mean=0.0, stddev=1.0)
# rand = tf.abs(rand)
# rand = tf.minimum(1.0, rand)
# rand = tf.floor(rand)
# rand = tf.random_shuffle(rand)

y_conv = tf.nn.softmax(y_conv)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# cross_entropy = -tf.reduce_sum(tf.log(1 - tf.abs(y_ - y_conv)))

# loss_weight = tf.constant([1.0, 0.0, 0.0, 0.0, 0.0])

# cross_entropy = tf.reduce_sum(tf.nn.l2_loss(tf.multiply(y_conv - y_, loss_weight)))

train_step = tf.train.AdamOptimizer(0.00005).minimize(cross_entropy)

# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

check_point_path = './check_point/'

i = 0


def my_shuffle(data, label):
    data = np.reshape(data, [-1, 36 * 36 * 3])
    c = np.hstack((data, label))

    np.random.shuffle(c)

    data = c[:, 0:36*36*3]
    label = c[:, 36*36*3:]

    data = np.reshape(data, [-1, 36, 36, 3])

    return data, label

train_num = 75000
test_num = 18000
batch_size = 100

def get_random_idx(count, max_num):
    rand_idx = np.array([], dtype=np.int32)
    for _ in range(count):
        rand_idx = np.append(rand_idx, np.random.randint(0, max_num, 1)[0])

    return rand_idx

p = ['/home/kevin/data_set/pickle_img_data/train/face_pickle',
    '/home/kevin/data_set/pickle_img_data/train/beijin_pickle',
     '/home/kevin/data_set/pickle_img_data/train/wrj_pickle']


bdm = BatchDataManage(batch_num=50, file_path=p)
# bdm = BatchDataManage(batch_num=50, file_path='../data/face_0_1_shuffle')

bdm_test = BatchDataManage(batch_num=50, reload_flag=False, file_path=['/home/kevin/data_set/pickle_img_data/test/face_pickle', '/home/kevin/data_set/pickle_img_data/test/beijin_pickle', '/home/kevin/data_set/pickle_img_data/test/wrj_pickle'])

bdm_my = GetBatchData(batch_num=6, reload_flag=False, file_path=['/home/kevin/data_set/pickle_img_data/valid/'])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./summary/', sess.graph)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver(tf.global_variables())

    module_file = tf.train.latest_checkpoint(check_point_path)
    saver.restore(sess, './check_point/pic_check-2000')

    batch_data, batch_label = bdm_my.get_batch_data_label()

    y_conv_test = y_conv.eval(feed_dict={x: batch_data, y_: batch_label, test_flag: True, keep_prob: 1.0})
    print(y_conv_test)
    print(batch_label)

    for img in batch_data:
        print(img.shape)
        show(img)

    time.sleep(1000)

    for i in range(100000):
        # if i < 10000:
        #     idx_a = int(i/1000)
        # else:
        #     idx_a = int(i/1000)
        #
        # idx_b = idx_a%10
        # t_name = 'dataA'

        idx_b = 0

        s_time = datetime.datetime.now()

        # if idx_b == 0:
        # rand_idx = np.random.randint(0, train_num, 1)[0]

        batch_data, batch_label = bdm.get_batch_data_label()

        # batch_data = data[61:63, :, :, :]
        # batch_label = label[61:63, :]
        # y_conv_test = y_conv.eval(feed_dict={x: batch_data, y_: batch_label, test_flag: True, keep_prob: 1.0})
        # print(y_conv_test)

        # batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            # data, label = my_shuffle(data, label)
            # y_conv_test = y_conv.eval(feed_dict={x: batch_data, y_: batch_label, test_flag: True, keep_prob: 1.0})
            # print(y_conv_test)
            print(batch_data.shape)
            print(batch_label.shape)

            train_accuracy = cross_entropy.eval(feed_dict={x: batch_data, y_: batch_label, test_flag: True, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            # summary = sess.run(merged, feed_dict={x: batch_data, y_: batch_label, test_flag: True, keep_prob: 1.0})
            # train_writer.add_summary(summary, i)

        if i % 200 == 0:
            # data_test, label_test = my_shuffle(data_test, label_test)
            # test_idx = np.random.randint(0, test_num, 1)[0]

            rand_idx = get_random_idx(batch_size, 9000)

            # t_data = data_test[rand_idx, :, :, :]
            # t_label = label_test[rand_idx, :]

            t_data, t_label = bdm_test.get_batch_data_label()

            test_accuracy = cross_entropy.eval(feed_dict={x: t_data, y_: t_label, test_flag: True, keep_prob: 1.0})
            print("step %d, test accuracy %g" % (i, test_accuracy))

        sess.run(train_step, feed_dict={x: batch_data, y_: batch_label, keep_prob: 0.8})
        if i % 2000 == 0:
            saver.save(sess, check_point_path + 'pic_check', i)
        # print('train run')
        # sess.run(update_ema, {x: batch[0], y_: batch[1], keep_prob: 1.0, tst: False, iter: i})

        e_time = datetime.datetime.now()

        use_time = (e_time - s_time)

        # print(use_time)

        # print('-----:' + str(i) + '---' + str(use_time))

    saver.save(sess, check_point_path + 'pic_check', i)


# test_batch = mnist.test.next_batch(2000)

# print(test_batch[0].shape)
# print(test_batch[1].shape)
#
# print("test accuracy %g" % accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], test_flag:False,
#                                                     keep_prob: 1.0}))

# print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, test_flag:False,
#                                                     keep_prob: 1.0}))


# if __name__ == '__main__':
    # data = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    # var = tf.Variable(data, dtype=tf.float32)
    # data = tf.linspace(1.0, 10.0, 5)

    # n_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # data = tf.constant(n_data, dtype=tf.float32)
    # # print(n_data.shape)
    #
    # # ret = tf.layers.batch_normalization(data, axis=0)
    # ret = tf.reduce_max(data, axis=1)
    #
    # # ret = tf.random_uniform([10], minval=0.0, maxval=1.0)
    #
    # a = tf.constant([1, 0, 0, 0, 0,], dtype=tf.float32)
    # ret = tf.random_shuffle(a)

    # a = tf.ones([2, 5], dtype=tf.float32)
    # b = tf.zeros([2, 5], dtype=tf.float32)
    #
    # c = tf.concat([a, b], 1)

    # c = tf.random_normal([10, 10], mean=0.0, stddev=10.0)

    # c =tf.random_uniform([5, 5], minval=0.0, maxval=1.0)

    # c =tf.truncated_normal([100], mean=0.0, stddev=0.5)

    # ret = tf.nn.l2_normalize(data, dim=[1])

    # ret = tf.random_shuffle(c)

    # ret = tf.abs(c)
    # ret = tf.minimum(ret, 1)
    # ret = tf.floor(ret)

    # a = tf.ones([9])
    # b = tf.zeros([1])
    # c = tf.concat([a, b], axis=0)
    # d = tf.random_shuffle(c)
    #
    # for i in range(8):
    #     t1 = tf.ones([9])
    #     t2 = tf.zeros([1])
    #     t3 = tf.concat([t1, t2], axis=0)
    #     t4 = tf.random_shuffle(t3)
    #     d = tf.concat([d, t4], axis=0)
    #
    #
    # print_var(d)

    # a = np.array([[0, 1, 1, 1, 1, 1, 1]])

    # a = np.array([[0, 1, 1, 1, 1, 1, 1]])
    # b = a.copy()
    # for i in range(9):
    #     np.random.shuffle(b[0])
    #     a = np.append(a, b, axis=0)
    #
    # print(a)

    # a = np.ones([1, 10])
    # a[0][0] = 0
    # np.random.shuffle(a[0])
    # b = a.copy()
    #
    # for i in range(9):
    #     np.random.shuffle(b[0])
    #     a = np.append(a, b, axis=0)
    #
    # print(a)

    # rand = tf.random_normal([100, 10], mean=0.0, stddev=2.0)
    # rand = tf.abs(rand)
    # rand = tf.minimum(1.0, rand)
    # rand = tf.floor(rand)
    # rand = tf.random_shuffle(rand)
    #
    # print(print_var(rand))

    # a = tf.ones([10, 7, 7, 32])
    #
    # filter = tf.constant(shape=[7, 7, 32, 10], value=0.25)
    #
    # b = tf.nn.conv2d(a, filter, strides=[1, 1, 1, 1], padding='SAME')
    #
    # c = print_var(b)
    # print(c.shape)

    # global_step = tf.Variable(0, trainable=False)
    #
    # initial_learning_rate = 0.1  # 初始学习率
    #
    # learning_rate = tf.train.exponential_decay(initial_learning_rate,
    #                                            global_step=global_step,
    #                                            decay_steps=10, decay_rate=0.9)
    # opt = tf.train.GradientDescentOptimizer(learning_rate)
    #
    # add_global = global_step.assign_add(1)
    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     print(sess.run(learning_rate))
    #     for i in range(10):
    #         _, rate = sess.run([add_global, learning_rate])
    #         print(rate)


    # a = tf.constant([[1 , 2, 5], [2, 3, 4]], dtype=tf.float32)

    # a = tf.constant(11, dtype=tf.int32)
    # b = tf.constant(3, dtype=tf.int32)
    #
    # print(print_var(a%b))

    # # print(print_var(tf.nn.softmax(a)))
    #
    # v1 = tf.Variable(0, dtype=tf.float32)  # 定义一个变量，初始值为0
    # step = tf.Variable(0, trainable=False)  # step为迭代轮数变量，控制衰减率
    #
    # ema = tf.train.ExponentialMovingAverage(0.99, step)  # 初始设定衰减率为0.99
    # maintain_averages_op = ema.apply([v1])  # 更新列表中的变量
    #
    # with tf.Session() as sess:
    #     init_op = tf.global_variables_initializer()  # 初始化所有变量
    # sess.run(init_op)
    #
    # print(sess.run([v1, ema.average(v1)]))  # 输出初始化后变量v1的值和v1的滑动平均值
    #
    # sess.run(tf.assign(v1, 5))  # 更新v1的值
    # sess.run(maintain_averages_op)  # 更新v1的滑动平均值
    # print(sess.run([v1, ema.average(v1)]))
    #
    # sess.run(tf.assign(step, 1))  # 更新迭代轮转数step
    # sess.run(tf.assign(v1, 10))
    # sess.run(maintain_averages_op)
    # print(sess.run([v1, ema.average(v1)]))
    # # 再次更新滑动平均值，
    # sess.run(maintain_averages_op)
    # print(sess.run([v1, ema.average(v1)]))
    # # 更新v1的值为15
    # sess.run(tf.assign(v1, 15))
    #
    # sess.run(maintain_averages_op)
    # print(sess.run([v1, ema.average(v1)]))

    # tf.nn.batch_normalization()

    # img = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]]), dtype=tf.float32)
    # axis = list(range(len(img.get_shape()) - 1))
    # mean, variance = tf.nn.moments(img, [0, 1])
    #
    # print(print_var([mean, variance]))
