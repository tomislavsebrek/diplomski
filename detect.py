import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np


def interpolate(array, sequence_length):
    xp = np.linspace(0, 1, num=len(array))
    new_x = np.linspace(0, 1, num=sequence_length)
    return np.interp(new_x, xp, array)


def normalize(array):
    return array / np.max(array, axis=1, keepdims=True)


def process_mhap_input(str):
    tokens = str.split(" ")
    result = {}
    keys = ['id1', 'id2', 'error', 'minmears', 'att1', 'start1', 'end1', 'len1', 'att2', 'start2', 'end2', 'len2']
    for index, token in enumerate(tokens):
        if index == 2:
            result[keys[index]] = float(tokens[index])
        else:
            result[keys[index]] = int(tokens[index])
    return result


class Detector:

    def __init__(self, load_location):
        self.sess = tf.Session()
        self.saver = None
        self.eps = 0.0001
        self.num_classes = 4
        self.input_length = 100
        self.data_type = tf.float32

        self.x = tf.placeholder(self.data_type, [None, self.input_length])
        self.is_train = tf.placeholder(tf.bool, [])

        self.create_model()
        self.sess = tf.Session()
        self.load_variables(load_location)

    def get_input_tensor(self):
        return tf.reshape(self.x, [-1, self.input_length, 1, 1])

    def add_fully_connected_layer(self, net, output_size, activation=tf.nn.relu, stddev=None, name="fc", reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            input_size = 1
            for i in range(1, len(net.get_shape())):
                input_size *= int(net.get_shape()[i])

            net = tf.reshape(net, [-1, input_size])

            if stddev is None:
                w = tf.get_variable("w", shape=[input_size, output_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
            else:
                w = tf.get_variable("w", initializer=tf.truncated_normal([input_size, output_size], stddev=stddev))

            b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[output_size]))
            return activation(tf.matmul(net, w) + b)

    def add_1d_convolution_layer(self, net, filter_size, num_filters, activation=tf.nn.relu, stride=1, padding='SAME',
                                 name="conv", reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            if len(net.get_shape()) != 4:
                net = tf.reshape(net, [-1, self.input_length, 1, 1])
            depth = net.get_shape()[3]

            w = tf.get_variable("w",
                                initializer=tf.truncated_normal([filter_size, 1, int(depth), num_filters], stddev=0.1))
            return activation(tf.nn.conv2d(net, w, [1, stride, 1, 1], padding))

    def add_batch_norm_layer(self, net, exp_decay=0.5, activation=tf.nn.relu, name="batch_norm", reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            x = net
            n_out = x.get_shape()[-1]

            beta = tf.get_variable("beta", initializer=tf.zeros([n_out]))
            gamma = tf.get_variable("gamma", initializer=tf.ones([n_out]))

            axis = []
            for i in range(len(x.get_shape()) - 1):
                axis.append(i)
            batch_mean, batch_var = tf.nn.moments(x, axis)
            ema = tf.train.ExponentialMovingAverage(exp_decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(self.is_train, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            return activation(tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3))

    def add_max_pool_layer(self, net, delta=2, name="max_pool", reuse=False):
        with tf.variable_scope(name) as scope:
            return tf.nn.max_pool(net, ksize=[1, delta, 1, 1], strides=[1, delta, 1, 1], padding='SAME')

    def create_model(self):
        with tf.variable_scope("discriminator"):

            net = self.add_1d_convolution_layer(self.get_input_tensor(), 5, 16, name="conv1")
            net = self.add_max_pool_layer(net, name="pool1")
            net = self.add_1d_convolution_layer(net, 3, 32, name="conv3")
            net = self.add_max_pool_layer(net, name="pool1")
            net = self.add_fully_connected_layer(net, 256, name="fc1")
            net = self.add_fully_connected_layer(net, 1024, name="fc2")
            self.output = self.add_fully_connected_layer(net, self.num_classes+1, activation=tf.nn.softmax, name="fc3")

    def load_variables(self, load_location):
        if self.saver is None:
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, load_location)

    def get_output_class(self, x):
        x = interpolate(x, self.input_length)
        x = x.reshape([1, x.shape[0]])
        x = normalize(x)

        output = self.sess.run([self.output], feed_dict={
            self.x: x,
            self.is_train: False
        })[0]

        y = np.argmax(output, axis=1)
        return y[0]

if __name__ == "__main__":
    detector = Detector("save/checkpoint.ckpt")
    #reads_file = input()    # .FASTQ fromat
    #overlaps_file = input()     # .MHAP format

    # initialize structures for all reads

    coverage = {}
    labels = {}
    total_reads = 0

    with open(reads_file, "r") as ins:
        for index, line in enumerate(ins):
            line = line.rstrip()
            if index % 4 == 0:
                read_id = int(line[1:])
                coverage[read_id] = []
                labels[read_id] = 0
                total_reads += 1

    # processing all overlaps

    with open(overlaps_file, "r") as ins:
        for line in ins:
            mhap_input = process_mhap_input(line.rstrip())

            for index in ['1', '2']:
                coverage[mhap_input['id' + index]].append((mhap_input['start' + index], mhap_input['end' + index]))

    # generate coverage

    with open(reads_file, "r") as ins:
        for index, line in enumerate(ins):
            line = line.rstrip()
            if index % 4 == 0:
                read_id = int(line[1:])

                if read_id % 100 == 0:
                    print("processed:" + str(read_id) + "/" + str(total_reads))

                if len(coverage[read_id]) == 0:
                    empty = True
                else:
                    last_coverage = coverage[read_id]
                    empty = False

            if index % 4 == 1 and not empty:
                y = np.zeros(len(line))

                for interval in last_coverage:
                    for i in range(interval[0], interval[1]):
                        y[i] += 1

                if np.max(y) < 4:
                    continue
                if y.shape[0] < 100:
                    continue

                out = detector.get_output_class(y)
                if out == 0:
                    print(out)
                    plt.plot(y)
                    plt.show()