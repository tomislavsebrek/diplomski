from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from database import *
from statistics import *

import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import math


class NeuralNetwork:

    def __init__(self, database, data_type=tf.float32, labeled_output=True):
        self.database = database
        self.data_type = data_type
        self.labeled_output = labeled_output

        self.sess = tf.Session()
        self.saver = None
        self.eps = 0.0001
        self.analyzer = Analyzer(database.num_classes)

        self.x = tf.placeholder(data_type, self.database.get_endpoint_shapes()[0])
        if labeled_output:
            self.y_true = tf.placeholder(data_type, self.database.get_endpoint_shapes()[1])
        self.is_train = tf.placeholder(tf.bool, [])


    def get_input_tensor(self):
        return tf.reshape(self.x, [-1, self.database.input_length, 1, 1])

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

    def add_1d_convolution_layer(self, net, filter_size, num_filters, activation=tf.nn.relu, stride=1, padding='SAME', name="conv", reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            if len(net.get_shape()) != 4:
                net = tf.reshape(net, [-1, self.database.input_length, 1, 1])
            depth = net.get_shape()[3]

            w = tf.get_variable("w", initializer=tf.truncated_normal([filter_size, 1, int(depth), num_filters], stddev=0.1))
            return activation(tf.nn.conv2d(net, w, [1, stride, 1, 1], padding))

    def add_max_pool_layer(self, net, delta=2, name="max_pool", reuse=False):
        with tf.variable_scope(name) as scope:
            return tf.nn.max_pool(net, ksize=[1, delta, 1, 1], strides=[1, delta, 1, 1], padding='SAME')

    def add_max_pool_transpose_layer(self, net, delta=2, name="max_pool_transpose", reuse=False):
        with tf.variable_scope(name) as scope:
            value = net
            sh = value.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(value, [-1] + sh[-dim:]))
            out = tf.concat(1, [out, tf.zeros_like(out)])
            out_size = [-1, sh[1]*delta, 1, sh[-1]]
            return tf.reshape(out, out_size)

    def add_1d_convolution_transpose_layer(self, net, filter_size, num_filters, activation=tf.nn.relu, stride=1, padding='SAME', name="conv_transpose", reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            size = tf.shape(net)[0]
            height = int(net.get_shape()[1])
            depth = int(net.get_shape()[-1])
            net = tf.reshape(net, [size, height, 1, -1])

            output_shape = tf.pack([size, height, 1, num_filters])
            filter_shape = tf.pack([filter_size, 1, num_filters, depth])

            w = tf.get_variable("w", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
            result = activation(tf.nn.conv2d_transpose(net, w, output_shape, [1, stride, 1, 1], padding))
            return tf.reshape(result, [-1, height, 1, num_filters])

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

    def add_reparametrization_sample_layer(self, net, output_size, get_parts=True, name="reparam", reuse=False, sample=True):
        x_mean = self.add_fully_connected_layer(net, output_size, activation=tf.identity, name=name+'_mean', reuse=reuse)
        x_log_sigma_sq = self.add_fully_connected_layer(net, output_size, activation=tf.identity, name=name+'_sigma', reuse=reuse)

        x_log_sigma_sq = tf.minimum(x_log_sigma_sq, 10)
        x_sigma = tf.sqrt(tf.exp(x_log_sigma_sq))

        eps = 0
        if sample:
            eps = tf.random_normal((tf.shape(net)[0], output_size), 0, 1, dtype=tf.float32)

        result = x_mean + tf.mul(x_sigma, eps)

        if get_parts:
            return result, x_mean, x_sigma
        return result

    def add_output_layer(self, net, name="output", reuse=False):
        return self.add_fully_connected_layer(net, self.database.num_classes, tf.nn.softmax, name=name, reuse=reuse)

    def load_variables(self, load_location):
        if self.saver is None:
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, load_location)

    def save(self, save_location):
        self.saver.save(self.sess, save_location)

    def get_feed_dict(self, phase, use_labeled=True, use_generated=False):
        is_train = False
        if phase == 'TRAIN':
            current_examples, current_labels, current_names = self.database.get_next_batch(use_labeled=use_labeled,
                                                                                           use_generated=use_generated)
            is_train = True
        elif phase == 'TEST':
            current_examples, current_labels, current_names = self.database.get_test_set()

        feed_dict = {
            self.x: current_examples,
            self.y_true: current_labels,
            self.is_train: is_train
        }
        return feed_dict, current_examples, current_labels, current_names

    def train(self, load_location=None, periodic_save=100, evaluate_every=1000, max_iter=50000, save_location="save/checkpoint.ckpt"):
        if load_location:
            self.load_variables(load_location)
        else:
            self.sess.run(tf.initialize_all_variables())

        if self.saver is None:
            self.saver = tf.train.Saver()

        step = 0
        while step < max_iter:
            if step > 0 and step % evaluate_every == 0:
                self._test_step(step)

            if step > 0 and periodic_save > 0 and step % periodic_save == 0:
                self.saver.save(self.sess, save_location)

            self._train_step(step)
            step += 1

    def close(self):
        self.sess.close()

    def create_model(self):
        pass
