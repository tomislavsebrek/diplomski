from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from database import *
from neural_network import *
from autoencoder import *

import tensorflow as tf
import math

from config import CONFIG


class CAE(Autoencoder):

    def __init__(self, database, latent_size=10, data_type=tf.float32):
        super().__init__(database, latent_size, data_type)

    def _create_encoder_part(self, x):
        with tf.variable_scope("encoder"):
            net = self.add_1d_convolution_layer(x, 5, 16, name="conv1")
            net = self.add_max_pool_layer(net, 2, name="max1")
            net = self.add_1d_convolution_layer(net, 3, 32, name="conv2")
            net = self.add_max_pool_layer(net, 2, name="max2")
            net = self.add_1d_convolution_layer(net, 3, 64, name="conv3")
            return self.add_fully_connected_layer(net, self.latent_size, name="latent")

    def _create_decoder_part(self, z):
        with tf.variable_scope("decoder"):
            net = self.add_fully_connected_layer(z, 16 * self.database.input_length, name="fc1")
            net = tf.reshape(net, [-1, int(self.database.input_length/4), 1, 64])
            net = self.add_1d_convolution_transpose_layer(net, 3, 32, name="deconv1")
            net = self.add_max_pool_transpose_layer(net, 2, name="max_transp1")
            net = self.add_1d_convolution_transpose_layer(net, 5, 16, name="deconv2")
            net = self.add_max_pool_transpose_layer(net, 2, name="max_transp2")
            net = self.add_1d_convolution_transpose_layer(net, 5, 1, name="deconv3")
            x_reconstruct = tf.reshape(net, [-1, self.database.input_length])

            self.add_loss_part(tf.reduce_mean(tf.square(self.x - x_reconstruct)))

            return x_reconstruct

    def _create_supervised_head(self, z):
        with tf.variable_scope("classification"):
            net = self.add_fully_connected_layer(z, 50, name="fc1")
            net = self.add_fully_connected_layer(net, 50, name="fc2")
            y_out = self.add_output_layer(net, name="output")

            loss = -tf.reduce_sum(self.y_true * tf.log(tf.maximum(y_out, 1e-15)), reduction_indices=[1])
            loss = tf.reduce_mean(loss * tf.reduce_sum(self.y_true, reduction_indices=[1]))
            self.add_loss_part(loss, 1)

            return y_out