from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from database import *
from neural_network import *
from autoencoder import *
from config import CONFIG

import tensorflow as tf
import math
import numpy as np

class VAE(Autoencoder):

    def __init__(self, database, data_type=tf.float32):
        super().__init__(database, data_type)

    def visualize_2d(self, k=10):
        assert self.latent_size == 2

        fig, subplots = plt.subplots(nrows=k, ncols=k)

        for x, dx in enumerate(list(np.linspace(-3, 3, k))):
            for y, dy in enumerate(list(np.linspace(-3, 3, k))):

                z = np.array([dx, dy]).reshape([1, 2])

                x_reconstructed = self.sess.run(
                    [self.net_gen],
                    feed_dict={self.z: z})

                subplots[y][x].plot(x_reconstructed[0][0])
                subplots[y][x].axis('off')
        plt.show()

    def visualize(self):
        x, y, name = self.database.get_test_set()

        for k in range(x.shape[0]):
            plt.plot(x[k])
            plt.show()

            z_original = self.sess.run(
                [self.z],
                feed_dict={self.x: x[k].reshape([1, 100])})[0]

            z = np.copy(z_original)

            print(z_original)
            print(z)

            fig, subplots = plt.subplots(nrows=10, ncols=self.latent_size)
            for i in range(self.latent_size):
                for index, dz in enumerate(list(np.linspace(-2, 2, 10))):
                    z[0][i] = dz

                    x_reconstructed = self.sess.run(
                        [self.net_gen],
                        feed_dict={self.z: z})

                    subplots[index][i].plot(x_reconstructed[0][0])

                z[0][i] = z_original[0][i]
            plt.show()

    def generate(self, k=10):
        z = np.random.normal(size=(k, CONFIG.LATENT_SIZE))
        x_reconstructed = self.sess.run(
            [self.net_gen],
            feed_dict={self.z: z})[0]

        for i in range(k):
            plt.plot(x_reconstructed[i])
            plt.show()

    def _create_encoder_part(self, x):
        with tf.variable_scope("encoder"):
            net = self.add_1d_convolution_layer(x, 5, 16, name="conv1")
            net = self.add_max_pool_layer(net, 2, name="max1")
            net = self.add_1d_convolution_layer(net, 3, 32, name="conv2")
            net = self.add_max_pool_layer(net, 2, name="max2")
            net = self.add_1d_convolution_layer(net, 3, 64, name="conv3")

            z, x_mean, x_sigma = self.add_reparametrization_sample_layer(net, self.latent_size, True)
            x_log_sigma_sq = tf.log(tf.square(x_sigma))
            loss = -0.5 * tf.reduce_sum(1 + x_log_sigma_sq - tf.exp(x_log_sigma_sq) - tf.square(x_mean), 1)
            self.add_loss_part(loss)

            return z

    def _create_decoder_part(self, z, distribution='GAUSS'):
        with tf.variable_scope("decoder"):
            net = self.add_fully_connected_layer(z, 16 * self.database.input_length, name="fc1")
            net = tf.reshape(net, [-1, int(self.database.input_length/4), 1, 64])
            net = self.add_1d_convolution_transpose_layer(net, 3, 32, name="deconv1")
            net = self.add_max_pool_transpose_layer(net, 2, name="max_transp1")
            net = self.add_1d_convolution_transpose_layer(net, 5, 16, name="deconv2")
            net = self.add_max_pool_transpose_layer(net, 2, name="max_transp2")
            net = self.add_1d_convolution_transpose_layer(net, 5, 1, name="deconv3")

            if distribution == 'GAUSS':
                x_reconstruct, x_mean, x_sigma = self.add_reparametrization_sample_layer(net, self.database.input_length, True, sample=False)
                nom = tf.exp(-(self.x - x_mean) * (self.x - x_mean) / (2 * x_sigma * x_sigma))
                denom = x_sigma * math.sqrt(2 * math.pi) + self.eps
                loss = -tf.reduce_sum(tf.log(self.eps + nom / denom), 1)

            elif distribution == 'BERNOULLI':
                x_reconstruct = self.add_fully_connected_layer(net, self.database.input_length, activation=tf.nn.sigmoid)
                loss = -tf.reduce_sum(self.x * tf.log(1e-9 + x_reconstruct) + (1 - self.x) * tf.log(1e-9 + 1 - x_reconstruct), 1)

            self.add_loss_part(loss)

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

if __name__ == '__main__':
    database = Database(CONFIG.FORMATTED_DATA, CONFIG.BATCH_SIZE, CONFIG.INTERPOLATION_LENGTH, CONFIG.NUM_CLASSES)
    nn = VAE(database, CONFIG.LATENT_SIZE)
    nn.create_model()
    #nn.train(evaluate_every=CONFIG.EVALUATE_EVERY, periodic_save=CONFIG.PERIODIC_SAVE,
    #         save_location=CONFIG.SAVE_LOCATION, max_iter=CONFIG.MAX_ITER)
    nn.load_variables("save/vae_10_500/checkpoint.ckpt")
    nn.generate()

    '''
    nn.load_variables("save/vae_10_500/checkpoint.ckpt")

    x1 = np.load("formatted_data/interpolation_500/test_x.npy")
    z1 = nn.get_latent(x1)
    np.save("formatted_data/interpolation_500_z/test_x.npy", z1)

    x1 = np.load("formatted_data/interpolation_500/labeled_x.npy")
    z1 = nn.get_latent(x1)
    np.save("formatted_data/interpolation_500_z/labeled_x.npy", z1)

    x1 = np.load("formatted_data/interpolation_500/unlabeled_x.npy")
    z1 = nn.get_latent(x1)
    np.save("formatted_data/interpolation_500_z/unlabeled_x.npy", z1)

    y = np.zeros([z1.shape[0], 4])
    for i in range(z1.shape[0]):
        y[i][0] = 1

    nn.visualize_tsne(z1, y, 2)
    #nn._test_step(0)
    '''