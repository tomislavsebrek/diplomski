import tensorflow as tf
import numpy as np

from neural_network import *
from config import CONFIG
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

def leakyRelu(x):
    f1 = 0.5 * (1 + 0.01)
    f2 = 0.5 * (1 - 0.01)
    return f1 * x + f2 * tf.abs(x)

class GAN(NeuralNetwork):

    def __init__(self, database, latent_size, data_type=tf.float32, bidirectional=False, minibatch=True):
        super().__init__(database, data_type)

        self.latent_size = latent_size
        self.bidirectional = bidirectional
        self.minibatch = minibatch

    def _generator(self, z, reuse=False):
        with tf.variable_scope("opponent"):
            with tf.variable_scope("generator") as scope:
                if reuse:
                    scope.reuse_variables()

                net = self.add_fully_connected_layer(z, 32 * self.database.input_length, name="fc1", activation=leakyRelu)
                net = self.add_batch_norm_layer(net, name="bn1")
                net = tf.reshape(net, [-1, int(self.database.input_length), 1, 32])
                net = self.add_1d_convolution_transpose_layer(net, 3, 32, name="deconv1", activation=leakyRelu)
                net = self.add_batch_norm_layer(net, name="bn2")
                net = self.add_1d_convolution_transpose_layer(net, 5, 16, name="deconv2", activation=leakyRelu)
                net = self.add_batch_norm_layer(net, name="bn3")
                net = self.add_1d_convolution_transpose_layer(net, 5, 1, name="deconv3", activation=leakyRelu)
                net = self.add_fully_connected_layer(net, 256, name="fc2", activation=leakyRelu)
                net = self.add_fully_connected_layer(net, 256, name="fc3", activation=leakyRelu)
                return self.add_fully_connected_layer(net, self.database.input_length, name="fc4", activation=tf.nn.sigmoid)

    def _discriminator(self, x, return_latent=False, reuse=False, minibatch=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            net = self.add_1d_convolution_layer(x, 5, 16, name="conv1")
            net = self.add_1d_convolution_layer(net, 3, 32, name="conv2")
            net = self.add_1d_convolution_layer(net, 3, 32, name="conv3")
            net = self.add_fully_connected_layer(net, 256, name="fc1")
            net = self.add_fully_connected_layer(net, self.latent_size, name="fc2")
            o = self.add_fully_connected_layer(net, 1, activation=tf.nn.sigmoid, name="fc3")

            if return_latent:
                return o, net
            return o

    def get_minibatch_features(self, net, num_kernels = 5, kernel_dim = 3):
        x = net
        net = self.add_fully_connected_layer(net, num_kernels * kernel_dim, stddev=0.02, name="minibatch")
        activation = tf.reshape(net, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat(1, [x, minibatch_features])

    def _double_discriminator(self, x, z, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            net = self.add_1d_convolution_layer(x, 5, 16, name="conv1", activation=leakyRelu)
            net = self.add_1d_convolution_layer(net, 3, 32, name="conv2", activation=leakyRelu)
            net = self.add_1d_convolution_layer(net, 3, 32, name="conv3", activation=leakyRelu)
            net = tf.reshape(net, [-1, 32*self.database.input_length])
            features_x = self.add_fully_connected_layer(net, 128, name="fc0", activation=leakyRelu)

            net = self.add_fully_connected_layer(z, 128, name="fc1", activation=leakyRelu)
            features_z = self.add_fully_connected_layer(net, 128, name="fc2", activation=leakyRelu)

            if self.minibatch:
                with tf.variable_scope("mb_1"):
                    features_x = self.get_minibatch_features(features_x)
                with tf.variable_scope("mb_2"):
                    features_z = self.get_minibatch_features(features_z)
            concat = tf.concat(1, [features_x, features_z])

            net = self.add_fully_connected_layer(concat, 1024, name="fc3", activation=leakyRelu)
            net = self.add_fully_connected_layer(net, 1024, name="fc4", activation=leakyRelu)
            return self.add_fully_connected_layer(net, 1, activation=tf.nn.sigmoid, name="fc5")

    def _encoder(self, x, reuse=False):
        with tf.variable_scope("opponent"):
            with tf.variable_scope("encoder") as scope:
                if reuse:
                    scope.reuse_variables()

                net = self.add_1d_convolution_layer(x, 5, 16, name="conv1", activation=leakyRelu)
                net = self.add_batch_norm_layer(net, name="bn1")
                net = self.add_1d_convolution_layer(net, 3, 32, name="conv2", activation=leakyRelu)
                net = self.add_batch_norm_layer(net, name="bn2")
                net = self.add_1d_convolution_layer(net, 3, 32, name="conv3", activation=leakyRelu)
                net = self.add_batch_norm_layer(net, name="bn3")
                net = self.add_fully_connected_layer(net, 256, name="fc1", activation=leakyRelu)
                net = self.add_batch_norm_layer(net, name="bn4")
                net = self.add_fully_connected_layer(net, 1024, name="fc2", activation=leakyRelu)
                return self.add_reparametrization_sample_layer(net, self.latent_size, name="fc2", get_parts=False)

    def _supervised_head(self, z):
        with tf.variable_scope("supervised") as scope:
            net = self.add_fully_connected_layer(z, 64, name="fc1")
            return self.add_fully_connected_layer(net, self.database.num_classes, name="fc2", activation=tf.nn.softmax)

    def _sample_latent(self, N):
        return np.random.normal(0, 1, [N, self.latent_size])

    def _finalize_model(self, learning_rate=1e-4):
        self.opt_dis = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_dis, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))
        self.opt_gen1 = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_gen1, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="opponent"))
        self.opt_sup = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_sup, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="supervised"))

        if self.bidirectional:
            self.opt_z = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_z)

        self.predicted_values = tf.argmax(self.y_out, 1)
        self.true_values = tf.argmax(self.y_true, 1)
        self.correct_prediction = tf.equal(self.predicted_values, self.true_values)

        labeled_mark = tf.reduce_sum(self.y_true, reduction_indices=[1])
        self.accuracy = tf.reduce_sum(labeled_mark * tf.cast(self.correct_prediction, self.data_type)) / tf.reduce_sum(
            labeled_mark)

    def create_model(self, learning_rate=1e-4):
        self.z = tf.placeholder(self.data_type, [None, self.latent_size])
        self.gen = self._generator(self.z)

        if self.bidirectional:
            self.latent_features = self._encoder(self.x)
            self.x_reconstruct = self._generator(self.latent_features, reuse=True)

            self.dis_real = self._double_discriminator(self.x, self.latent_features)
            self.dis_fake = self._double_discriminator(self.gen, self.z, reuse=True)
        else:
            self.dis_real, self.latent_features = self._discriminator(self.x, return_latent=True, minibatch=self.minibatch)
            self.dis_fake = self._discriminator(self.gen, reuse=True, minibatch=self.minibatch)

        self.y_out = self._supervised_head(self.latent_features)

        eps = 1e-9
        self.loss_dis = tf.reduce_mean(-0.5*tf.log(self.dis_real + eps) - 0.5*tf.log(1 - self.dis_fake + eps))

        if self.bidirectional:
            #self.loss_gen1 = tf.reduce_mean(-0.5 * tf.log(1 - self.dis_real + eps) - 0.5 * tf.log(self.dis_fake + eps))
            self.loss_gen1 = -self.loss_dis
        else:
            self.loss_gen1 = tf.reduce_mean(-0.5 * tf.log(self.dis_fake + eps))

        self.loss_sup = tf.reduce_mean(
            -tf.reduce_sum(self.y_true * tf.log(tf.maximum(self.y_out, 1e-15)), reduction_indices=[1]) * tf.reduce_sum(
                self.y_true, reduction_indices=[1]))

        if self.bidirectional:
            self.z_reconstruct = self._encoder(self.gen, reuse=True)
            self.loss_z = tf.reduce_mean((self.z - self.z_reconstruct)**2)

        self._finalize_model(learning_rate)

    def _test_step(self, step, num_components=2):
        feed_dict, xs, ys, names = self.get_feed_dict('TEST')

        acc = self.sess.run([self.accuracy], {
            self.x: xs,
            self.y_true: ys,
            self.is_train: False
        })
        print("step", step)
        print("accuracy", acc)

        self.save(CONFIG.SAVE_LOCATION)

        x, y, name = database.get_test_set()
        nn.plot_tsne(nn.get_latent_features(x), y, 2)

    def train_d(self, feed_dict, xs, ys, names):
        for k in range(1):
            feed_dict, xs, ys, names = self.get_feed_dict('TRAIN', use_labeled=False)
            z = self._sample_latent(xs.shape[0])

            loss_d, _, df, dr = self.sess.run([self.loss_dis, self.opt_dis, self.dis_fake, self.dis_real], {
                self.x: xs,
                self.z: z,
                self.is_train: True
            })
        return loss_d

    def train_g(self, feed_dict, xs, ys, names):
        for k in range(1):
            z = self._sample_latent(xs.shape[0])

            loss_g, _ = self.sess.run([self.loss_gen1, self.opt_gen1], {
                self.x: xs,
                self.z: z,
                self.is_train: True
            })

        return loss_g

    def train_s(self, feed_dict, xs, ys, names):
        for k in range(1):
            loss_s, _ = self.sess.run([self.loss_sup, self.opt_sup], {
                self.x: xs,
                self.y_true: ys,
                self.is_train: True
            })
            return loss_s

    def _train_step(self, step):
        feed_dict, xs, ys, names = self.get_feed_dict('TRAIN', use_labeled=False)
        k = 10

        if step % 1 == 0:
            loss_d = self.train_d(feed_dict, xs, ys, names)
            if step > k:
                loss_g = self.train_g(feed_dict, xs, ys, names)
        else:
            if step > k:
                loss_g = self.train_g(feed_dict, xs, ys, names)
            loss_d = self.train_d(feed_dict, xs, ys, names)

        #if step % 100 == 0:
        #    self.train_z_head()

        if step % 2000 == 0:
            generated = self.sess.run([self.gen], {
                self.z: self._sample_latent(20),
                self.is_train: True
            })[0]

            for i in range(20):
                plt.plot(generated[i])
                plt.show()

        if step > k:
            if step % 2000 == 0:
                if self.bidirectional:
                    generated = self.sess.run([self.x_reconstruct], {
                        self.x: xs,
                        self.is_train: True
                    })[0]

                    for i in range(5):
                        plt.plot(generated[i])
                        plt.plot(xs[i])
                        plt.show()

            if step % 500 == 0:
                print("dis", loss_d)
                print("gen", loss_g)
                print()

    def get_latent_features(self, x):
        return self.sess.run([self.latent_features], {
            self.x: x,
            self.is_train: False
        })[0]

    def plot_tsne(self, zs, ys, num_components):
        model = TSNE(n_components=num_components, random_state=0, perplexity=30, init='pca')
        points = model.fit_transform(zs)
        colors = ['red', 'green', 'blue', 'yellow']

        if num_components == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for i in range(zs.shape[0]):
                ax.scatter(points[i][0], points[i][1], points[i][2], c=colors[np.argmax(ys[i])])
            plt.show()

        elif num_components == 2:
            for i in range(zs.shape[0]):
                plt.scatter(points[i][0], points[i][1], c=colors[np.argmax(ys[i])])
            plt.show()

    def plot_gen(self):
        z = self._sample_latent(10)
        gen = self.sess.run([self.gen], {
            self.z: z,
            self.is_train: False
        })[0]

        for g in gen:
            plt.plot(g)
            plt.show()

    def train_classification_head(self):
        for i in range(50000):
            feed_dict, xs, ys, names = self.get_feed_dict('TRAIN', use_labeled=True)
            loss_s = self.train_s(feed_dict, xs, ys, names)

            if i % 100 == 0:
                self._test_step(i)
                print(loss_s)

    def train_z_head(self):
        for i in range(100):
            z = self._sample_latent(20)

            loss_z, _ = self.sess.run([self.loss_z, self.opt_z], {
                self.z: z,
                self.is_train: True
            })

            if i % 100 == 0:
                print(loss_z)

if __name__ == '__main__':
    train = True
    database = Database(CONFIG.FORMATTED_DATA, CONFIG.BATCH_SIZE, CONFIG.INTERPOLATION_LENGTH, CONFIG.NUM_CLASSES)
    nn = GAN(database, CONFIG.LATENT_SIZE, bidirectional=False, minibatch=False)
    nn.create_model()

    if train:
        nn.train(evaluate_every=CONFIG.EVALUATE_EVERY, periodic_save=CONFIG.PERIODIC_SAVE, save_location=CONFIG.SAVE_LOCATION,
                 max_iter=CONFIG.MAX_ITER)
    else:
        nn.load_variables(CONFIG.LOAD_LOCATION)
        x, y, name = database.get_test_set()
        nn.plot_tsne(nn.get_latent_features(x), y, 2)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            