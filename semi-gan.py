import tensorflow as tf
import numpy as np

from neural_network import *
from config import CONFIG
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


class SemiGAN(NeuralNetwork):

    def __init__(self, database, latent_size, data_type=tf.float32, bidirectional=False, minibatch=True):
        super().__init__(database, data_type)

        self.latent_size = latent_size
        self.bidirectional = bidirectional
        self.minibatch = minibatch

        self.best_score = 0

    def _generator(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            net = self.add_fully_connected_layer(z, 16 * self.database.input_length, name="fc1")
            net = self.add_batch_norm_layer(net, name="bn1")
            net = tf.reshape(net, [-1, int(self.database.input_length / 4), 1, 64])
            net = self.add_1d_convolution_transpose_layer(net, 3, 32, name="deconv1")
            net = self.add_batch_norm_layer(net, name="bn2")
            net = self.add_max_pool_transpose_layer(net, 2, name="max_transp1")
            net = self.add_1d_convolution_transpose_layer(net, 5, 16, name="deconv2")
            net = self.add_batch_norm_layer(net, name="bn3")
            net = self.add_max_pool_transpose_layer(net, 2, name="max_transp2")
            net = self.add_1d_convolution_transpose_layer(net, 5, 1, name="deconv3")
            return self.add_fully_connected_layer(net, self.database.input_length, name="fc2", activation=tf.nn.sigmoid)

    def _discriminator(self, x, return_latent=False, reuse=False, minibatch=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            net = x
            net = self.add_1d_convolution_layer(net, 5, 16, name="conv1")
            net = self.add_max_pool_layer(net, name="pool1")
            net = self.add_1d_convolution_layer(net, 3, 32, name="conv3")
            net = self.add_max_pool_layer(net, name="pool1")
            net = self.add_fully_connected_layer(net, 256, name="fc1")

            if minibatch:
                num_kernels = 5
                kernel_dim = 3
                x = net

                net = self.add_fully_connected_layer(net, num_kernels * kernel_dim, stddev=0.02, name="minibatch")
                activation = tf.reshape(net, (-1, num_kernels, kernel_dim))
                diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
                abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
                minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
                net = tf.concat(1, [x, minibatch_features])
            else:
                net = self.add_fully_connected_layer(net, 1024, name="fc2")

            o = self.add_fully_connected_layer(net, self.database.num_classes+1, activation=tf.nn.softmax, name="fc3")

            if return_latent:
                return o, net
            return o

    def _sample_latent(self, N, uniform=True):
        if uniform:
            return np.random.uniform(-1, 1, [N, self.latent_size])
        return np.random.normal(0, 1, [N, self.latent_size])

    def _finalize_model(self, learning_rate=1e-4*2):
        self.opt_dis = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(self.loss_dis, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))
        self.opt_gen = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(self.loss_gen, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))
        self.opt_sup = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_sup)

        self.predicted_values = tf.argmax(self.y_out, 1)
        self.true_values = tf.argmax(self.y_true, 1)
        self.correct_prediction = tf.equal(self.predicted_values, self.true_values)

        labeled_mark = tf.reduce_sum(self.y_true, reduction_indices=[1])
        self.accuracy = tf.reduce_sum(labeled_mark * tf.cast(self.correct_prediction, self.data_type)) / tf.reduce_sum(
            labeled_mark)

    def create_model(self, learning_rate=1e-4):
        self.z = tf.placeholder(self.data_type, [None, self.latent_size])
        self.gen = self._generator(self.z)
        self.dis_real, self.latent_features = self._discriminator(self.x, return_latent=True, minibatch=self.minibatch)
        self.dis_fake = self._discriminator(self.gen, reuse=True, minibatch=self.minibatch)

        self.y_out = self.dis_real[:, 0:self.database.num_classes]
        self.y_real = 1-self.dis_real[:, self.database.num_classes]
        self.y_fake = 1-self.dis_fake[:, self.database.num_classes]

        eps = 1e-9
        self.loss_dis = tf.reduce_mean(-0.5*tf.log(self.y_real + eps) - 0.5*tf.log(1 - self.y_fake + eps))
        self.loss_gen = tf.reduce_mean(-0.5*tf.log(self.y_fake + eps))

        self.loss_sup = tf.reduce_mean(
            -tf.reduce_sum(self.y_true * tf.log(tf.maximum(self.y_out, 1e-15)), reduction_indices=[1]) * tf.reduce_sum(
                self.y_true, reduction_indices=[1]))

        self._finalize_model(learning_rate)

    def _test_step(self, step, num_components=2):
        feed_dict, xs, ys, names = self.get_feed_dict('TEST')

        acc, true, predicted, y_out = self.sess.run([self.accuracy, self.true_values, self.predicted_values, self.y_out], {
            self.x: xs,
            self.y_true: ys,
            self.is_train: False
        })

        if acc > self.best_score:
            self.best_score = acc
            self.pending = 0
            self.save("save/checkpoint-gan.ckpt")

        print("step", step)
        print("Current accuracy:", acc)
        print("Confusion matrix:")
        print(self.analyzer.get_confusion_matrix(predicted, true))

        self.analyzer.plot_precision_recall(y_out, ys)

    def train_d(self, feed_dict, xs, ys, names):
        for k in range(1):
            z = self._sample_latent(xs.shape[0])

            loss_d, _ = self.sess.run([self.loss_dis, self.opt_dis], {
                self.x: xs,
                self.z: z,
                self.is_train: True
            })
        return loss_d

    def train_g(self, feed_dict, xs, ys, names):
        for k in range(1):
            z = self._sample_latent(xs.shape[0])

            loss_g, _ = self.sess.run([self.loss_gen, self.opt_gen], {
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
        th_step = 0

        if step >= th_step:
            xs, ys, names = self.database.get_next_batch(labeled_number=20)

            feed_dict = {
                self.x: xs,
                self.y_true: ys,
                self.is_train: True
            }

            if step % 2 == 0:
                loss_d = self.train_d(feed_dict, xs, ys, names)
                loss_s = self.train_s(feed_dict, xs, ys, names)
                if step > 100:
                    loss_g = self.train_g(feed_dict, xs, ys, names)
            else:
                if step > 100:
                    loss_g = self.train_g(feed_dict, xs, ys, names)
                loss_d = self.train_d(feed_dict, xs, ys, names)
                loss_s = self.train_s(feed_dict, xs, ys, names)
        else:
            xs, ys, names = self.database.get_next_batch(labeled_number=0)

            feed_dict = {
                self.x: xs,
                self.y_true: ys,
                self.is_train: True
            }

            if step % 2 == 0:
                loss_d = self.train_d(feed_dict, xs, ys, names)
                if step > 100:
                    loss_g = self.train_g(feed_dict, xs, ys, names)
            else:
                if step > 100:
                    loss_g = self.train_g(feed_dict, xs, ys, names)
                loss_d = self.train_d(feed_dict, xs, ys, names)

        if step > 100:
            if step % 1000000 == 0:
                generated = self.sess.run([self.gen],{
                    self.z: self._sample_latent(20),
                    self.is_train: True
                })[0]

                for i in range(20):
                    plt.plot(generated[i])
                    plt.show()

    def get_latent_features(self, x):
        return self.sess.run([self.latent_features], {
            self.x: x,
            self.is_train: False
        })[0]

    def plot_tsne(self, zs, ys, num_components):
        model = TSNE(n_components=num_components, random_state=0, perplexity=60, init='pca')
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


if __name__ == '__main__':
    train = False
    database = Database(CONFIG.FORMATTED_DATA, CONFIG.BATCH_SIZE, CONFIG.INTERPOLATION_LENGTH, CONFIG.NUM_CLASSES, use_max=0)
    nn = SemiGAN(database, CONFIG.LATENT_SIZE, bidirectional=False, minibatch=False)
    nn.create_model()

    if train:
        nn.train(evaluate_every=100, periodic_save=CONFIG.PERIODIC_SAVE, save_location="save/checkpoint-gan.ckpt",
                 max_iter=CONFIG.MAX_ITER)
    else:
        nn.load_variables("save/optimal/checkpoint-gan.ckpt")
        nn._test_step(10000)
