from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from database import *
from neural_network import *
from autoencoder import *
from config import CONFIG

import tensorflow as tf
import numpy as np
import math

import matplotlib.pyplot as plt


class SemiVAE(NeuralNetwork):

    def __init__(self, database, latent_size=10, data_type=tf.float32, labeled_output=True):
        super().__init__(database, data_type, labeled_output)

        self.latent_size = latent_size
        self.loss = tf.constant(0, dtype=tf.float32)
        self.best_score = 0
        self.pending = 0

    def add_loss_part(self, loss_part, loss_constant=1.0):
        self.loss += tf.reduce_mean(loss_part) * loss_constant

    def create_model(self, learning_rate=1e-3):
        self.net_dis = self._create_supervised_head(self.get_input_tensor())
        self.zs = [tf.placeholder(tf.float32, [None, self.latent_size]) for i in range(self.database.num_classes)]
        self.net_gens = []

        for i in range(self.database.num_classes):
            y_mock = np.zeros([self.database.num_classes])
            y_mock[i] = 1.0
            y_mock = tf.constant(y_mock, dtype=self.data_type)
            x = self.x
            con = x[:, 0:self.database.num_classes] * 0 + y_mock

            print(con.get_shape)
            xy = tf.concat(1, [x, con])

            self.zs[i] = (self._create_encoder_part(xy, index=i))

            zy = tf.concat(1, [self.zs[i], con])
            self.net_gens.append(self._create_decoder_part(zy, index=i))

        self._finalize_model(learning_rate)

    def _finalize_model(self, learning_rate=1e-3):
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.predicted_values = tf.argmax(self.net_dis, 1)
        self.true_values = tf.argmax(self.y_true, 1)
        self.correct_prediction = tf.equal(self.predicted_values, self.true_values)

        labeled_mark = tf.reduce_sum(self.y_true, reduction_indices=[1])
        self.accuracy = tf.reduce_sum(labeled_mark * tf.cast(self.correct_prediction, self.data_type)) / tf.reduce_sum(
            labeled_mark)

    def _test_step(self, step, num_components=2):
        feed_dict, xs, ys, names = self.get_feed_dict('TEST')

        accuracy, output, predicted_labels, true_labels, zs, gens, dis, y_out = self.sess.run(
            [self.accuracy, self.net_dis, self.predicted_values, self.true_values, self.zs, self.net_gens, self.net_dis, self.y_out],
            feed_dict=feed_dict)

        if accuracy > self.best_score:
            self.best_score = accuracy
            self.pending = 0
            self.save("save/checkpoint-m1m2.ckpt")
        else:
            self.pending += 1

        if CONFIG.USE_CONFUSION:
            print("Current accuracy:", accuracy)
            print("Confusion matrix:")
            print(self.analyzer.get_confusion_matrix(predicted_labels, true_labels))

            for i in range(xs.shape[0]):
                if predicted_labels[i] != true_labels[i]:
                    print(np.max(dis[i]))

        print("best:", self.best_score)

        self.analyzer.plot_precision_recall(y_out, ys)

        if CONFIG.USE_TSNE:
            z = [None, None, None, None]
            z[0], z[1], z[2], z[3] = self.sess.run([self.zs[0], self.zs[1], self.zs[2], self.zs[3]], feed_dict={
                self.x: xs
            })
            zzz = np.zeros([xs.shape[0], 3])
            for i in range(xs.shape[0]):
                for j in range(3):
                    zzz[i][j] = z[np.argmax(ys[i])][i][j]

            model = TSNE(n_components=num_components, random_state=0, perplexity=30, init='pca')
            points = model.fit_transform(zzz)
            colors = ['red', 'green', 'blue', 'yellow']

            for i in range(xs.shape[0]):
                plt.scatter(points[i][0], points[i][1], c=colors[np.argmax(ys[i])])
            plt.show()

    def _train_step(self, step):
        use_labeled = False
        if step > 0:
            use_labeled = True

        feed_dict, xs, ys, names = self.get_feed_dict('TRAIN', use_labeled=use_labeled)

        _, curr_loss, curr_acc = self.sess.run([self.train_step, self.loss, self.accuracy], feed_dict=feed_dict)

    def _create_encoder_part(self, x, index):
        with tf.variable_scope("encoder") as scope:
            if index > 0:
                scope.reuse_variables()

            net = self.add_fully_connected_layer(x, 50, name="fc1")
            net = self.add_fully_connected_layer(net, 50, name="fc2")
            net = self.add_fully_connected_layer(net, 50, name="fc3")

            z, x_mean, x_sigma = self.add_reparametrization_sample_layer(net, self.latent_size, True)
            x_log_sigma_sq = tf.log(tf.square(x_sigma))
            loss = -0.5 * tf.reduce_sum(1 + x_log_sigma_sq - tf.exp(x_log_sigma_sq) - tf.square(x_mean), 1)

            loss = loss * tf.reduce_sum(self.y_true, reduction_indices=[1])*self.y_true[:, index] + \
                   loss * (1 - tf.reduce_sum(self.y_true, reduction_indices=[1])) * self.net_dis[:, index]
            self.add_loss_part(loss)

            return z

    def _create_decoder_part(self, z, index, distribution='GAUSS'):
        with tf.variable_scope("decoder") as scope:
            if index > 0:
                scope.reuse_variables()

            net = self.add_fully_connected_layer(z, 50, name="fc1")
            net = self.add_fully_connected_layer(net, 50, name="fc2")
            net = self.add_fully_connected_layer(net, 50, name="fc3")

            if distribution == 'GAUSS':
                x_reconstruct, x_mean, x_sigma = self.add_reparametrization_sample_layer(net,
                                                                                         self.database.input_length,
                                                                                         True)

                nom = tf.exp(-(self.x - x_mean) * (self.x - x_mean) / (2 * x_sigma * x_sigma))
                denom = x_sigma * math.sqrt(2 * math.pi) + self.eps
                loss = -tf.reduce_sum(tf.log(self.eps + nom / denom), 1)

            elif distribution == 'BERNOULLI':
                x_reconstruct = self.add_fully_connected_layer(net, self.database.input_length,
                                                               activation=tf.nn.sigmoid)
                loss = -tf.reduce_sum(
                    self.x * tf.log(1e-9 + x_reconstruct) + (1 - self.x) * tf.log(1e-9 + 1 - x_reconstruct), 1)

            loss = loss * tf.reduce_sum(self.y_true, reduction_indices=[1]) * self.y_true[:, index] + \
                   (loss + tf.log(self.net_dis[:, index]+0.0001)) * (1 - tf.reduce_sum(self.y_true, reduction_indices=[1])) * self.net_dis[:, index]
            self.add_loss_part(loss)

            return x_reconstruct

    def _create_supervised_head(self, x):
        with tf.variable_scope("classification"):
            net = self.add_fully_connected_layer(x, 50, name="fc1")
            net = self.add_fully_connected_layer(net, 50, name="fc2")
            self.y_out = self.add_output_layer(net, name="output")

            loss = -tf.reduce_sum(self.y_true * tf.log(tf.maximum(self.y_out, 1e-15)), reduction_indices=[1])
            loss = tf.reduce_mean(loss * tf.reduce_sum(self.y_true, reduction_indices=[1]))
            self.add_loss_part(loss, 0)

            return self.y_out

if __name__ == '__main__':
    database = Database("formatted_data/interpolation_500_z", CONFIG.BATCH_SIZE, 10, CONFIG.NUM_CLASSES, False, use_max=0)
    nn = SemiVAE(database, 3)
    nn.create_model()
    nn.train(evaluate_every=200, periodic_save=0, save_location="save/ckeckpoint-m1m2.ckpt", max_iter=CONFIG.MAX_ITER)
    nn._test_step(0)
