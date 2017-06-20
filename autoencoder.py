from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from database import *
from neural_network import *

import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from sklearn.manifold import TSNE
from config import CONFIG


class Autoencoder(NeuralNetwork):

    def __init__(self, database, latent_size=10, data_type=tf.float32, labeled_output=True):
        super().__init__(database, data_type, labeled_output)

        self.latent_size = latent_size
        self.loss = tf.constant(0, dtype=tf.float32)

    def add_loss_part(self, loss_part, loss_constant=1.0):
        self.loss += tf.reduce_mean(loss_part) * loss_constant

    def create_model(self, learning_rate=1e-3):
        self.z = self._create_encoder_part(self.get_input_tensor())
        self.net_gen = self._create_decoder_part(self.z)
        self.net_dis = self._create_supervised_head(self.z)
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

        accuracy, output, xs_reconstructed, zs, predicted_labels, true_labels = self.sess.run(
            [self.accuracy, self.net_dis, self.net_gen, self.z, self.predicted_values, self.true_values],
            feed_dict=feed_dict)
        print(step)

        if CONFIG.USE_CONFUSION:
            print("Current accuracy:", accuracy)
            print("Confusion matrix:")
            print(self.analyzer.get_confusion_matrix(predicted_labels, true_labels))

        if CONFIG.USE_TSNE:
            xs = np.load("formatted_data/interpolation_500/unlabeled_x.npy")[0:1500, :]
            ys, zs = self.sess.run([self.predicted_values, self.z], feed_dict={
                self.x: xs
            })

            y_o = np.zeros([xs.shape[0], self.database.num_classes])
            for i in range(xs.shape[0]):
                y_o[i][ys[i]] = 1

            self.visualize_tsne(zs, y_o, CONFIG.TSNE_COMPONENTS)

        if CONFIG.USE_RECONSTRUCTION:
            self.plot_reconstruction(xs, xs_reconstructed)

    def _train_step(self, step):
        use_labeled = False
        if step > CONFIG.USE_LABELED_AFTER:
            use_labeled = True

        feed_dict, xs, ys, names = self.get_feed_dict('TRAIN', use_labeled=use_labeled)

        _, curr_loss, curr_acc = self.sess.run([self.train_step, self.loss, self.accuracy], feed_dict=feed_dict)

    def _create_encoder_part(self):
        pass

    def _create_decoder_part(self):
        pass

    def _create_supervised_head(self):
        pass

    def visualize_tsne(self, zs, ys, num_components):
        assert num_components == 2 or num_components == 3

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

    def plot_reconstruction(self, xs, xs_reconstructed):
        for i in range(xs.shape[0]):
            plt.plot(xs[i])
            plt.plot(xs_reconstructed[i])
            plt.axis([0, CONFIG.INTERPOLATION_LENGTH, 0, 1])
            plt.show()

    def plot_differences(self, xs, xs_reconstructed, ys_true, ys_predicted):
        for i in range(xs.shape[0]):
            if ys_true[i] != ys_predicted[i]:
                print("Predicted:", ys_predicted[i])
                print("True:", ys_true[i])

                plt.plot(xs[i])
                plt.plot(xs_reconstructed[i])
                plt.show()

    def get_latent(self, xs):
        return self.sess.run(
            [self.z], feed_dict={
                self.x: xs
            })[0]