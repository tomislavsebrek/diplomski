from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import random
import numpy as np
import os
import matplotlib.pyplot as plt

from coverage_generator import generate_batch


def normalize(array):
    return array / np.max(array, axis=1, keepdims=True)


def change_data(data, random_flip=False):
    mask = np.zeros(data.shape)

    for i in range(data.shape[0]):
        for j in range(10):
            mask[i][j] = 1

    for m in mask:
        np.random.shuffle(m)

    if random_flip:
        for i in range(mask.shape[0]):
            if i % 2 == 0:
                mask[i] = mask[i][::-1]

    noise = np.random.normal(0, 0.01, data.shape)
    return data + noise*mask


class Database:

    def __init__(self, data_dir, batch_size, input_length, num_classes, data_change=True, use_max=70):
        self.batch_size = batch_size
        self.input_length = input_length
        self.num_classes = num_classes
        self.data_change = data_change

        self.unlabeled_x = np.load(data_dir + '/unlabeled_x.npy')
        self.unlabeled_name = np.load(data_dir + '/unlabeled_name.npy')
        self.unlabeled_index = 0

        self.unlabeled_perm = np.random.permutation(self.unlabeled_x.shape[0])

        self.labeled_x = np.load(data_dir + '/labeled_x.npy')
        self.labeled_y = np.load(data_dir + '/labeled_y.npy')

        if use_max > 0:
            labeled_x = np.zeros([use_max*num_classes, input_length])
            labeled_y = np.zeros([use_max*num_classes, num_classes])
            total = dict()

            for i in range(num_classes):
                total[i] = 0

            index = 0
            for i in range(self.labeled_x.shape[0]):
                if total[np.argmax(self.labeled_y[i])] < use_max:
                    labeled_x[index] = self.labeled_x[i]
                    labeled_y[index] = self.labeled_y[i]

                    total[np.argmax(self.labeled_y[i])] += 1
                    index += 1

            self.labeled_x = labeled_x
            self.labeled_y = labeled_y

        print(self.labeled_x.shape[0])

        self.labeled_name = np.load(data_dir + '/labeled_name.npy')
        self.labeled_index = 0

        self.labeled_perm = np.random.permutation(self.labeled_x.shape[0])

        self.current_batch_x = np.zeros([batch_size, input_length])
        self.current_batch_y = np.zeros([batch_size, num_classes])
        self.current_batch_name = ['']*batch_size

        self.test_x = np.load(data_dir + '/test_x.npy')
        self.test_y = np.load(data_dir + '/test_y.npy')
        self.test_name = np.load(data_dir + '/test_name.npy')

        self.validation_x = np.load(data_dir + '/validation_x.npy')
        self.validation_y = np.load(data_dir + '/validation_y.npy')
        self.validation_name = np.load(data_dir + '/validation_name.npy')

    def get_next_batch(self, use_labeled=True, use_generated=False, labeled_number=8):
        if use_generated:
            x, y, name = generate_batch(self.batch_size, False, self.input_length)
            return normalize(x), y, name

        for index in range(self.batch_size):
            if use_labeled and index < labeled_number:
                x, y, name = self.get_element_from_labeled_set()
            else:
                x, y, name = self.get_element_from_unlabeled_set()

            self.current_batch_x[index] = x
            self.current_batch_y[index] = y
            self.current_batch_name[index] = name

        if self.data_change:
            self.current_batch_x = change_data(self.current_batch_x)

        return self.current_batch_x, self.current_batch_y, np.array(self.current_batch_name)

    def get_batch_size(self):
        return self.batch_size

    def get_endpoint_shapes(self):
        return [None, self.input_length], [None, self.num_classes]

    def get_element_from_unlabeled_set(self):
        element_index = self.unlabeled_perm[self.unlabeled_index]

        x = self.unlabeled_x[element_index]
        y = np.zeros([self.num_classes])
        name = self.unlabeled_name[element_index]

        self.unlabeled_index += 1
        self.unlabeled_index %= len(self.unlabeled_x)

        if self.unlabeled_index == 0:
            np.random.shuffle(self.unlabeled_perm)

        return x, y, name

    def get_element_from_labeled_set(self):
        element_index = self.labeled_perm[self.labeled_index]

        x = self.labeled_x[element_index]
        y = self.labeled_y[element_index]
        name = self.labeled_name[element_index]

        self.labeled_index += 1
        self.labeled_index %= len(self.labeled_x)

        if self.labeled_index == 0:
            np.random.shuffle(self.labeled_perm)

        return x, y, name

    def get_test_set(self):
        return self.test_x, self.test_y, self.test_name

    def get_validation_set(self):
        return self.validation_x, self.validation_y, self.validation_name
