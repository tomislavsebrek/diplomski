import matplotlib.pyplot as plt
import numpy as np
import os
import json
import random

from config import CONFIG


def normalize(array):
    return array / np.max(array)


def interpolate(array, sequence_length):
    xp = np.linspace(0, 1, num=len(array))
    new_x = np.linspace(0, 1, num=sequence_length)
    return np.interp(new_x, xp, array)


read_set = set()
for file in [f for f in os.listdir(CONFIG.PATH) if os.path.isfile(os.path.join(CONFIG.PATH, f))]:
    read_set.add(file.split('.')[0])

read_list = []
for file in read_set:
    read_list.append(file)

num_train = 0
num_valid = 0
num_test = 0

num_labeled = 0
num_unlabeled = 0
num = [0] * len(CONFIG.CLASS_NAMES)

random.shuffle(read_list)
for name in read_list:
    file = CONFIG.PATH + "/" + name + ".annotation.json"
    with open(file) as data_file:
        data = json.load(data_file)

    if data['phase'] == "train":
        num_train += 1
    if data['phase'] == "validation":
        num_valid += 1
    if data['phase'] == "test":
        num_test += 1

    if data['label'] != 'unlabeled':
        for index, c_name in enumerate(CONFIG.CLASS_NAMES):
            if data['label'] == c_name:
                num[index] += 1
        num_labeled += 1
    else:
        num_unlabeled += 1


print("Size of train set:", num_train)
print("Size of validation set:", num_valid)
print("Size of test set:", num_test)
print()
print("Labeled:", num_labeled)

for index, c_name in enumerate(CONFIG.CLASS_NAMES):
    print("\t" + c_name + " - ", num[index])

print("Unlabeled:", num_unlabeled)

while True:
    print("Press")
    print("\t1 - for labeling train set")
    print("\t2 - for labeling validation set")
    print("\t3 - for labeling test set")
    user_input = input()

    if user_input == '1':
        phase = 'train'
    elif user_input == '2':
        phase = 'validation'
    elif user_input == '3':
        phase = 'test'
    else:
        print("Wrong input...")
        continue
    break


for name in read_list:
    file = CONFIG.PATH + "/" + name
    x = np.load(file + ".npy")

    with open(file + ".annotation.json") as data_file:
        data = json.load(data_file)

    print("Heuristic label", data['heuristic_label'])
    print("Label", data['label'])
    print("Name", name)

    plt.figure(1)
    plt.subplot(511)
    plt.plot(x)
    plt.subplot(512)
    plt.plot(normalize(interpolate(x, 100)))
    plt.subplot(513)
    plt.plot(normalize(interpolate(x, 250)))
    plt.subplot(514)
    plt.plot(normalize(interpolate(x, 500)))
    plt.subplot(515)
    plt.plot(normalize(interpolate(x, 1000)))
    plt.show()

    labeled = False
    print("Press")
    for index, c_name in enumerate(CONFIG.CLASS_NAMES):
        print("\t" + str(index + 1) + " - for labeling example as " + c_name)

    user_input = input()
    for index, c_name in enumerate(CONFIG.CLASS_NAMES):
        if user_input == str(index+1):
            label = c_name
            labeled = True
            break

    if labeled:
        print("new label -->", label)
        print("new phase -->", phase)

        data['label'] = label
        data['phase'] = phase

        with open(file + ".annotation.json", "w") as data_file:
            data_file.write(json.dumps(data))
