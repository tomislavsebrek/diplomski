import os
import numpy as np
import json
from config import CONFIG


def normalize(array):
    return array / np.max(array)


def interpolate(array, sequence_length):
    xp = np.linspace(0, 1, num=len(array))
    new_x = np.linspace(0, 1, num=sequence_length)
    return np.interp(new_x, xp, array)


np_dict = dict()
np_dict['unlabeled_x'] = np.zeros([CONFIG.INTERPOLATION_LENGTH])
np_dict['unlabeled_name'] = np.array([''])

np_dict['labeled_x'] = np.zeros([CONFIG.INTERPOLATION_LENGTH])
np_dict['labeled_y'] = np.zeros([CONFIG.NUM_CLASSES])
np_dict['labeled_name'] = np.array([''])

np_dict['test_x'] = np.zeros([CONFIG.INTERPOLATION_LENGTH])
np_dict['test_y'] = np.zeros([CONFIG.NUM_CLASSES])
np_dict['test_name'] = np.array([''])

np_dict['validation_x'] = np.zeros([CONFIG.INTERPOLATION_LENGTH])
np_dict['validation_y'] = np.zeros([CONFIG.NUM_CLASSES])
np_dict['validation_name'] = np.array([''])

read_set = set()
for file in [f for f in os.listdir(CONFIG.PATH) if os.path.isfile(os.path.join(CONFIG.PATH, f))]:
    read_set.add(file.split('.')[0])

i = 0
for file in read_set:
    if i % 100 == 0:
        print(i)
    i += 1

    read = file.split('.')[0]
    x = np.load(CONFIG.PATH + "/" + read + ".npy")

    y = np.zeros([CONFIG.NUM_CLASSES])
    x = interpolate(x, CONFIG.INTERPOLATION_LENGTH)

    if np.max(x) == 0:
        continue

    x = normalize(x)

    with open(CONFIG.PATH + "/" + read + ".annotation.json") as data_file:
        data = json.load(data_file)

    for index, c_name in enumerate(CONFIG.CLASS_NAMES):
        if c_name == data['label']:
            y[index] = 1
            break
    else:
        if data['label'] != 'unlabeled':
            print("ERROR")

    if data["phase"] == 'test':
        np_dict['test_x'] = np.vstack((np_dict['test_x'], x))
        np_dict['test_y'] = np.vstack((np_dict['test_y'], y))
        np_dict['test_name'] = np.vstack((np_dict['test_name'], file))
        continue

    if data["phase"] == 'validation':
        np_dict['validation_x'] = np.vstack((np_dict['validation_x'], x))
        np_dict['validation_y'] = np.vstack((np_dict['validation_y'], y))
        np_dict['validation_name'] = np.vstack((np_dict['validation_name'], file))
        continue

    # train

    if np.sum(y) == 1:  # labeled
        np_dict['labeled_x'] = np.vstack((np_dict['labeled_x'], x))
        np_dict['labeled_y'] = np.vstack((np_dict['labeled_y'], y))
        np_dict['labeled_name'] = np.vstack((np_dict['labeled_name'], file))

    else:
        np_dict['unlabeled_x'] = np.vstack((np_dict['unlabeled_x'], x))
        np_dict['unlabeled_name'] = np.vstack((np_dict['unlabeled_name'], file))


for key in np_dict:
    if len(np_dict[key].shape) > 1:
        np.save("formatted_data/interpolation_" + str(CONFIG.INTERPOLATION_LENGTH) + "/" + key + ".npy", np_dict[key][1:])
