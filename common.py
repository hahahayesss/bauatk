import os
import numpy as np
import tensorflow
from tensorflow import keras


def load_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


def load_dataset():
    (train_x, train_y), (test_x, test_y) = tensorflow.keras.datasets.mnist.load_data()
    train_x, test_x = train_x[..., np.newaxis] / 255.0, test_x[..., np.newaxis] / 255.0
    return train_x, train_y, train_x, train_y


def slice_data(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]] < dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1
    return np.array(dx), np.array(dy)


def genOutDir():
    if not os.path.exists('out'):
        os.mkdir('out')
