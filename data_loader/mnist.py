import os
import sys
import time
import pickle
import random
import numpy as np

from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_data = np.expand_dims(train_data, axis=-1) # train_data.reshape(-1, 28, 28, 1)
    test_data = np.expand_dims(test_data, axis=-1) #test_data.reshape(-1, 28, 28, 1)

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)

    num_of_test_data = 50000
    val_data = train_data[num_of_test_data:]
    val_labels = train_labels[num_of_test_data:]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)