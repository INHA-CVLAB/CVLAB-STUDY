"""
 Generates numpys
- Generates images
- Generates tfrecords
"""
import os

import numpy as np
import imageio
import pickle
from tqdm import tqdm

from subprocess import call
import tarfile

import tensorflow as tf

#================================== Easy way =================================#
from keras.datasets import cifar100
from keras.utils import to_categorical

def normalize(X_train, X_test):

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def load_cifar100() :
    (train_data, train_labels), (test_data, test_labels) = cifar100.load_data()

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 100)
    test_labels = to_categorical(test_labels, 100)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)

    return train_data, train_labels, test_data, test_labels
#===============================================================================#

#================================== Hard way =================================#
def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def save_imgs_to_disk(path, arr, file_names):
    for i, img in tqdm(enumerate(arr)):
        imageio.imwrite(path + file_names[i], img, 'PNG-PIL')


def save_numpy_to_disk(path, arr):
    np.save(path, arr)


def save_tfrecord_to_disk(path, arr_x, arr_y):
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in tqdm(range(arr_x.shape[0])):
            image_raw = arr_x[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_y[i]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())


def main():
    fname = '../data/cifar-100-python.tar.gz'

    if not os.path.exists("../data/cifar-100-python.tar.gz"):
        call(
            "wget http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
            shell=True
        )
        print("Downloading done.\n")
    else:
        print("Dataset already downloaded. Did not download twice.\n")

    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
    elif (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()

    dic_train, dic_test = unpickle('../data/cifar-100-python/train'), unpickle('../data/cifar-100-python/test')

    for key, val in dic_train.items():
        print(key)

    x_train = dic_train[b'data']
    x_test = dic_test[b'data']

    x_train_filenames = dic_train[b'filenames']
    x_test_filenames = dic_test[b'filenames']

    y_train = np.array(dic_train[b'fine_labels'], np.int32)
    y_test = np.array(dic_test[b'fine_labels'], np.int32)

    # Reshape and transposing the numpy array of the images
    x_train = np.transpose(x_train.reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
    x_test = np.transpose(x_test.reshape((-1, 3, 32, 32)), (0, 2, 3, 1))

    x_train_len = x_train.shape[0]
    x_test_len = x_test.shape[0]

    print(x_train.shape, x_train.dtype) #(50000, 32, 32, 3), uint8
    print(y_train.shape, y_train.dtype) #(50000,), int32
    print(x_test.shape, x_test.dtype) #(10000, 32, 32, 3), uint8
    print(y_test.shape, y_test.dtype) #(10000,), int32

    if not os.path.exists('../data/cifar-100-python/imgs/'):
        os.makedirs('../data/cifar-100-python/imgs/')

    for i in range(x_train_len):
        x_train_filenames[i] = 'imgs/' + str(x_train_filenames[i].decode('ascii'))
    for i in range(x_test_len):
        x_test_filenames[i] = 'imgs/' + str(x_test_filenames[i].decode('ascii'))

    # Save the filename of x_train and y_train
    with open('../data/cifar-100-python/x_train_filenames.pkl', 'wb') as f:
        pickle.dump(x_train_filenames, f)
    with open('../data/cifar-100-python/x_test_filenames.pkl', 'wb') as f:
        pickle.dump(x_test_filenames, f)

    print("FILENAMES OF IMGS saved successfully")

    save_imgs_to_disk('../data/cifar-100-python/', x_train, x_train_filenames)
    save_imgs_to_disk('../data/cifar-100-python/', x_test, x_test_filenames)
    print("IMGS saved successfully")
#===============================================================================#

   #  save_numpy_to_disk('cifar-100-python/x_train.npy', x_train)
   # save_numpy_to_disk('cifar-100-python/y_train.npy', y_train)
   #  save_numpy_to_disk('cifar-100-python/x_test.npy', x_test)
   #  save_numpy_to_disk('cifar-100-python/y_test.npy', y_test)
   #  print("Numpys saved successfully")
   #
   #
   #  # SAVE ALL the data with one pickle
   #  with open('cifar-100-python/data_numpy.pkl', 'wb')as f:
   #      pickle.dump({'x_train': x_train,
   #                   'y_train': y_train,
   #                   'x_test': x_test,
   #                   'y_test': y_test,
   #                   }, f)
   #  print("DATA NUMPY PICKLE saved successfully..")
   #
   #
   #  save_tfrecord_to_disk('cifar-100-python/train.tfrecord', x_train, y_train)
   #  save_tfrecord_to_disk('cifar-100-python/test.tfrecord', x_test, y_test)
   #  print('tfrecord saved successfully..')


if __name__ == '__main__':
    main()

