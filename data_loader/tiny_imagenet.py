# -*- coding:utf-8 -*-

import os
import sys
import time
import pickle
import random
import numpy as np
from scipy import misc

from keras.utils import to_categorical

def tiny_imagenet():
    '''
    Download: https://tiny-imagenet.herokuapp.com/
    '''
    IMAGENET_MEAN = [123.68, 116.78, 103.94]
    path = '../data/tiny-imagenet-200'
    num_classes = 200

    print('Loading ' + str(num_classes) + ' classes')

    X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype=np.float32)
    y_train = np.zeros([num_classes * 500], dtype=np.float32)

    trainPath = path + '/train'

    print('loading training images...')

    i = 0
    j = 0
    annotations = {}
    for sChild in os.listdir(trainPath):
        sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
        annotations[sChild] = j
        for c in os.listdir(sChildPath):
            X = misc.imread(os.path.join(sChildPath, c), mode='RGB')
            if len(np.shape(X)) == 2:
                X_train[i] = np.array([X, X, X])
            else:
                X_train[i] = np.transpose(X, (2, 0, 1))
            y_train[i] = j
            i += 1
        j += 1
        if (j >= num_classes):
            break

    print('finished loading training images')

    val_annotations_map = get_annotations_map()

    X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype=np.float32)
    y_test = np.zeros([num_classes * 50], dtype=np.float32)

    print('loading test images...')

    i = 0
    testPath = path + '/val/images'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X = misc.imread(sChildPath, mode='RGB')
            if len(np.shape(X)) == 2:
                X_test[i] = np.array([X, X, X])
            else:
                X_test[i] = np.transpose(X, (2, 0, 1))
            y_test[i] = annotations[val_annotations_map[sChild]]
            i += 1
        else:
            pass

    print('finished loading test images : ' + str(i))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    # X_train /= 255.0
    # X_test /= 255.0

    # for i in range(3) :
    #     X_train[:, :, :, i] =  X_train[:, :, :, i] - IMAGENET_MEAN[i]
    #     X_test[:, :, :, i] = X_test[:, :, :, i] - IMAGENET_MEAN[i]

    X_train, X_test = normalize(X_train, X_test)


    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    X_train = np.transpose(X_train, [0, 3, 2, 1])
    X_test = np.transpose(X_test, [0, 3, 2, 1])

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    return X_train, y_train, X_test, y_test

def get_annotations_map():
    valAnnotationsPath = './tiny-imagenet-200/val/val_annotations.txt'
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations
