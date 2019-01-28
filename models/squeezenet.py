import os, cv2
import re
import h5py
import pandas as pd
import numpy as np 
from keras.datasets import mnist, imdb, reuters, boston_housing
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# To construct layer
from keras import models, layers
from keras.layers import Input, Concatenate
from keras.models import Model
from keras.layers import Input, merge, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

# For training
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

# For visualization
#from visual_callbacks import AccLossPlotter
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

#%matplotlib inline
"""
# 	Ref - https://forums.fast.ai/t/what-is-subsample-of-convolution2d-doing/3555
#		 : Conv2D에서 subsample은 padding으로 바뀌었다. subsample == padding
# 		- https://faroit.github.io/keras-docs/1.2.2
#		 : 이전 버전 Convolution2D
# 	     - tykimos.github.io/2017/07/09/Training_Monitoring
#		 : 학습 과정 표시 
"""
# #################################################################################################################
# # Setup some keras variables
# np.random.seed(3)
# batch_size = 64 # 128
# nb_epoch = 1 # 100
# img_rows, img_cols = 28, 28
# #################################################################################################################
# # train = pd.read_csv("../data/train.csv", header=0).values
# # test = pd.read_csv("../data/test.csv", header=0).values
#
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#
# ###################################################  IMDB  #######################################################
#(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) #Binary classification, Most frequently appear words
#(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000) #single-label, multiclass classification
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data() #Regression

# print(train_data.shape) #(25000,) / (8982,)
# print(len(train_data[0])) # /87
# print(train_labels[0]) # /3
# print(len(train_data)) # /8982
# print(len(test_data)) # /2246
#
def vectorize_sequences(sequences, dimenstion=10000):
    results = np.zeros((len(sequences), dimenstion))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.0
    return results

def to_one_hot(labels, dimenstion=46):
    results = np.zeros((len(labels), dimenstion))
    for i, sequences in enumerate(labels):
        results[i, sequences] = 1.0
    return results

# imdb, reuters, boston_housing
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)

# imdb
#y_train = np.asarray(train_labels).astype('float32') #Labels to vector??. Actually no need to do this.
#y_test = np.asarray(test_labels).astype('float32') #Labels to vector??. maybe 1D vector, binary classification, [1,] or [0,]
#print(x_train.shape) #(25000, 10000)

# reuters
# print(test_labels.shape)
# one_hot_train_labels = to_categorical(train_labels) #to_one_hot(train_labels), single labels, multi-classification
# one_hot_test_labels = to_categorical(test_labels) #to_one_hot(test_labels), single labels, multi-classification
# print(one_hot_test_labels.shape)

# boston_housing
print(train_data.shape) #(404, 13), Each labels have different categories.
print(test_data.shape) #(102, 13)
print(train_targets.shape) #(404,)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],))) #(13,)
    #model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    #model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# IMDB
#model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])  # binary classification
# Reuter
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # single labels, multi-classification
# Boston
K = 4
num_val_samples = len(train_data) // K
num_epochs = 500
all_mae_histories = []
for i in range(K):
    print('Processing fold #', i)
    val_data = train_data[i*num_val_samples: (i+1) * num_val_samples]


# x_val = x_train[:10000]
# partial_x_train = x_train[10000:] #(15000,)
# print(partial_x_train.shape)
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]

x_val = x_train[:1000]
partial_x_train = x_train[1000:] #(15000,)
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,partial_y_train,
                 batch_size=512,
                 epochs=20,
                 validation_data=(x_val, y_val))

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss)+1)

# plt.plot(epochs, loss, 'bo', label='Training_loss')
# plt.plot(epochs, val_loss, 'b', label='Validation_loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.clf()
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
# plt.plot(epochs, acc, 'bo', label='Training_acc')
# plt.plot(epochs, val_acc, 'b', label='Validation_acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

result = model.evaluate(x_test, one_hot_test_labels)
print(result)
# ##################################################### CIFAR10 #######################################################
# X_val = X_train[50000:]
# Y_val = Y_train[50000:]
# X_train = X_train[:50000]
# Y_train = Y_train[:50000]
# print('X_Train Shape:',X_train.shape, ', X_Test Shape:', X_test.shape) #(60000,28,28) (10000,28,28)
# print('Y_Train Shape:', Y_train.shape, ', Y_Test Shape:', Y_test.shape) #(60000,) (10000,)
# print('X_Val Shape:', X_val.shape, ', Y_Val Shape: ', Y_val.shape)
#
# #################################################################################################################
# # digit = X_train[4]
# # my_slice = digit[7:-7, 7:-7]
# # plt.imshow(my_slice, cmap=plt.cm.binary)
# # plt.show()
# #################################################################################################################
#
#
# print('=====================================================')
# X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1).astype('float32') / 255.0 #50000,w,h,c
# X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1).astype('float32') / 255.0 #10000,w,h,c
# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255.0 #10000,w,h,c
# print('X_Train reshape:', X_train.shape, ', X_val reshape:',X_val.shape, ', X_test reshape:', X_test.shape) #(60000,28,28,1) (10000,28,28,1) (10000,28,28,1)
#
# # Train, Test set select
# # train_rand_idxs = np.random.choice(50000, 700)
# # val_rand_idxs = np.random.choice(10000, 100)
# # X_train = X_train[train_rand_idxs]
# # Y_train = Y_train[train_rand_idxs]
#
# # Convert the y output to a categorical output for keras
# Y_train = np_utils.to_categorical(Y_train,10)
# Y_val = np_utils.to_categorical(Y_val,10)
# Y_test = np_utils.to_categorical(Y_test,10)
# print('Y_train_categorical:', Y_train.shape, ', Y_val_categorical:', Y_val.shape, ', Y_test_categorical:',  Y_test.shape) #(60000,) (10000,) (10000,)
#
# nb_classes = Y_train.shape[1]

def SqueezeNet(nb_classes, inputs=(3, 224, 224)):
    """
    Keras Implementation of SqueezeNet(arXiv 1602.07360)
    Arguments:
        nb_classes: total number of final categories

        inputs -- shape of the input images (channel, cols, rows)
    """

    input_img = Input(shape=inputs, dtype='float32', name='Input')

    # conv1
    conv1 = Conv2D(filters=96, kernel_size=7, strides=2, padding='valid', activation='relu', data_format='channels_first', kernel_initializer='glorot_uniform', name='con1')(input_img)
    # max1
    max1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='max1')(conv1)  # (None, 64, 32, 64)

    ###################################################################################################################
    # fire2
    fire2_squeeze = Conv2D(filters=16, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire2_1')(max1)
    fire2_expand1 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire2_2')(fire2_squeeze)
    fire2_expand2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire2_3')(fire2_squeeze)
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])
    fire2 = Activation("linear")(merge2)

    # fire3
    fire3_squeeze = Conv2D(filters=16, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire3_1')(fire2)
    fire3_expand1 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire3_2')(fire3_squeeze)
    fire3_expand2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire3_3')(fire3_squeeze)
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])
    fire3 = Activation("linear")(merge3)

    # fire4
    fire4_squeeze = Conv2D(filters=32,  kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire4_1')(fire3)
    fire4_expand1 = Conv2D(filters=128, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire4_2')(fire4_squeeze)
    fire4_expand2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire4_3')(fire4_squeeze)
    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    fire4 = Activation("linear")(merge4)

    # max5
    max4 = MaxPooling2D(pool_size=(2, 2), name='max5')(fire4)
    ###################################################################################################################

    # fire5
    fire5_squeeze = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire5_1')(max4)
    fire5_expand1 = Conv2D(filters=128, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire5_2')(fire5_squeeze)
    fire5_expand2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire5_3')(fire5_squeeze)
    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand1])
    fire5 = Activation("linear")(merge5)

    # fire6
    fire6_squeeze = Conv2D(filters=48, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire6_1')(fire5)
    fire6_expand1 = Conv2D(filters=192, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire6_2')(fire6_squeeze)
    fire6_expand2 = Conv2D(filters=192, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire6_3')(fire6_squeeze)
    merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])
    fire6 = Activation("linear")(merge6)

    # fire7
    fire7_squeeze = Conv2D(filters=48, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire7_1')(fire6)
    fire7_expand1 = Conv2D(filters=192, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire7_2')(fire7_squeeze)
    fire7_expand2 = Conv2D(filters=192, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire7_3')(fire7_squeeze)
    merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])
    fire7 = Activation("linear")(merge7)

    # fire8
    fire8_squeeze = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire8_1')(fire7)
    fire8_expand1 = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire8_2')(fire8_squeeze)
    fire8_expand2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire8_3')(fire8_squeeze)
    merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])
    fire8 = Activation("linear")(merge8)

    # max10
    max8 = MaxPooling2D(pool_size=(2, 2), name='max10')(fire8)
    ###################################################################################################################

    # fire9
    fire9_squeeze = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire9_1')(max8)
    fire9_expand1 = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire9_2')(fire9_squeeze)
    fire9_expand2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform', name='fire9_3')(fire9_squeeze)
    merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])
    fire9 = Activation("linear")(merge9)
    ###################################################################################################################

    fire9_dropout = Dropout(0.5)(fire9)
    conv10 = Conv2D(filters=10, kernel_size=1, padding='valid', activation='relu', kernel_initializer='glorot_uniform', name='conv10')(fire9_dropout)

    global_avg_poling10 = GlobalAveragePooling2D(data_format='channels_last')(conv10)

    # flatten = Flatten()(conv12)
    softmax = Dense(nb_classes, activation="softmax")(global_avg_poling10)
    '''
    여기서 argmax 로 가장 높은 확률을 가지는 인덱스를 반환 하는 것이 아니라, 확률 그대로 넘기는 구나. 
    '''
    return Model(inputs=input_img, outputs=softmax)

def main():
    np.random.seed(45)
    nb_class = 2
    sn = SqueezeNet(nb_classes=nb_class, inputs=(3,224,224))

    sgd = SGD(lr=0.001, decay=0.0002, momentum=0.9, nesterov=True)
    sn.compile(optimizer=sgd, loss='categorical_crossentropy', metics=['accuracy'])

    # Describe the model, and plot the model
    sn.summary()

    # Training
    train_data_dir = '../data/train/'
    validation_data_dir = '../data/test/'
    # nb_train_samples = 2000
    # nb_validation_samples = 800
    nb_epoch = 500

    train_images = [train_data_dir + i for i in os.listdir(train_data_dir)]  # use this for full dataset
    #train_dogs = [train_data_dir + i for i in os.listdir(train_data_dir) if 'dog' in i]
    #train_cats = [train_data_dir + i for i in os.listdir(train_data_dir) if 'cat' in i]

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    #train_images.sort(key=natural_keys)
    train_images = train_images[0:1300] + train_images[12500:13800]

    #test_images.sort(key=natural_keys)

    def atoi(text):
        return int(text) if text.isdigit() else text

    def prepare_data(list_of_images):
        """
        Returns two arrays:
            x is an array of resized images
            y is an array of labels
        """
        x = []  # images as arrays
        y = []  # labels

        for image in list_of_images:
            x.append(cv2.resize(cv2.imread(image), (224, 224), interpolation=cv2.INTER_CUBIC))

        for i in list_of_images:
            if 'dog' in i:
                y.append(1)
            elif 'cat' in i:
                y.append(0)
            # else:
            # print('neither cat nor dog name present in images')

        return x, y

    X, Y = prepare_data(train_images)

    # First split the data in two sets, 80% for training, 20% for Val/Test)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)

    nb_train_samples = len(X_train)
    nb_validation_samples = len(X_val)

    #   Generator
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=32)
    validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=32)

    # train_generator = train_datagen.flow_from_directory(
    #         train_data_dir,
    #         target_size=(224, 224),
    #         batch_size=32,
    #         class_mode='categorical')
    #
    # validation_generator = test_datagen.flow_from_directory(
    #         validation_data_dir,
    #         target_size=(224, 224),
    #         batch_size=32,
    #         class_mode='categorical')

    for data_batch, labels_batch in train_generator:
        print('Batch Data Size:', data_batch.shape) #(32, 224,224,3)
        print('Batch Label Size:', labels_batch.shape) #(32, 224,224,3)
        break

    # Instantiate AccLossPlotter to visualise training
    plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

    checkpoint = ModelCheckpoint(
                    'weights.{epoch:02d}-{val_loss:.2f}.h5',
                    monitor='val_loss',
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='min',
                    period=1)

    history = sn.fit_generator(
        train_generator,
        steps_per_epoch=int(nb_train_samples/32),
        epochs=30,
        validation_data=validation_generator,
        validation_steps=int(nb_validation_samples/32)
    )

    # sn.fit_generator(
    #         train_generator,
    #         samples_per_epoch=nb_train_samples,
    #         nb_epoch=nb_epoch,
    #         validation_data=validation_generator,
    #         nb_val_samples=nb_validation_samples,
    #         callbacks=[plotter, checkpoint])

    sn.save_weights('weights.h5')
    sn.save('model_keras.h5')

    test_images = [validation_data_dir + i for i in os.listdir(validation_data_dir)]
    X_test, Y_test = prepare_data(test_images)  # Y_test in this case will be []
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)
    prediction_probabilities = sn.predict_generator(test_generator, verbose=1)

# if __name__ == '__main__':
#     main()
#     input('Press ENTER to exit...')

# #%matplotlib inline
# fig, loss_ax = plt.subplots()
#
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(hist.history['loss'], 'y', label='train_loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val_loss')
# acc_ax.plot(hist.history['acc'], 'b', label='train_acc')
# acc_ax.plot(hist.history['val_acc'], 'g', label='val_acc')
#
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuracy')
#
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
#
# plt.show()

# model.summary()
# input_layer = Input(shape=(28,28,1), name="input")

# #conv 1
# conv1 = Conv2D(96, 3, strides=(2,2), activation='relu', kernel_initializer='glorot_uniform',padding='valid')(input_layer)

# #maxpool 1
# maxpool1 = MaxPooling2D(pool_size=(2,2))(conv1)

# #fire 1
# fire2_squeeze = Conv2D(16, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(maxpool1)
# fire2_expand1 = Conv2D(64, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire2_squeeze)
# fire2_expand2 = Conv2D(64, 3, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire2_squeeze)
# merge1 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])
# #merge1 = merge(inputs=[fire2_expand1, fire2_expand2], mode="concat", concat_axis=1)
# fire2 = Activation("linear")(merge1)

# #fire 2
# fire3_squeeze = Conv2D(16, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire2)
# fire3_expand1 = Conv2D(64, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire3_squeeze)
# fire3_expand2 = Conv2D(64, 3, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire3_squeeze)
# merge2 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])
# #merge2 = merge(inputs=[fire3_expand1, fire3_expand2], mode="concat", concat_axis=1)
# fire3 = Activation("linear")(merge2)

# #fire 3
# fire4_squeeze = Conv2D(32, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire3)
# fire4_expand1 = Conv2D(128, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire4_squeeze)
# fire4_expand2 = Conv2D(128, 3, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire4_squeeze)
# merge3 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
# #merge3 = merge(inputs=[fire4_expand1, fire4_expand2], mode="concat", concat_axis=1)
# fire4 = Activation("linear")(merge3)

# #maxpool 4
# maxpool4 = MaxPooling2D((2,2))(fire4)

# #fire 5
# fire5_squeeze = Conv2D(32, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(maxpool4)
# fire5_expand1 = Conv2D(128, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire5_squeeze)
# fire5_expand2 = Conv2D(128, 3, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire5_squeeze)
# merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])
# #merge5 = merge(inputs=[fire5_expand1, fire5_expand2], mode="concat", concat_axis=1)
# fire5 = Activation("linear")(merge5)

# #fire 6
# fire6_squeeze = Conv2D(48, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire5)
# fire6_expand1 = Conv2D(192, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire6_squeeze)
# fire6_expand2 = Conv2D(192, 3, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire6_squeeze)
# merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])
# #merge6 = merge(inputs=[fire6_expand1, fire6_expand2], mode="concat", concat_axis=1)
# fire6 = Activation("linear")(merge6)

# #fire 7
# fire7_squeeze = Conv2D(48, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire6)
# fire7_expand1 = Conv2D(192, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire7_squeeze)
# fire7_expand2 = Conv2D(192, 3, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire7_squeeze)
# merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])
# #merge7 = merge(inputs=[fire7_expand1, fire7_expand2], mode="concat", concat_axis=1)
# fire7 =Activation("linear")(merge7)

# #fire 8
# fire8_squeeze = Conv2D(64, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire7)
# fire8_expand1 = Conv2D(256, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire8_squeeze)
# fire8_expand2 = Conv2D(256, 3, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire8_squeeze)
# merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])
# #merge8 = merge(inputs=[fire8_expand1, fire8_expand2], mode="concat", concat_axis=1)
# fire8 = Activation("linear")(merge8)

# #maxpool 8
# maxpool8 = MaxPooling2D((2,2))(fire8)

# #fire 9
# fire9_squeeze = Conv2D(64, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(maxpool8)
# fire9_expand1 = Conv2D(256, 1, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire9_squeeze)
# fire9_expand2 = Conv2D(256, 3, activation='relu', kernel_initializer='glorot_uniform',padding='same')(fire9_squeeze)
# merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])
# #merge8 = merge(inputs=[fire9_expand1, fire9_expand2], mode="concat", concat_axis=1)
# fire9 = Activation("linear")(merge8)
# fire9_dropout = Dropout(0.5)(fire9)

# #conv 10
# conv10 = Conv2D(10, 1, kernel_initializer='glorot_uniform',padding='valid')(fire9_dropout)

# # The original SqueezeNet has this avgpool1 as well. But since MNIST images are smaller (1,28,28)
# # than the CIFAR10 images (3,224,224), AveragePooling2D reduces the image size to (10,0,0), 
# # crashing the script.

# #avgpool 1
# #avgpool10 = AveragePooling2D((13,13))(conv10)

# flatten = Flatten()(conv10)

# softmax = Dense(nb_classes, activation="softmax")(flatten)

# model = Model(input=input_layer, output=softmax)


# model.compile(optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"])

# yPred = model.predict(testX, verbose=1)
# yPred = np.argmax(yPred, axis=1)

#np.savetxt('mnist-squeezenet.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

# score = model.evaluate(X_test, Y_test)


# SVG(model_to_dot(model, show_shapes=True).creat(prog='dot', format='svg'))

# # batch_size = 1024
# nb_epoch = 1
# img_rows, img_cols = 28, 28


