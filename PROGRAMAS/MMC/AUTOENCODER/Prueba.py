# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:21:39 2024

@author: pjime
"""

import numpy as np
import csv
import math
import \
    tensorflow as tf  # Importamos TensorFlow, una biblioteca para aprendizaje autom√°tico                                            # Importamos el m√≥dulo os para interactuar con el sistema operativo
from sklearn.model_selection import \
    train_test_split  # Importamos train_test_split para dividir los datos en conjuntos de entrenamiento y prueba
from keras.utils import to_categorical  # Importamos to_categorical para codificar las etiquetas en formato one-hot
from keras.models import Sequential, Model  # Importamos Sequential, un modelo lineal para apilar capas de red neuronal
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, \
    UpSampling2D  # Importamos capas espec√≠ficas para construir una red neuronal convolucional
import pickle
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
import time
from matplotlib import pyplot as plt


class TimeHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def split_and_encode_data(data, labels, test_size=0.1, random_state=10):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)

    y_train -= 1
    y_test -= 1

    y_test = to_categorical(y_test, 9)
    y_train = to_categorical(y_train, 9)

    return X_train, X_test, y_train, y_test


def encoder(input_img):
    # encoder
    # input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4


def decoder(conv4):
    # decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)  # 7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)  # 7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2, 2))(conv6)  # 14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
    return decoded


def compilar(input_img):
    pass


def step_decay(epoch, drop):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = drop

    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate


def historial(history, time_callback, nombre):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    times = []

    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    times = time_callback.times

    with open(nombre, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(train_loss)):
            writer.writerow(
                {'epoch': i + 1, 'train_loss': train_loss[i], 'train_acc': train_acc[i], 'val_loss': val_loss[i],
                 'val_acc': val_acc[i], 'time': times[i]})


def train_model(input_img, file_name, X_train, y_train, X_test, y_test, batch_size=8, epochs=25):
    autoencoder = Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001))
    autoencoder.summary()

    time_callback = TimeHistory()

    with tf.device('/GPU:0'):
        autoencoder_train = autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                            validation_data=(X_test, X_test), shuffle=True, callbacks=[time_callback])

    historial(autoencoder_train, time_callback, file_name)

    autoencoder.save(file_name + '.keras')

    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    autoencoder.save_weights(file_name + '-weights' + '.keras')


if __name__ == "__main__":
    input_img = Input(shape=(224, 224, 3))
    nombre = 'primerautoencoder'

    with open('resultados.pkl', 'rb') as f:
        data, labels = pickle.load(f)

    labels = labels.astype(np.int64)

    X_train, X_test, y_train, y_test = split_and_encode_data(data, labels)

    train_model(input_img, nombre, X_train, y_train, X_test, y_test, 8, 25)













