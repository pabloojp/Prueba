import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train_full = x_train_full.astype(np.float32) / 255.0
X_test = x_test.astype(np.float32) / 255.0
x_train,x_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train,y_valid = y_train_full[:-5000], y_train_full[-5000:]

stacked_encoder = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu", name="30UnitsDense"),
])
stacked_decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=[30]),
    tf.keras.layers.Dense(28*28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])