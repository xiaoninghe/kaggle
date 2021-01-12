import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import tensorflowjs as tfjs

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

reg_factor = 8e-5
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=l2(reg_factor)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=l2(reg_factor)))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# print(list(np.asarray(X_train[0])))

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=34)

graph = list(history.history.keys())
_, axs = plt.subplots(1, 2, constrained_layout=True)
for n in range(2):
    graph_type = graph[n]
    axs[n].plot(history.history[graph_type])
    axs[n].plot(history.history[f"val_{graph_type}"])
    axs[n].set_title(f"Model {graph_type}")
    axs[n].set_ylabel(graph_type)
    axs[n].set_xlabel('epoch')
    axs[n].legend(['train', 'validation'], loc='upper left')

model.save('./minst_model')
tfjs.converters.save_keras_model(model, "./modeljs")

plt.show()