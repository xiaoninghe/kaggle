import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_valid = tf.keras.utils.normalize(X_valid, axis=1)

reg_factor = 5e-5
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=l2(reg_factor)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=l2(reg_factor)))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=36)

val_loss, val_acc = model.evaluate(X_valid, y_valid)
print(val_loss, val_acc)

model.save('./minst_model')