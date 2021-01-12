# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

y = np.array(train_data.pop("label"))
X = np.array(train_data)
X_test = np.array(test_data)

X = tf.keras.utils.normalize(X, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

reg_factor = 1e-4
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42000, 784)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=l2(reg_factor)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=l2(reg_factor)),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)

submission = pd.DataFrame({
    'ImageId': range(1, len(X_test) + 1),
    'Label': pred,
})
submission.to_csv('submission.csv', index=False)

model.save('./minst_model')

