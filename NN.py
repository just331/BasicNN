# Basic NN to teach the basics of deep learning

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
    ax.imshow(x_train[idx], cmap=plt.cm.binary)
    ax.set_title(str(y_train[idx]))

x_train = x_train.reshape(60000, 784).astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, verbose=1)

_, (x_test_, y_test_) = tf.keras.datasets.mnist.load_data()
x_test = x_test_.reshape(10000, 784).astype('float32') / 255
y_test = to_categorical(y_test_, num_classes=10)

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)

image = 5
_ = plt.imshow(x_test_[image], cmap=plt.cm.binary)
prediction = model.predict(x_test)
print("Model prediction: ", np.argmax(prediction[image]))
