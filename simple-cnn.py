import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist, cifar10

### MNIST ###
# ((X_train, Y_train), (X_test, Y_test)), nlabel = mnist.load_data(), 10

### CIFAR-10 ###
((X_train, Y_train), (X_test, Y_test)), nlabel = cifar10.load_data(), 100
X_train = np.array(map(rgb2hsv, X_train))
X_test = np.array(map(rgb2hsv, X_test))

print "train:", X_train.shape, Y_train.shape
print "test :", X_test.shape, Y_test.shape

### preprocess X ###
if len(X_train.shape) == 3:
	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

### preprocess y ###
if len(Y_train.shape) == 1:
	Y_train = Y_train.reshape((-1, 1))
	Y_test = Y_test.reshape((-1, 1))
Y_train = np_utils.to_categorical(Y_train, nlabel)
Y_test = np_utils.to_categorical(Y_test, nlabel)

print "new train:", X_train.shape, Y_train.shape
print "new test :", X_test.shape, Y_test.shape

### model architecture ###
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nlabel, activation='softmax'))

### compilation: loss + opt + stat ###
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

### fit ###
model.fit(X_train, Y_train, batch_size=100, epochs=50, verbose=1)

### evaluate ###
score = model.evaluate(X_test, Y_test, verbose=1)
print "score:", score