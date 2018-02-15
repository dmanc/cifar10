from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

import dataset

import os

class cifar10vgg:
    def __init__(self, train_path=None, train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        self.train_path = train_path
        if train_path and os.path.isfile(train_path):
            self.model.load_weights(train_path)

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model,x_train,x_test,y_train,y_test):

        #training parameters
        batch_size = 128
        maxepoches = 100
        learning_rate = 0.1
        lr_decay = 1e-6

        # The data, shuffled and split between train and test sets:
        '''
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        '''

        lrf = learning_rate


        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.
        gen = datagen.flow(x_train, y_train, batch_size=batch_size)
        for epoch in range(1,maxepoches):

            if epoch%25==0 and epoch>0:
                lrf/=2
                sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            historytemp = model.fit_generator(gen,
                                steps_per_epoch=x_train.shape[0] // batch_size,
                                epochs=epoch,
                                validation_data=(x_test, y_test),initial_epoch=epoch-1)
        model.save_weights(self.train_path)
        return model

import skimage
def rgbfocus(vae, sess, imgs):
    gimgs = []
    for img in imgs:
        gimgs.append(skimage.color.rgb2grey(img/255.0))
        gimgs[-1] = gimgs[-1].reshape((gimgs[-1].shape[0], gimgs[-1].shape[1], 1))
    gimgs = np.array(gimgs)

    focus = []
    for i in range(0, imgs.shape[0], 1000):
        print(i)
        focus.extend(vae.get_focus(sess, gimgs[i:i+1000]))
    focus = np.array(focus)
    del gimgs

    fimgs = []
    for img, f in zip(imgs, focus):
        fimg = np.zeros((img.shape[0], img.shape[1], 3))
        for k in range(3):
            fimg[:, :, k] = img[:, :, k] * f[:, :, 0]
        fimgs.append(255.0 * fimg / np.max(fimg))
    fimgs = np.array(fimgs)
    return fimgs

if __name__ == '__main__':
    ### from kaggle
    labels = dataset.cifar10_read_label("../trainLabels.csv")
    idb = dataset.image_db("../train")
    idb.transform_label(lambda x: labels.i2n(x))
    x_train, y_train = idb.get_batch(idb.get_size('list'), mode='list', cmap='rgb')
    x_test, y_test = idb.get_batch(idb.get_size('test'), mode='test', cmap='rgb')

    ### TODO: get small batches + classify + put in csv
    #idb_test = dataset.image_db("../test")
    idb_test = dataset.image_db("../train")

    #x_test, y_test = idb_test.get_batch(idb_test.get_size('test'), mode='test', cmap='rgb')
    x_test, y_test = idb.get_batch(idb.get_size('test'), mode='test', cmap='rgb')

    ### pull all dataset
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    ### vae attention ###
    import tensorflow as tf
    from vae_build import Vae
    sess = tf.InteractiveSession()
    vae = Vae(name="2d_vae_0", in_size=[32, 32], \
            cnvlf=[32, 32, 16], kernel_size=[[5,5], [5,5], [3, 3]], strides=[2, 2,2], \
            d_cnvlf=[32, 32, 1], d_kernel_size=[[5,5], [5,5], [3, 3]],
            d_strides=[2, 2, 2], \
            nlatent=16, lr=0.001)
    sess.run(vae.init)
    vae.restore(sess)

    ### dataset with focus ###
    # x_train_f = rgbfocus(vae, sess, x_train)
    # x_test_f = rgbfocus(vae, sess, x_test)

    """
    import matplotlib.pyplot as plt
    pc, pn = 20, 5
    fig, axs = plt.subplots(ncols=2, nrows=pn)
    for i in range(pn):
        axs[i][0].imshow(x_train[i+pc]/255.0)
        axs[i][1].imshow(x_train_f[i+pc]/255.0)
    plt.show()
    """

    # name = None
    # name = 'cifar10vgg16.h5'
    # name = 'cifar10vgg16_focus.h5'
    name = 'cifar10vgg16_focus.h5py'
    # model = cifar10vgg()
    model = cifar10vgg(train_path=name)
    model.train(model.model, x_train, x_test, y_train, y_test)

    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)
