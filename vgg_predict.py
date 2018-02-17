from __future__ import print_function
import keras
import numpy as np
import pandas as pd
import os

from vgg import cifar10vgg
import dataset

BATCH_SIZE = 100

if __name__ == '__main__':
    idb_test = dataset.image_db("../test")
    labels = dataset.cifar10_read_label("../trainLabels.csv")
    # idb_test.transform_label(lambda x: labels.i2n(x))

    name = 'cifar10vgg16_focus.h5py'
    model = cifar10vgg(train_path=name)

    df = pd.DataFrame(columns=['id', 'label'])
    # for i in range(0, idb_test.get_size('list'), BATCH_SIZE): # uncomment for full predict
    for i in range(0, BATCH_SIZE, BATCH_SIZE): # uncomment for test predict
        print(i)

        x_test, y_test = idb_test.get_batch(BATCH_SIZE, mode='list', cmap='rgb')
        x_test = x_test.astype('float32')

        y_pred = model.predict(x_test)
        # print(y_pred)
        # print(map(lambda x: labels.n2s(x), np.argmax(y_pred, 1)))

        for x, y in zip(y_test, map(lambda x: labels.n2s(x), np.argmax(y_pred, 1))):
            df = df.append({'id':x, 'label':y}, ignore_index=True)

        # y_test = keras.utils.to_categorical(y_test, 10)
        # residuals = np.argmax(y_pred,1)!=np.argmax(y_test,1)

        # loss = sum(residuals)/len(residuals)
        # print("the validation 0/1 loss is: ",loss)

    df.id = pd.to_numeric(df.id, errors='coerce')
    df = df.sort_values(['id'], ascending=[1])
    print(df)
    df.to_csv('pred.csv', index=False)