from __future__ import division
from test import *
import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import random
import h5py

# BASE_PATH='/home/pranjal-artivatic/Desktop'

epochs = 20

train_data_path = '/home/artivatic/Desktop/front_back_car/Training'
validation_data_path = '/home/artivatic/Desktop/front_back_car/Validation'
BASE_PATH = '/home/artivatic'

"""
Parameters
"""
img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 1000
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 4
lr = 0.0004

def train_model_3():
    model = Sequential()
    model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')


    validation_generator = test_datagen.flow_from_directory(
        validation_data_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    """
    Tensorboard log
    """
    log_dir = './tf-log/'
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cbks = [tb_cb]

    model.fit_generator(
        train_generator,
        samples_per_epoch=samples_per_epoch,
        epochs=epochs,
        validation_data=validation_generator)



    target_dir = BASE_PATH+  '/models/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)



    model.save(target_dir+'/model4.h5')
    model.save_weights(target_dir+'/weights4.h5')




def ImageVerification(args,file):
    """

    :param args:
    :param file:
    :return:
    """

    model_path=BASE_PATH + '/module/model3.h5'
    test_model = load_model(model_path)
    img = load_img(file,False,target_size=(150,150))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print '-------------'
    preds = test_model.predict_classes(x)
    prob = test_model.predict_proba(x)
    pr=test_model.predict(x)
    print(preds, prob,pr,'in test_model_3')
    return preds

# if __name__=='__main__':
#     train_model_3()

