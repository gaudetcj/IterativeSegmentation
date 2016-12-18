# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:31:46 2016

@author: Chase
"""

import cv2
import models
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping


img_rows = 80
img_cols = 100


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train
    
    
def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    return imgs_test
    
    
def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    print('-'*30)
    print('Creating and compiling CNN model...')
    print('-'*30)
    model = models.get_cnn_model(img_rows, img_cols) 
    
    print('-'*30)
    print('Loading saved weights into CNN model...')
    print('-'*30)
    model.load_weights('weights_cnn.hdf5')

    print('-'*30)
    print('Predicting masks on test data using CNN model...')
    print('-'*30)
    imgs_cnn_mask = model.predict(imgs_train, verbose=1)
    
    print('-'*30)
    print('Creating and compiling iterative model...')
    print('-'*30)
    model = models.get_iterative_model(img_rows, img_cols)
    
    print('-'*30)
    print('Begin training...')
    print('-'*30)
    callbacks = [
        EarlyStopping(monitor='loss', patience=5, verbose=0),
        ModelCheckpoint('weights_iter.hdf5', monitor='loss', save_best_only=True)
    ]
    model.fit([imgs_cnn_mask, imgs_train], imgs_mask_train, batch_size=4, nb_epoch=100, verbose=1, shuffle=True,
              callbacks=callbacks)
              
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights_iter.hdf5')

    print('-'*30)
    print('Predicting masks on test data using iterative model...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()