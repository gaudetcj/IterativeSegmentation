# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:03:57 2016

@author: Chase
"""

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, merge, SpatialDropout2D
from keras.layers import Convolution2D, AveragePooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam


def get_cnn_model(img_rows, img_cols):
    input = Input(shape=(1, img_rows, img_cols))
    
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(input)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    pool1 = AveragePooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    pool2 = AveragePooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)
    
    comb1 = merge([conv2, UpSampling2D(size=(2,2))(conv3)], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(comb1)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)
    conv4 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)
    
    comb2 = merge([conv1, UpSampling2D(size=(2,2))(conv4)], mode='concat', concat_axis=1)
    conv5 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(comb2)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)
    conv5 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)
    
    output = Convolution2D(1, 1, 1, activation='sigmoid')(conv5)

    model = Model(input=input, output=output)
    model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy')
    return model


def get_iterative_model(img_rows, img_cols):
    # the heatmap from CNN
    input1 = Input(shape=(1, img_rows, img_cols))
    
    # the image heatmap was generated from (change 1 to 3 for color images)
    input2 = Input(shape=(1, img_rows, img_cols))
    
    # these are shared layer declarations
    convh1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')
    convh2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')
    
    convi1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')
    
    convmlp1 = Convolution2D(32, 1, 1, activation='relu')
    convmlp2 = Convolution2D(32, 1, 1, activation='relu')
    convmlp3 = Convolution2D(1, 1, 1)
    
    # now the actual model
    upperconv1 = convh1(input1)
    lowerconv1 = convi1(input2)
    
    concat1 = merge([upperconv1, lowerconv1], mode='concat', concat_axis=1)
    upperconv2 = convh2(concat1)
    
    mlp11 = convmlp1(upperconv2)
    mlp12 = convmlp2(mlp11)
    mlp13 = convmlp3(mlp12)
    
    add1 = merge([input1, mlp13], mode='sum')
    
    
    upperconv2 = convh1(add1)
    lowerconv2 = convi1(input2)
    
    concat2 = merge([upperconv2, lowerconv2], mode='concat', concat_axis=1)
    upperconv3 = convh2(concat2)
    
    mlp21 = convmlp1(upperconv3)
    mlp22 = convmlp2(mlp21)
    mlp23 = convmlp3(mlp22)
    
    add2 = merge([add1, mlp23], mode='sum')
    
    
    upperconv3 = convh1(add2)
    lowerconv3 = convi1(input2)
    
    concat3 = merge([upperconv3, lowerconv3], mode='concat', concat_axis=1)
    upperconv4 = convh2(concat3)
    
    mlp31 = convmlp1(upperconv4)
    mlp32 = convmlp2(mlp31)
    mlp33 = convmlp3(mlp32)
    
    add3 = merge([add2, mlp33], mode='sum')
    
    
    upperconv4 = convh1(add3)
    lowerconv4 = convi1(input2)
    
    concat4 = merge([upperconv4, lowerconv4], mode='concat', concat_axis=1)
    upperconv5 = convh2(concat4)
    
    mlp41 = convmlp1(upperconv5)
    mlp42 = convmlp2(mlp41)
    mlp43 = convmlp3(mlp42)
    
    add4 = merge([add3, mlp43], mode='sum')
    
    
    upperconv5 = convh1(add4)
    lowerconv5 = convi1(input2)
    
    concat5 = merge([upperconv5, lowerconv5], mode='concat', concat_axis=1)
    upperconv6 = convh2(concat5)
    
    mlp51 = convmlp1(upperconv6)
    mlp52 = convmlp2(mlp51)
    mlp53 = convmlp3(mlp52)
    
    add5 = merge([add4, mlp53], mode='sum')
    
    
    model = Model(input=[input1, input2], output=add5)
    model.compile(Adam(lr=3e-4), 'binary_crossentropy')
    
    return model