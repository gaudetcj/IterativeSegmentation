# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:03:57 2016

@author: Chase
"""

from keras.models import Model
from keras.layers import Input, merge
from keras.layers import Convolution2D
from keras.optimizers import Adam


def get_iterative_model(img_rows, img_cols):
    # the heatmap from CNN
    input1 = Input(shape=(1, img_rows, img_cols))
    
    # the image heatmap was generated from
    input2 = Input(shape=(3, img_rows, img_cols))
    
    # these are shared layer declarations
    convh1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')
    convh2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')
    
    convi1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')
    
    convmlp1 = Convolution2D(64, 1, 1, activation='relu')
    convmlp2 = Convolution2D(64, 1, 1, activation='relu')
    convmlp3 = Convolution2D(1, 1, 1, activation='relu')
    
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