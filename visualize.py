# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 08:32:42 2016

@author: Chase
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


image_rows = 420
image_cols = 580


def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows))
    return img


def visualize_results():
    images  = np.load('imgs_test.npy')
    results_cnn = np.load('imgs_mask_test.npy')
    results_iter = np.load('imgs_mask_iter_test.npy')
    
    total = results_cnn.shape[0]
    for i in range(total):
        image = images[i,0]
        image = cv2.resize(image, (image_cols, image_rows))
        result_cnn = results_cnn[i,0]
        result_cnn = prep(result_cnn)
        result_iter = results_iter[i,0]
        result_iter = prep(result_iter)
        
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.imshow(result_cnn, alpha=0.75)
        
        plt.subplot(1,2,2)
        plt.imshow(image)
        plt.imshow(result_iter, alpha=0.75)
        plt.show()


if __name__ == '__main__':
    visualize_results()