# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 13:12:28 2016

@author: antoinemovschin

Copied from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/data.py

"""
from __future__ import print_function

import os
import numpy as np

import cv2


image_rows = 420
image_cols = 580


def create_train_data(data_path):
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_patient = np.ndarray((total, 1), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        patient = image_name.split('_')[0]
        imgs_patient[i] = patient
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path + '/imgs_train.npy', imgs)
    np.save(data_path + '/imgs_mask_train.npy', imgs_mask)
    np.save(data_path + '/imgs_patient.npy', imgs_patient)
    print('Saving to .npy files done.')


def load_train_data(data_path):
    imgs_train = np.load(data_path + '/imgs_train.npy')
    imgs_mask_train = np.load(data_path + '/imgs_mask_train.npy')
    imgs_patient_train = np.load(data_path + '/imgs_patient.npy')
    return imgs_train, imgs_mask_train, imgs_patient_train


def create_test_data(data_path):
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path + '/imgs_test.npy', imgs)
    np.save(data_path + '/imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data(data_path):
    imgs_test = np.load(data_path + '/imgs_test.npy')
    imgs_id = np.load(data_path + '/imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    data_path = 'data_original'
    create_train_data(data_path)
    create_test_data(data_path)
