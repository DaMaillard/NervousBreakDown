# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 13:15:20 2016

@author: antoinemovschin

Copied from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py

"""

from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.cross_validation import train_test_split
from data import load_train_data, load_test_data
from submission import submission

img_rows = 64
img_cols = 80

smooth = 1.


def K_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ( (2. * intersection + smooth) / 
             (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) )

def np_dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return ( (2. * intersection + smooth) / 
             (np.sum(y_true_f) + np.sum(y_pred_f) + smooth) )

def dice_coef(y_true, y_pred):
    return (K_dice_coef(y_true, y_pred))
    

def dice_coef_loss(y_true, y_pred):
    return (1-dice_coef(y_true, y_pred))



def get_unet(dropout):
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    if dropout:
        pool1d = Dropout(dropout)(pool1)
    else:
        pool1d = pool1

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1d)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    if dropout:
        pool2d = Dropout(dropout)(pool2)
    else:
        pool2d = pool2

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2d)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    if dropout:
        pool3d = Dropout(dropout)(pool3)
    else:
        pool3d = pool3

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3d)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    if dropout:
        pool4d = Dropout(dropout)(pool4)
    else:
        pool4d = pool4

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4d)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # Todo: tester optimizer SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

    

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p



def train_unet(data_path, save_path, basename="", weight_load = "", 
                      valid_size=.2, batch_size = 32, nb_epoch = 10, dropout = .25):
                          
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train, imgs_patient_train = load_train_data(data_path)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    random_state = 51
    X_train, X_valid, Y_train, Y_valid = train_test_split(imgs_train, imgs_mask_train, 
                                                          test_size=valid_size, random_state=random_state)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet(dropout)
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    
    # pas à faire tout le temps...
    if weight_load:
        model.load_weights(weight_load)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
              verbose=1, shuffle=True, callbacks=[model_checkpoint])
    
    
    Y_train_pred = model.predict(X_train, verbose=1)
    if (valid_size > 0):
        Y_valid_pred = model.predict(X_valid, verbose=1)

    
    score_training = np_dice_coef(Y_train, Y_train_pred)
    if (valid_size > 0):
        score_validation = np_dice_coef(Y_valid, Y_valid_pred)
    print('Score on training set: dice_coef = ', score_training)
    if (valid_size > 0):
        print('Score on validation set: dice_coef = ', score_validation)
    
    weight_save = save_path + '/unet' + basename + '.hdf5'
    if weight_save:
        model.save_weights(weight_save)
    
    return mean, std


def predict_unet(data_path, save_path, basename="", weight_load = "", train_mean=0, train_std=1):
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data(data_path)
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= train_mean
    imgs_test /= train_std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(weight_load)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    mask_filename = save_path + 'imgs_mask_test' + basename + '.npy'
    np.save(mask_filename, imgs_mask_test)
    
    submission_filename = save_path + '/submission' + basename + '.csv'
    submission(data_path, submission_filename, mask_filename, .6)
    
    
def train_and_predict(data_path, save_path, basename="", weight_load = "", 
                      valid_size=.2, batch_size = 32, nb_epoch = 10, dropout = .25):
                          
    train_mean, train_std = train_unet(data_path, save_path, basename, weight_load, 
                                       valid_size, batch_size, nb_epoch, dropout)
    predict_unet(data_path, save_path, basename, weight_load, train_mean, train_std)
                          

def predict_test(weight_file, data_path, dropout):
    # load and process data
    imgs_train = np.load(data_path + '/imgs_train.npy')
    imgs_train = preprocess(imgs_train)
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_test = np.load(data_path + '/imgs_test.npy')
    imgs_test = preprocess(imgs_test)
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    model = get_unet(dropout)
    model.load_weights(weight_file)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save(data_path + '/imgs_mask_test.npy', imgs_mask_test)



if __name__ == '__main__':
#    train_and_predict('data_original', valid_size=.0, batch_size = 32, nb_epoch = 1, dropout=.0)
    load_folder = "backup_160718"
    save_folder = "backup_160719"
    
    weight_file_load = save_folder + '/unet_65.hdf5'
    weight_file_save = save_folder + '/unet_70.hdf5'
    train_and_predict('data_original', weight_load = weight_file_load, weight_save = weight_file_save, 
                      valid_size=0., batch_size = 16, nb_epoch = 4, dropout = 0.)
    
    weight_file_load = weight_file_save
    weight_file_save = save_folder + '/unet_75.hdf5'
    train_and_predict('data_original', weight_load = weight_file_load, weight_save = weight_file_save, 
                      valid_size=0., batch_size = 16, nb_epoch = 5, dropout = 0.)

    weight_file_load = weight_file_save
    weight_file_save = save_folder + '/unet_80.hdf5'
    train_and_predict('data_original', weight_load = weight_file_load, weight_save = weight_file_save, 
                      valid_size=0., batch_size = 16, nb_epoch = 5, dropout = 0.)

    weight_file_load = weight_file_save
    weight_file_save = save_folder + '/unet_85.hdf5'
    train_and_predict('data_original', weight_load = weight_file_load, weight_save = weight_file_save, 
                      valid_size=0., batch_size = 16, nb_epoch = 5, dropout = 0.)

