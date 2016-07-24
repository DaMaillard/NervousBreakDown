# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 21:28:18 2016

@author: antoinemovschin
"""

from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.core import Dense, Activation, Flatten 
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from data import load_train_data, load_test_data
from train import preprocess, load_data, split_train_valid, img_rows, img_cols
from submission import submission
import scipy as sp



def logloss(Y_true, Y_pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, Y_pred)
    pred = sp.minimum(1-epsilon, Y_pred)
    ll = sum(Y_true*sp.log(pred) + sp.subtract(1,Y_true)*sp.log(sp.subtract(1,Y_pred)))
    ll = ll * -1.0/len(Y_true)
    return ll

def is_there_mask(imgs_mask):
    out = []
    for i in range(len(imgs_mask)):
        if imgs_mask[i,0].sum() == 0:
            out.append(0)
        else:
            out.append(1)
    
    return np.array(out)

def get_unet_exist_mask_zfturbo(img_rows, img_cols, dropout = .15):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, border_mode='same', init='he_normal',
                            input_shape=(1, img_rows, img_cols)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Convolution2D(8, 3, 3, border_mode='same', init='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(2))
    model.add(Activation('softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def get_unet_exist_mask_mine(dropout):
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

    flat11 = Flatten()(conv10)
    dense12 = Dense(2)(flat11)
    out = Activation('sigmoid')(dense12)

    model = Model(input=inputs, output=out)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    return model

def get_unet_exist_mask(dropout):
#    model = get_unet_exist_mask_mine(dropout)
    model = get_unet_exist_mask_zfturbo(img_rows, img_cols, dropout)
    return model
    
def preprocess_exist_mask(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)

    imgs_p = imgs_p.astype('float32')
    imgs_p /= 255.  # scale masks to [0, 1]
    
    imgs_p = is_there_mask(imgs_p)
    imgs_p = np_utils.to_categorical(imgs_p, 2)    
    return imgs_p

    
def train_unet_exist_mask(imgs_train, imgs_mask_train, imgs_patient_train, 
                          weight_load = "", save_path='', basename="",  
                          valid_size=.2, batch_size = 32, nb_epoch = 10, dropout = .25):
                          
    imgs_exist_mask_train = preprocess_exist_mask(imgs_mask_train)

    seed=1234
    train_ind, val_ind = split_train_valid(imgs_train, imgs_patient_train, valid_size, seed)

    X_train = imgs_train[train_ind]
    X_valid = imgs_train[val_ind]
    Y_train = imgs_exist_mask_train[train_ind]
    Y_valid = imgs_exist_mask_train[val_ind]
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    model = get_unet_exist_mask(dropout)

    model_checkpoint = ModelCheckpoint('unet_exist.hdf5', monitor='loss', save_best_only=True)
    
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

    
    score_training = logloss(Y_train[:,1], Y_train_pred[:,1])
    if (valid_size > 0):
        score_validation = logloss(Y_valid[:,1], Y_valid_pred[:,1])
    print('Score on training set: logloss = ', score_training)
    if (valid_size > 0):
        print('Score on validation set: logloss = ', score_validation)
    
    weight_save = save_path + '/unet_exist' + basename + '.hdf5'
    if weight_save:
        model.save_weights(weight_save, overwrite=True)

    
    
def predict_unet_exist_mask(imgs_test, save_path, basename="", weight_load = "", 
                            dropout = 0.15, threshold = .5):

    model = get_unet_exist_mask(dropout)
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(weight_load)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_exist_mask_test = model.predict(imgs_test, verbose=1)
    imgs_exist_mask_test = imgs_exist_mask_test[:,1]
    imgs_exist_mask_test = (imgs_exist_mask_test > threshold).astype('float32')
    mask_filename = save_path + '/imgs_exist_mask_test' + basename + '.npy'
    np.save(mask_filename, imgs_exist_mask_test)
    
    print('on a trouvé ' + imgs_exist_mask_test.sum().astype(int).astype(str) + ' masques présents dans le test set')
    
    
def train_and_predict_exist_mask(imgs_train, imgs_mask_train, imgs_patient_train, imgs_test,
                                 data_path, save_path, basename="", weight_load = "", 
                                 valid_size=.2, batch_size = 32, nb_epoch = 10, dropout = .25, threshold=.5):
                          
    train_unet_exist_mask(imgs_train, imgs_mask_train, imgs_patient_train, 
                          weight_load, save_path, basename, 
                          valid_size, batch_size, nb_epoch, dropout)
                                       
    weights = save_path + '/unet_exist' + basename + '.hdf5'
    predict_unet_exist_mask(imgs_test, save_path, basename, weights, dropout, threshold)
             

if __name__ == '__main__':
    print('youhou')
#    data_path = 'data_original'
#    save_path = 'exist_mask_160721'
#    basename = '_16'
#    weight_load = ''
#    train_and_predict_exist_mask(data_path, save_path, basename, weight_load, 
#                      valid_size=0.2, batch_size = 16, nb_epoch = 16, dropout = 0.2, threshold = .2)
#
#    mask_filename = save_path + '/imgs_exist_mask_test' + basename + '.npy'
#    pred_msk = np.load(mask_filename)
#    print(imgs_exist_mask_train[:,1])
#    print(pred_msk)
#    