# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 18:26:12 2016

@author: antoinemovschin
"""



from data import load_train_data, load_test_data
from submission import submission
from train import load_data, train_and_predict
from train_exist_mask import train_and_predict_exist_mask
import numpy as np

def generate_basename(batch_size, nb_epoch_mask, nb_epoch_mask_exist, 
                      dropout_mask, dropout_exist_mask):
    basename = '_bs' + str(batch_size) \
                + '_em' + str(nb_epoch_mask) \
                + '_eme' + str(nb_epoch_mask_exist) \
                + '_dm' + str(dropout_mask) \
                + '_dem' + str(dropout_exist_mask)
    return basename
    

data_path = 'data_original'
save_path = 'backup_160723'

batch_size = 16
nb_epoch_mask = 5
nb_epoch_mask_exist = 5
dropout_mask = .2
dropout_exist_mask = .15
 
basename = generate_basename(batch_size, nb_epoch_mask, nb_epoch_mask_exist, 
                             dropout_mask, dropout_exist_mask)
            
weight_load = ''
mask_filename = ''
mask_exist_filename = ''
weight_mask_filename = ''
weight_mask_exist_filename = ''

imgs_train, imgs_mask_train, imgs_patient_train, imgs_test = load_data(data_path)

for n in range(10):
    
    nb_epoch_mask_total = (n+1)*nb_epoch_mask
    
    nb_epoch_mask_exist_total = (n+1)*nb_epoch_mask_exist
    basename = generate_basename(batch_size, nb_epoch_mask_total, nb_epoch_mask_exist_total, 
                             dropout_mask, dropout_exist_mask)
    
    weight_load = weight_mask_filename
    train_and_predict(imgs_train, imgs_mask_train, imgs_patient_train, imgs_test, 
                      data_path, save_path, basename, weight_load, 
                      valid_size=0.2, batch_size = batch_size, nb_epoch = nb_epoch_mask, 
                      dropout = dropout_mask)
                          
    weight_load = weight_mask_exist_filename    
    train_and_predict_exist_mask(imgs_train, imgs_mask_train, imgs_patient_train, imgs_test, 
                                 data_path, save_path, basename, weight_load, 
                                 valid_size=0.2, batch_size = batch_size, nb_epoch = nb_epoch_mask_exist, 
                                 dropout = dropout_exist_mask, threshold = .2)
                            
    mask_exist_filename = save_path + '/imgs_exist_mask_test' + basename + '.npy'    
    mask_filename = save_path + '/imgs_mask_test' + basename + '.npy'    
    weight_mask_filename = save_path + '/unet' + basename + '.hdf5'
    weight_mask_exist_filename = save_path + '/unet_exist' + basename + '.hdf5'
    
    
    imgs_mask_test = np.load(mask_filename)
    imgs_mask_exist_test = np.load(mask_exist_filename)
    
    for n in range(len(imgs_mask_exist_test)):
        if (imgs_mask_exist_test[n] == 0):
            imgs_mask_test[n,0] = 0
    
    total_mask_filename = save_path + '/imgs_mask_test_total_epoch' + str((n+1)*nb_epoch_mask) + '.npy'
    np.save(total_mask_filename, imgs_mask_test)
    
    submisison_filename = save_path + '/submission_total_epoch' + str((n+1)*nb_epoch_mask) + '.csv'
    submission(data_path, submisison_filename, total_mask_filename, .65)
    