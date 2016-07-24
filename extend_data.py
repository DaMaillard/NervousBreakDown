# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:14:02 2016

@author: amovschin
"""


from __future__ import print_function

import os
import numpy as np
import cv2
import random as rd
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from train import split_train_valid

def get_nb_images(data_path):
    images = os.listdir(data_path)
    total = int(len(images) / 2)
    return total
    

def get_indices(maxi, ratio):
    nb = int(round(maxi * ratio))
    return rd.sample(list(np.arange(maxi)), nb)

def get_all_names_without_mask(data_path):
    images = os.listdir(data_path)
    im_names = list()
    for image_name in images:
        if 'mask' in image_name:
            continue
        if not ('.tif' in image_name):
            continue
        im_names.append(image_name)
    return im_names

def get_subset_names_wo_mask(data_path, indices):
    all_names = get_all_names_without_mask(data_path)
    subset_names = [all_names[n] for n in indices]
    return subset_names


# Function to distort image
# By  Bruno G. do Amaral : 
# https://www.kaggle.com/bguberfain/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


    
def create_rotated_imgs(data_path, ratio, max_angle):
    # get the names of images to be processed
    nb_im = get_nb_images(data_path)
    ind_rotated_imgs = get_indices(nb_im, ratio)
    rotated_names = get_subset_names_wo_mask(data_path, ind_rotated_imgs)
    
    # process images
    for image_name in rotated_names:
        # get image and mask
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        image_rows, image_cols = img.shape
        
        # random rotation of the image and its mask
        angle = rd.randint(-max_angle, max_angle)
        Rotation_matrix = cv2.getRotationMatrix2D((image_cols/2,image_rows/2),angle,1)
        rotated_im = cv2.warpAffine(img,R,(image_cols, image_rows))
        rotated_im_mask = cv2.warpAffine(img_mask,Rotation_matrix,(image_cols, image_rows))
        
        # filenames for the rotated images
        rot_name = image_name.split('.')[0] + '_rot.tif'
        rot_mask_name = image_name.split('.')[0] + '_rot_mask.tif'
        
        # saving the rotated image and mask
        cv2.imwrite(os.path.join(data_path, rot_name), rotated_im)
        cv2.imwrite(os.path.join(data_path, rot_mask_name), rotated_im_mask)
        


def create_distorted_imgs(data_path, ratio, alpha, sigma, alpha_affine):
    # get the names of images to be processed
    nb_im = get_nb_images(data_path)
    ind_rotated_imgs = get_indices(nb_im, ratio)
    rotated_names = get_subset_names_wo_mask(data_path, ind_rotated_imgs)
    
    # process images
    for image_name in rotated_names:
        # get image and mask
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        image_rows, image_cols = img.shape
        im_merge = np.concatenate((img[...,None], img_mask[...,None]), axis=2)
        
        # Apply transformation on image
        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * alpha, 
                                       im_merge.shape[1] * sigma, im_merge.shape[1] * alpha_affine)

        # Split image and mask
        im_t = im_merge_t[...,0]
        im_mask_t = im_merge_t[...,1]
       
        # filenames for the rotated images
        rot_name = image_name.split('.')[0] + '_dist.tif'
        rot_mask_name = image_name.split('.')[0] + '_dist_mask.tif'
        
        # saving the rotated image and mask
        cv2.imwrite(os.path.join(data_path, rot_name), im_t)
        cv2.imwrite(os.path.join(data_path, rot_mask_name), im_mask_t)

if __name__ == '__main__':
    rd.seed(1234)
    #create_rotated_imgs('data_extended/train', .2, 180)
    create_distorted_imgs('data_augmented/train', .2, 2, 0.08, 0.08)
    

