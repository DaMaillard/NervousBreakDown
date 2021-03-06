# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 22:59:54 2016

@author: antoinemovschin

Copied from https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/submission.py

"""

from __future__ import print_function

import numpy as np
import cv2
from data import image_cols, image_rows


def prep(img, threshold):
    img = img.astype('float32')
    img = cv2.threshold(img, threshold, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission(data_path, submission_filename, mask_test_filename, threshold):
    from data import load_test_data
    imgs_test, imgs_id_test = load_test_data(data_path)
    imgs_test = np.load(mask_test_filename)

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, 0]
        img = prep(img, threshold)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = submission_filename

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


if __name__ == '__main__':
    submission('data_original', 'submission_0.csv', 'imgs_mask_test.npy', .6)