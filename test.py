#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:20:52 2021

@author: zhuoyin94
"""

import os
import tensorflow as tf
import multiprocessing as mp

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

PATH = './data/Test_debug/Public_test_new/'

all_image_paths = os.listdir(PATH)
all_image_paths = [PATH + item for item in all_image_paths]

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=mp.cpu_count())

for img in image_ds.take(4):
    print(img.shape)
