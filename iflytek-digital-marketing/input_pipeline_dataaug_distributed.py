#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202108201031
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(input_pipeline.py)构建数据读取与预处理的pipline，并训练神经网络模型。
其中本模块采用Mixup，Mixmatch，Cutmix等数据增强策略，并采用分布式的策略进行训练。
'''

import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from dingtalk_remote_monitor import RemoteMonitorDingTalk
from models import build_model_resnet50_v2, build_model_resnet101_v2
from utils import LearningRateWarmUpCosineDecayScheduler, LoadSave

GLOBAL_RANDOM_SEED = 7555
# np.random.seed(GLOBAL_RANDOM_SEED)
# tf.random.set_seed(GLOBAL_RANDOM_SEED)

TASK_NAME = 'iflytek_2021_digital_marketing'
GPU_ID = 'distributed'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 限制Tensorflow只使用GPU ID编号的GPU
        tf.config.experimental.set_visible_devices(gpus[1, 2], 'GPU')

        # 限制Tensorflow不占用所有显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

# ----------------------------------------------------------------------------

def build_efficentnet_model(verbose=False, is_compile=True, **kwargs):
    '''构造基于imagenet预训练的ResNetV2的模型，并返回编译过的模型。'''

    # 解析preprocessing与model的参数
    # ---------------------
    input_shape = kwargs.pop('input_shape', (None, 224, 224))
    n_classes = kwargs.pop('n_classes', 1000)

    model_name = kwargs.pop('model_name', 'EfficentNetB0')
    model_lr = kwargs.pop('model_lr', 0.01)
    model_label_smoothing = kwargs.pop('model_label_smoothing', 0.1)

    # 依据关键字，构建模型
    # ---------------------
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential()

        if 'B0' in model_name:
            model_tmp = tf.keras.applications.EfficientNetB0
        elif 'B1' in model_name:
            model_tmp = tf.keras.applications.EfficientNetB1
        elif 'B2' in model_name:
            model_tmp = tf.keras.applications.EfficientNetB2
        elif 'B3' in model_name:
            model_tmp = tf.keras.applications.EfficientNetB3
        elif 'B4' in model_name:
            model_tmp = tf.keras.applications.EfficientNetB4
        elif 'B5' in model_name:
            model_tmp = tf.keras.applications.EfficientNetB5
        elif 'B6' in model_name:
            model_tmp = tf.keras.applications.EfficientNetB6
        elif 'B7' in model_name:
            model_tmp = tf.keras.applications.EfficientNetB7

        model.add(
            model_tmp(
                input_shape=input_shape, 
                include_top=False,
                weights='imagenet',
                drop_connect_rate=0.4,
            )
        )
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            256, activation='relu',
        ))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

        # 编译模型
        # ---------------------
        if verbose:
            model.summary()

        if is_compile:
            model.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(
                    label_smoothing=model_label_smoothing),
                optimizer=Adam(model_lr),
                metrics=['acc'])

    return model


def load_preprocessing_img(image_size, stage):
    '''通过闭包实现参数化的Image Loading与TTA数据增强。'''
    if stage not in ['train', 'valid', 'test']:
        raise ValueError('stage must be either train, valid or test !')

    if stage is 'train' or stage is 'test':
        def load_img(path=None):
            image = tf.io.read_file(path)
            image = tf.cond(
                tf.image.is_jpeg(image),
                lambda: tf.image.decode_jpeg(image, channels=3),
                lambda: tf.image.decode_gif(image)[0])

            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, 0.3)

            # image = tf.image.random_flip_left_right(image)
            # image = tf.image.random_flip_up_down(image)

            image = tf.image.resize(image, image_size)
            return image
    else:
        def load_img(path=None):
            image = tf.io.read_file(path)
            image = tf.cond(
                tf.image.is_jpeg(image),
                lambda: tf.image.decode_jpeg(image, channels=3),
                lambda: tf.image.decode_gif(image)[0])

            image = tf.image.resize(image, image_size)
            return image

    return load_img