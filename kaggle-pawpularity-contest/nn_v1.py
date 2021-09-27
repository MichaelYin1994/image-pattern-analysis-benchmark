#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202109271509
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(input_pipeline_dataaug.py)构建数据读取与预处理的pipline，并训练神经网络模型。
其中本模块采用Mixup，MixMatch等数据增强策略。
'''

import gc
import json
import multiprocessing as mp
import os
import urllib.request
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_batch_ops import batch
from tqdm import tqdm

GLOBAL_RANDOM_SEED = 7555
# np.random.seed(GLOBAL_RANDOM_SEED)
# tf.random.set_seed(GLOBAL_RANDOM_SEED)

TASK_NAME = 'kaggle_pawpularity_contest'
GPU_ID = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 限制Tensorflow只使用GPU ID编号的GPU
        tf.config.experimental.set_visible_devices(gpus[GPU_ID], 'GPU')

        # 限制Tensorflow不占用所有显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)


def send_msg_to_dingtalk(info_text, is_send_msg=False, is_print_msg=True):
    '''发送info_text给指定API_URL的钉钉机器人。'''
    if is_send_msg:
        API_URL = 'https://oapi.dingtalk.com/robot/send?access_token=d1b2a29b2ae62bc709693c02921ed097c621bc33e5963e9e0a5d5adf5eac10c1'

        # HTTP Head信息
        header = {
            'Content-Type': 'application/json',
            'Charset': 'UTF-8' }

        # 组装为json
        my_data = {
            'msgtype': 'markdown',
            'markdown': {'title': '[INFO]Neural Network at: {}'.format(datetime.now()),
                         'text': info_text},
            'at': {'isAtAll': False}}

        # 发送消息
        data_send = json.dumps(my_data)
        data_send = data_send.encode('utf-8')

        try:
            request = urllib.request.Request(url=API_URL, data=data_send, headers=header)
            opener = urllib.request.urlopen(request)
            opener.read()
        except:
            # 若无网络链接，则不执行操作
            pass

    if is_print_msg:
        print(info_text)


def build_efficentnet_model(verbose=False, is_compile=True, **kwargs):
    '''构造基于imagenet预训练的EfficentNet的模型，并返回编译过的模型。'''

    # 解析preprocessing与model的参数
    # *******************
    input_img_shape = kwargs.pop('input_img_shape', (None, 224, 224))
    input_dense_shape = kwargs.pop('input_dense_shape', (None, 12))

    model_name = kwargs.pop('model_name', 'EfficentNetB0')
    model_lr = kwargs.pop('model_lr', 0.01)

    # 依据关键字，构建模型
    # *******************
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

    model_img_embedding = model_tmp(
        input_shape=input_img_shape,
        include_top=False,
        weights='imagenet',
        drop_connect_rate=0.4,
    )
    layer_img_embedding_output = model_img_embedding.output

    layer_img_embedding_output = tf.keras.layers.GlobalAveragePooling2D()(layer_img_embedding_output)
    layer_img_embedding_output = tf.keras.layers.Flatten()(layer_img_embedding_output)

    layer_input_feats = tf.keras.layers.Input(
        shape=input_dense_shape, dtype=tf.float32
    )
    layer_dense_feats = tf.keras.layers.Dense(
        32, activation='relu',
    )(layer_input_feats)
    layer_dense_feats = tf.keras.layers.Dropout(0.4)(layer_dense_feats)

    # Dense layer
    # ----------
    layer_total = tf.keras.layers.concatenate(
        [layer_dense_feats, layer_img_embedding_output]
    )

    layer_total = tf.keras.layers.Dense(256, activation='relu')(layer_total)
    layer_total = tf.keras.layers.Dropout(0.4)(layer_total)
    layer_total = tf.keras.layers.BatchNormalization()(layer_total)

    layer_pred = tf.keras.layers.Dense(
        units=1, activation='linear', name='layer_pawpularity_score'
    )(layer_total)
    model = tf.keras.models.Model(
        [model_img_embedding.input, layer_input_feats], layer_pred
    )

    # 模型训练参数
    # *******************
    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(model_lr),
            metrics=['mean_squared_error']
    )

    return model


def load_preprocessing_img(image_size, stage):
    '''通过闭包实现参数化的Image Loading与TTA数据增强。'''
    if stage not in ['train', 'valid', 'test']:
        raise ValueError('stage must be either train, valid or test !')

    if stage == 'train' or stage == 'test':
        def load_img(path=None):
            '''载入并预处理单张图片'''

            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)

            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, 0.3)

            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

            image = tf.image.resize(image, image_size)

            return image
    else:
        def load_img(path=None):
            '''载入并预处理单张图片'''

            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)

            image = tf.image.resize(image, image_size)

            return image

    return load_img


def create_tf_dataset(img_path_list, dense_feats_list, map_fcn, label_list=None):
    '''生成数据pipeline对象，并构建性能参数'''

    # 组装数据集对象
    # ----------
    img_path_ds = tf.data.Dataset.from_tensor_slices(
        img_path_list
    )
    dense_feats_ds = tf.data.Dataset.from_tensor_slices(
        dense_feats_list
    )
    img_ds = img_path_ds.map(
        map_fcn, num_parallel_calls=mp.cpu_count()
    )
    concat_ds = tf.data.Dataset.zip(
        (img_ds, dense_feats_ds)
    )

    if label_list != None:
        label_ds = tf.data.Dataset.from_tensor_slices(
            label_list
        )
        concat_ds = tf.data.Dataset.zip(
            (concat_ds, label_ds)
        )

    return concat_ds


def set_tf_dataset_performance(
        ds=None,
        batch_size=16,
        prefetch_size=16,
        shuffle_buffer_size=None
    ):
    '''设定tf.data.Dataset对象的性能参数'''

    if shuffle_buffer_size:
        ds = ds.shuffle(
            shuffle_buffer_size).batch(batch_size).prefetch(prefetch_size)
    else:
        ds = ds.batch(batch_size).prefetch(prefetch_size)

    return ds


if __name__ == '__main__':
    # 全局化的参数列表
    # *******************

    # 全局化路径参数
    # ----------
    ONLINE = False

    if ONLINE:
        GLOBAL_DATA_DIR = '/kaggle/input/petfinder-pawpularity-score/'
        CACHE_DIR = '/kaggle/working/'
    else:
        GLOBAL_DATA_DIR = './kaggle/input/petfinder-pawpularity-score/'
        CACHE_DIR = './kaggle/working/'

    # 预处理/后处理相关参数
    # ----------
    N_FOLDS = 5
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 64
    NUM_EPOCHS = 128
    EARLY_STOP_ROUNDS = 6
    TTA_ROUNDS = 5

    MODEL_NAME = 'EfficentNetB0_dataaug_rtx3090'
    MODEL_LR = 0.0003
    MODEL_LR_DECAY_RATE = 0.7
    DECAY_LR_PATIENCE_ROUNDS = 5
    IS_RANDOM_VISUALIZING_PLOTS = False
    IS_SEND_MSG_TO_DINGTALK = False

    # 数据loading的path
    # ----------
    TRAIN_IMG_PATH = os.path.join(
        GLOBAL_DATA_DIR, 'train/'
    )
    TRAIN_META_FILE_NAME = os.path.join(
        GLOBAL_DATA_DIR, 'train.csv'
    )

    TEST_IMG_PATH = os.path.join(
        GLOBAL_DATA_DIR, 'test/'
    )
    TEST_META_FILE_NAME = os.path.join(
        GLOBAL_DATA_DIR, 'test.csv'
    )

    # 基础元信息准备
    # *******************
    train_df = pd.read_csv(TRAIN_META_FILE_NAME)
    test_df = pd.read_csv(TEST_META_FILE_NAME)

    train_df['Id'] = train_df['Id'].apply(
        lambda x: os.path.join(TRAIN_IMG_PATH, x + '.jpg')
    )
    test_df['Id'] = test_df['Id'].apply(
        lambda x: os.path.join(TEST_IMG_PATH, x + '.jpg')
    )
    train_target = train_df['Pawpularity'].values
    train_df.drop(['Pawpularity'], axis=1, inplace=True)

    n_train_samples, n_test_samples = len(train_df), len(test_df)

    # 构造模型训练关键参数
    # *******************
    send_msg_to_dingtalk(
        '\n++++++++++++++++++++++++++++', IS_SEND_MSG_TO_DINGTALK
    )
    INFO_TEXT = '[BEGIN][{}] {}, #Training: {}, #Testing: {}'.format(
        MODEL_NAME, str(datetime.now())[:-7], n_train_samples, n_test_samples
    )
    send_msg_to_dingtalk(
        info_text=INFO_TEXT, is_send_msg=IS_SEND_MSG_TO_DINGTALK
    )

    # 交叉验证策略
    # ----------
    folds = KFold(
        n_splits=N_FOLDS, shuffle=True, random_state=GLOBAL_RANDOM_SEED
    )
    fold_generator = folds.split(
        np.arange(0, len(train_df)), train_target
    )

    # 各种Callback函数
    # ----------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode="min",
            verbose=1, patience=EARLY_STOP_ROUNDS,
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=MODEL_LR_DECAY_RATE,
                patience=DECAY_LR_PATIENCE_ROUNDS,
                min_lr=0.000003),
    ]

    # 原始数据处理
    # ----------
    train_path_list = train_df['Id'].values.tolist()
    train_dense_list = train_df.drop(['Id'], axis=1).values.tolist()
    train_dense_array_list = [
        np.array(item, dtype=np.float32) for item in train_dense_list
    ]

    test_path_list = test_df['Id'].values.tolist()
    test_dense_list = test_df.drop(['Id'], axis=1).values.tolist()
    test_dense_array_list = [
        np.array(item, dtype=np.float32).reshape(1, -1) for item in test_dense_list
    ]

    # 交叉验证相关参数
    # ----------
    valid_score_df = np.zeros((N_FOLDS, 4))
    valid_pred_result_df = np.zeros((len(train_df), ))
    test_pred_result_list = []

    # 数据pipeline loading方法
    # ----------
    load_preprocess_train_image = load_preprocessing_img(
        image_size=IMAGE_SIZE, stage='train')
    load_preprocess_valid_image = load_preprocessing_img(
        image_size=IMAGE_SIZE, stage='valid')
    load_preprocess_test_image = load_preprocessing_img(
        image_size=IMAGE_SIZE, stage='test')

    # 测试数据Loading pipeline
    # ----------
    test_ds = create_tf_dataset(
        img_path_list=test_path_list,
        dense_feats_list=test_dense_list,
        map_fcn=load_preprocess_test_image
    )
    test_ds = set_tf_dataset_performance(
        ds=test_ds,
        batch_size=BATCH_SIZE,
        prefetch_size=int(8 * BATCH_SIZE)
    )

    # 模型训练部分
    # *******************
    send_msg_to_dingtalk(
        '[INFO][{}] {} training start...'.format(
            MODEL_NAME, str(datetime.now())[:-4]
        ), IS_SEND_MSG_TO_DINGTALK
    )
    print('==================================')
    for fold, (train_idx, valid_idx) in enumerate(fold_generator):
        # 清空内存结构中的图
        # ----------
        K.clear_session()
        gc.collect()

        # 构造数据的pipeline
        # ----------
        train_path_ds = [
            train_path_list[i] for i in train_idx
        ]
        train_dense_ds = [
            train_dense_array_list[i] for i in train_idx
        ]
        train_label_ds = [train_target[i] for i in train_idx]
        train_ds = create_tf_dataset(
            img_path_list=train_path_ds,
            dense_feats_list=train_dense_ds,
            label_list=train_label_ds,
            map_fcn=load_preprocess_train_image
        )
        train_ds = set_tf_dataset_performance(
            ds=train_ds,
            batch_size=BATCH_SIZE,
            prefetch_size=int(8 * BATCH_SIZE)
        )

        valid_path_ds = [
            train_path_list[i] for i in valid_idx
        ]
        valid_dense_ds = [
            train_dense_array_list[i] for i in valid_idx
        ]
        valid_label_ds = [train_target[i] for i in valid_idx]
        valid_ds = create_tf_dataset(
            img_path_list=valid_path_ds,
            dense_feats_list=valid_dense_ds,
            label_list=valid_label_ds,
            map_fcn=load_preprocess_valid_image
        )
        valid_ds = set_tf_dataset_performance(
            ds=valid_ds,
            batch_size=BATCH_SIZE,
            prefetch_size=int(8 * BATCH_SIZE)
        )

        # 构造模型
        # ----------
        model = build_efficentnet_model(
            input_img_shape=IMAGE_SIZE + (3, ),
            input_dense_shape=(12, ),
            network_type=MODEL_NAME,
            model_name=MODEL_NAME,
            model_lr=MODEL_LR,
        )

        # 训练模型
        # ----------
        history = model.fit(
            train_ds,
            epochs=NUM_EPOCHS,
            validation_data=valid_ds,
            callbacks=callbacks
        )

        # 保存预测结果
        # ----------
        test_pred_result_list.append(
            model.predict(test_ds).ravel().clip(1, 100)
        )
        val_pred_result = model.predict(valid_ds).ravel().clip(1, 100)
        valid_pred_result_df[valid_idx] = val_pred_result

        valid_mse = mean_squared_error(
            valid_label_ds.reshape(-1, 1), val_pred_result.reshape(-1, 1)
        )
        val_mae = mean_absolute_error(
            valid_label_ds.reshape(-1, 1), val_pred_result.reshape(-1, 1)
        )
        val_mape = mean_absolute_percentage_error(
            valid_label_ds.reshape(-1, 1), val_pred_result.reshape(-1, 1)
        )

        valid_score_df[fold, 0] = fold
        valid_score_df[fold, 1] = valid_mse
        valid_score_df[fold, 2] = val_mae
        valid_score_df[fold, 3] = val_mape

        # 发送预测结果
        # ----------
        INFO_TEXT = '-- [INFO][{}] {} folds {}({}), valid mse: {:.5f}, mae {:5f}, mape: {:.5f}'.format(
            MODEL_NAME, str(datetime.now())[:-4], fold+1, N_FOLDS,
            valid_score_df[fold, 1],
            valid_score_df[fold, 2],
            valid_score_df[fold, 3],
        )
        send_msg_to_dingtalk(
            INFO_TEXT, IS_SEND_MSG_TO_DINGTALK
        )
    print('==================================')

    # 后处理评估与分析阶段
    # *******************

    # Out of fold预测结果评估
    # ----------
    valid_mse = mean_squared_error(
        train_target.reshape(-1, 1), valid_pred_result_df.reshape(-1, 1)
    )
    val_mae = mean_absolute_error(
        train_target.reshape(-1, 1), valid_pred_result_df.reshape(-1, 1)
    )
    val_mape = mean_absolute_percentage_error(
        train_target.reshape(-1, 1), valid_pred_result_df.reshape(-1, 1)
    )

    INFO_TEXT = '-- [INFO][{}] {} TOTAL OOF valid mse: {:.5f}, mae {:5f}, mape: {:.5f}'.format(
        MODEL_NAME, str(datetime.now())[:-4],
        valid_mse, val_mae, val_mape,
    )
    send_msg_to_dingtalk(
        INFO_TEXT, IS_SEND_MSG_TO_DINGTALK
    )
