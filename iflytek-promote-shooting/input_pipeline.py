#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202108072008
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(input_pipeline.py)构建数据读取与预处理的pipline，并训练神经网络模型。
其中本模块仅采用简单的数据增强的相关策略。
'''

import gc
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from dingtalk_remote_monitor import RemoteMonitorDingTalk
from utils import LearningRateWarmUpCosineDecayScheduler, LoadSave

GLOBAL_RANDOM_SEED = 7555
# np.random.seed(GLOBAL_RANDOM_SEED)
# tf.random.set_seed(GLOBAL_RANDOM_SEED)

TASK_NAME = 'iflytek_2021_promote_shooting'
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

# ----------------------------------------------------------------------------

def build_resnetv2_model(verbose=False, is_compile=True, **kwargs):
    '''构造preprocessing与model的pipeline，并返回编译过的模型。'''

    # 解析preprocessing与model的参数
    # ---------------------
    input_shape = kwargs.pop('input_shape', (None, 224, 224))
    n_classes = kwargs.pop('n_classes', 1000)

    model_name = kwargs.pop('model_name', 'EfficentNetB0')
    model_lr = kwargs.pop('model_lr', 0.01)
    model_label_smoothing = kwargs.pop('model_label_smoothing', 0.1)

    # 依据关键字，构建模型
    # ---------------------
    model = tf.keras.Sequential()

    if '50' in model_name:
        model_tmp = tf.keras.applications.ResNet50V2
    elif '101' in model_name:
        model_tmp = tf.keras.applications.ResNet101V2
    elif '152' in model_name:
        model_tmp = tf.keras.applications.ResNet101V2152

    model.add(
        model_tmp(
            input_shape=input_shape, 
            include_top=False,
            weights='imagenet',
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



if __name__ == '__main__':
    # 全局化的参数列表
    # ---------------------
    IMAGE_SIZE = (960, 544)
    BATCH_SIZE = 16
    NUM_EPOCHS = 256
    EARLY_STOP_ROUNDS = 30
    N_FOLDS = 5
    TTA_ROUNDS = 20
    IS_STRATIFIED = True
    MODEL_NAME = 'ResNet50v2_rtx3090'

    MODEL_LR = 0.00003
    MODEL_LABEL_SMOOTHING = 0

    CKPT_DIR = './ckpt/'
    CKPT_FOLD_NAME = '{}_GPU_{}_{}'.format(TASK_NAME, GPU_ID, MODEL_NAME)

    IS_TRAIN_FROM_CKPT = False
    IS_SEND_MSG_TO_DINGTALK = False
    IS_DEBUG = False
    IS_RANDOM_VISUALIZING_PLOTS = False

    # 数据loading的path
    if IS_DEBUG:
        TRAIN_PATH = './data/train_debug/'
        TEST_PATH = './data/test_debug/'
    else:
        TRAIN_PATH = './data/train/'
        TEST_PATH = './data/test/'
    N_CLASSES = 8

    # 利用tensorflow的preprocessing方法读取数据集
    # ---------------------

    # 读取label的*.csv格式
    train_label_df = pd.read_csv('./data/train.csv')
    train_label_df['label'] = train_label_df['label'].apply(lambda x: int(x-1))

    train_file_full_name_list = train_label_df['image'].values.tolist()
    train_file_full_name_list = \
        [os.path.join(TRAIN_PATH, item) for item in train_file_full_name_list]
    train_label_list = train_label_df['label'].values.tolist()

    test_file_name_list = os.listdir(TEST_PATH)
    test_file_name_list = \
        sorted(test_file_name_list, key=lambda x: int(x.split('.')[0][5:]))
    test_file_fullname_list = \
        [os.path.join(TEST_PATH, item) for item in test_file_name_list]

    # 编码训练标签
    train_label_oht_array = np.array(train_label_list)

    encoder = OneHotEncoder(sparse=False)
    train_label_oht_array = encoder.fit_transform(
        train_label_oht_array.reshape(-1, 1)).astype(np.float32)

    folds = KFold(n_splits=N_FOLDS, shuffle=True,
                  random_state=GLOBAL_RANDOM_SEED)

    # 准备train数据与test数据的loader
    processor_train_image = load_preprocessing_img(
        image_size=IMAGE_SIZE, stage='train'
    )
    processor_valid_image = load_preprocessing_img(
        image_size=IMAGE_SIZE, stage='test'
    )
    processor_test_image = load_preprocessing_img(
        image_size=IMAGE_SIZE, stage='train'
    )

    # 准备训练数据与训练模型
    # --------------------------------
    n_train_samples = len(train_label_df)
    n_test_samples = len(test_file_fullname_list)
    test_pred_proba_list = []

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_acc', mode='max',
        verbose=1, patience=EARLY_STOP_ROUNDS,
        restore_best_weights=True)
    remote_monitor = RemoteMonitorDingTalk(
        is_send_msg=IS_SEND_MSG_TO_DINGTALK, model_name=MODEL_NAME)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=0.7,
        patience=15,
        min_lr=0.000003)

    if IS_STRATIFIED:
        folds = StratifiedKFold(
            n_splits=N_FOLDS,
            random_state=GLOBAL_RANDOM_SEED,
            shuffle=True,
        )
    else:
        folds = KFold(
            n_splits=N_FOLDS,
            random_state=GLOBAL_RANDOM_SEED,
            shuffle=True,
        )

    test_path_ds = tf.data.Dataset.from_tensor_slices(test_file_fullname_list)
    test_ds = test_path_ds.map(
        processor_test_image,
        num_parallel_calls=mp.cpu_count()
    )
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(buffer_size=int(BATCH_SIZE * 16))

    print('==================================')
    for fold, (tra_id, val_id) in enumerate(
            folds.split(train_file_full_name_list, train_label_list)):
        # 销毁所有内存中的图结构，便于多fold验证
        K.clear_session()
        gc.collect()

        # 划分train与val数据集，并且构造train与val的数据集loader
        # ***********
        X_train, X_val = \
            [train_file_full_name_list[i] for i in tra_id], \
            [train_file_full_name_list[i] for i in val_id]
        y_train, y_val = train_label_oht_array[tra_id, :], train_label_oht_array[val_id, :]

        train_path_ds = tf.data.Dataset.from_tensor_slices(X_train)
        train_img_ds = train_path_ds.map(
            processor_train_image, num_parallel_calls=mp.cpu_count()
        )
        train_label_ds = tf.data.Dataset.from_tensor_slices(y_train)
        train_ds = tf.data.Dataset.zip((train_img_ds, train_label_ds))

        val_path_ds = tf.data.Dataset.from_tensor_slices(X_val)
        val_img_ds = val_path_ds.map(
            processor_valid_image, num_parallel_calls=mp.cpu_count()
        )
        val_label_ds = tf.data.Dataset.from_tensor_slices(y_val)
        val_ds = tf.data.Dataset.zip((val_img_ds, val_label_ds))

        # 数据集性能相关参数
        # ************
        train_ds = train_ds.batch(BATCH_SIZE).prefetch(16 * BATCH_SIZE)
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(16 * BATCH_SIZE)

        # 构造与编译模型
        # ***********
        model = build_resnetv2_model(
            n_classes=N_CLASSES,
            input_shape=IMAGE_SIZE + (3,),
            network_type=MODEL_NAME,
            model_name=MODEL_NAME,
            model_lr=MODEL_LR,
            model_label_smoothing=MODEL_LABEL_SMOOTHING,
        )

        # 完善ckpt保存机制
        # ***********
        # 如果模型名的ckpt文件夹不存在，创建该文件夹
        ckpt_fold_name_tmp = CKPT_FOLD_NAME + '_fold_{}'.format(fold)

        if ckpt_fold_name_tmp not in os.listdir(CKPT_DIR):
            os.mkdir(CKPT_DIR + ckpt_fold_name_tmp)

        # 如果指定ckpt weights文件名，则从ckpt位s置开始训练
        ckpt_file_name_list = os.listdir(CKPT_DIR + ckpt_fold_name_tmp)

        if IS_TRAIN_FROM_CKPT:
            if len(ckpt_file_name_list) != 0:
                latest_ckpt = tf.train.latest_checkpoint(CKPT_DIR + ckpt_fold_name_tmp)
                model.load_weights(latest_ckpt)
        else:
            # https://www.geeksforgeeks.org/python-os-remove-method/
            try:
                for file_name in ckpt_file_name_list:
                    os.remove(os.path.join(CKPT_DIR + ckpt_fold_name_tmp, file_name))
            except OSError:
                print('File {} can not be deleted !'.format(file_name))

        ckpt_saver = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    CKPT_DIR + ckpt_fold_name_tmp,
                    MODEL_NAME + '.ckpt'),
                monitor='val_acc',
                mode='max',
                save_weights_only=True,
                save_best_only=True),

        # fitting模型
        # ***********
        callbacks = [early_stop, remote_monitor, reduce_lr, ckpt_saver]
    
        history = model.fit(
            train_ds,
            epochs=NUM_EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks
        )

        # TTA强化，生成预测结果
        # ***********
        for i in tqdm(range(TTA_ROUNDS)):
            test_pred_proba_list.append(model.predict(test_ds))

    # 保存测试预测结果
    # --------------------------------
    test_pred_proba = np.mean(test_pred_proba_list, axis=0)
    test_pred_label_list = np.argmax(test_pred_proba, axis=1) + 1
    test_pred_label_list = test_pred_label_list.astype('int')

    test_pred_df = pd.DataFrame(
        test_file_name_list,
        columns=['image']
    )
    test_pred_df['label'] = test_pred_label_list

    sub_file_name = str(len(os.listdir('./submissions')) + 1) + \
        '_{}_sub'.format(MODEL_NAME)
    test_pred_df.to_csv('./submissions/{}.csv'.format(sub_file_name), index=False)

    # 保存提交概率情况
    file_processor = LoadSave(dir_name='./submissions_proba/')
    test_pkl_res = [test_pred_df] + test_pred_proba_list

    file_processor.save_data(
        file_name=sub_file_name + '.pkl',
        data_file=test_pkl_res
    )
