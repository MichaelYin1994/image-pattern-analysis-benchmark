#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202108091732
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
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from dingtalk_remote_monitor import RemoteMonitorDingTalk
from utils import LearningRateWarmUpCosineDecayScheduler, LoadSave

GLOBAL_RANDOM_SEED = 1256
# np.random.seed(GLOBAL_RANDOM_SEED)
# tf.random.set_seed(GLOBAL_RANDOM_SEED)

TASK_NAME = 'iflytek_2021_human_face_emotion'
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
def build_simple_convnet(verbose=False, is_compile=True, **kwargs):
    '''构造preprocessing与model的pipeline，并返回编译过的模型。'''

    # 解析preprocessing与model的参数
    # ---------------------
    input_shape = kwargs.pop('input_shape', (None, 224, 224))
    n_classes = kwargs.pop('n_classes', 1000)

    model_name = kwargs.pop('model_name', 'EfficentNetB0')
    model_lr = kwargs.pop('model_lr', 0.01)
    model_label_smoothing = kwargs.pop('model_label_smoothing', 0.1)

    # 构建模型
    # ---------------------
    layer_input = tf.keras.layers.Input(shape=input_shape, dtype='float32')

    # layer_input = tf.keras.layers.BatchNormalization()(layer_input)
    layer_conv = tf.keras.layers.Conv2D(
        filters=128, kernel_size=5, padding='same',
    )(layer_input)

    # layer_conv = tf.keras.layers.BatchNormalization()(layer_conv)
    layer_conv = tf.keras.layers.Conv2D(
        filters=256, kernel_size=3, padding='same',
    )(layer_conv)

    # layer_conv = tf.keras.layers.BatchNormalization()(layer_conv)
    layer_conv = tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, padding='same',
    )(layer_conv)

    layer_avg_pool = tf.keras.layers.GlobalAveragePooling2D()(layer_conv)
    layer_max_pool = tf.keras.layers.GlobalMaxPool2D()(layer_conv)

    layer_avg_pool_flatten = tf.keras.layers.Flatten()(layer_avg_pool)
    layer_max_pool_flatten = tf.keras.layers.Flatten()(layer_max_pool)

    layer_output = tf.keras.layers.concatenate(
        [layer_avg_pool_flatten, layer_max_pool_flatten]
    )
    layer_output = tf.keras.layers.Dropout(0.3)(layer_output)
    layer_output = tf.keras.layers.Dense(
        n_classes, activation='softmax'
    )(layer_output)

    # 编译模型
    # ---------------------
    model = tf.keras.Model(layer_input, layer_output)

    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=model_label_smoothing),
            optimizer=Adam(model_lr),
            metrics=['acc'])

    return model


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
        model_tmp = tf.keras.applications.ResNet152V2

    # 构建ResNet模型
    model_tmp = model_tmp(
        input_shape=input_shape, 
        include_top=False,
        weights='imagenet',
    )
    model_tmp.trainable = False

    model.add(model_tmp)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        1024, activation='relu',
    ))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(
        256, activation='relu',
    ))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(
        64, activation='relu',
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

    model_tmp = model_tmp(
        input_shape=input_shape, 
        include_top=False,
        weights='imagenet',
        drop_connect_rate=0.4,
    )
    model_tmp.trainable = False

    model.add(model_tmp)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        1024, activation='relu',
    ))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(
        256, activation='relu',
    ))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(
        64, activation='relu',
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

    if stage == 'train' or stage == 'test':
        def load_img(path=None):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=1)
            image = tf.concat([image, image, image], axis=-1)

            # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            # image = tf.image.random_hue(image, max_delta=0.2)
            # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            # image = tf.image.random_brightness(image, 0.3)

            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

            image = tf.image.resize(image, image_size)
            image = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.)(image)
            return image
    else:
        def load_img(path=None):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=1)
            image = tf.concat([image, image, image], axis=-1)

            # image = tf.image.resize(image, image_size)
            image = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.)(image)
            return image

    return load_img


if __name__ == '__main__':
    # 全局化的参数列表
    # ---------------------
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 8
    NUM_EPOCHS = 256
    EARLY_STOP_ROUNDS = 30
    N_FOLDS = 5
    TTA_ROUNDS = 5
    IS_STRATIFIED = True
    MODEL_NAME = 'EfficentNetB5_rtx3090'

    MODEL_LR = 0.00003
    MODEL_LABEL_SMOOTHING = 0

    CKPT_DIR = './ckpt/'
    CKPT_FOLD_NAME = '{}_GPU_{}_{}'.format(TASK_NAME, GPU_ID, MODEL_NAME)

    IS_TRAIN_FROM_CKPT = False
    IS_SEND_MSG_TO_DINGTALK = False
    IS_RANDOM_VISUALIZING_PLOTS = False

    # 数据loading的path
    TRAIN_PATH = './data/train/'
    TEST_PATH = './data/test/'
    N_CLASSES = len(os.listdir(TRAIN_PATH))

    # 利用tensorflow的preprocessing方法读取数据集
    # ---------------------
    train_file_full_name_list = []
    train_label_list = []
    for dir_name in os.listdir(TRAIN_PATH):
        full_path_name = os.path.join(TRAIN_PATH, dir_name)
        for file_name in os.listdir(full_path_name):
            train_file_full_name_list.append(
                os.path.join(full_path_name, file_name)
            )
            train_label_list.append(dir_name)

    label2id = {
        'angry': 0,
        'disgusted': 1,
        'fearful': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprised': 6,
    }
    id2label = {
        0: 'angry',
        1: 'disgusted',
        2: 'fearful',
        3: 'happy',
        4: 'neutral',
        5: 'sad',
        6: 'surprised',
    }
    train_label_oht_array = [label2id[item] for item in train_label_list]
    train_label_oht_array = np.array(train_label_oht_array)

    test_file_name_list = os.listdir(TEST_PATH)
    test_file_fullname_list = \
        [os.path.join(TEST_PATH, item) for item in test_file_name_list]

    # 编码训练标签
    train_label_oht_array = to_categorical(train_label_oht_array.reshape(-1, 1))

    # shuffle类间样本
    shuffled_idx = np.arange(0, len(train_label_oht_array))
    np.random.shuffle(shuffled_idx)
    train_file_full_name_list = [train_file_full_name_list[i] for i in shuffled_idx]
    train_label_list = [train_label_list[i] for i in shuffled_idx]
    train_label_oht_array = train_label_oht_array[shuffled_idx, :]

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
    n_train_samples = len(train_file_full_name_list)
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

        # 随机可视化几张图片
        # ************
        if IS_RANDOM_VISUALIZING_PLOTS:
            plt.close('all')
            plt.figure(figsize=(10, 10))
            for images, labels in train_ds.take(1):
                for i in range(9):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i].numpy().astype('uint8'))
                    plt.title(str(labels[i].numpy()), fontsize=9)
                    plt.axis('off')
            plt.tight_layout()

        # 构造与编译模型
        # ***********
        model = build_efficentnet_model(
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
    test_pred_label_list = np.argmax(test_pred_proba, axis=1)
    test_pred_label_list = test_pred_label_list.astype('int')

    test_pred_label_list = [id2label[int(item)] for item in test_pred_label_list]

    test_pred_df = pd.DataFrame(
        test_file_name_list,
        columns=['name']
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
