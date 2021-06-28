#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106121937
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(input_pipeline.py)构建数据读取与预处理的pipline，并训练神经网络模型。
'''

import os
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam

from dingtalk_remote_monitor import RemoteMonitorDingTalk
from models import build_model_resnet50_v2, build_model_resnet101_v2

GLOBAL_RANDOM_SEED = 192
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)

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

"""
def build_model(verbose=False, is_compile=True, **kwargs):
    '''构造preprocessing与model的pipline，并返回编译过的模型。'''
    network_type = kwargs.pop('network_type', 'resnet50')

    # 解析preprocessing与model的参数
    # ---------------------
    input_shape = kwargs.pop('input_shape', (None, 224, 224))
    n_classes = kwargs.pop('n_classes', 1000)

    # 构造data input与preprocessing的pipline
    # ---------------------
    layer_input = keras.Input(shape=input_shape, name='layer_input')

    layer_data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip('horizontal'),
            layers.experimental.preprocessing.RandomRotation(0.2),
        ])
    layer_input_aug = layer_data_augmentation(layer_input)
    layer_input_aug = layers.experimental.preprocessing.Rescaling(
        1 / 255)(layer_input_aug)

    # 构造Model的pipline
    # ---------------------
    if 'resnet50' in network_type: 
        x = build_model_resnet50_v2(layer_input_aug)
    elif 'resnet101' in network_type:
        x = build_model_resnet101_v2(layer_input_aug)

    x = layers.GlobalAveragePooling2D()(x)
    if n_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = n_classes

    x = layers.Dropout(0.5)(x)
    layer_output = layers.Dense(units, activation=activation)(x)

    # 编译模型
    # ---------------------
    model = Model(layer_input, layer_output)

    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.001),
            metrics=['acc'])

    return model
"""

def build_model(verbose=False, is_compile=True, **kwargs):
    '''构造preprocessing与model的pipline，并返回编译过的模型。'''
    # network_type = kwargs.pop('network_type', 'resnet50')

    # 解析preprocessing与model的参数
    # ---------------------
    input_shape = kwargs.pop('input_shape', (None, 224, 224))
    n_classes = kwargs.pop('n_classes', 1000)

    model = tf.keras.Sequential()
    # initialize the model with input shape
    model.add(
        tf.keras.applications.EfficientNetB3(
            input_shape=input_shape, 
            include_top=False,
            weights='imagenet',
            drop_connect_rate=0.6,
        )
    )
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        256,
        activation='relu', 
        bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)
    ))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    # 编译模型
    # ---------------------
    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.001),
            metrics=['acc'])

    return model


def load_preprocess_image(image_size=None):
    '''通过闭包实现参数化的Image loading。'''

    def fcn(path=None):
        image = tf.io.read_file(path)
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, channels=3),
            lambda: tf.image.decode_gif(image)[0])
        image = tf.image.resize(image, image_size)

        return image
    return fcn


if __name__ == '__main__':
    # 全局化的参数列表
    # ---------------------
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 64
    NUM_EPOCHS = 128
    EARLY_STOP_ROUNDS = 10
    MODEL_NAME = 'EfficientNetB3_quadrop5000'
    CKPT_PATH = './ckpt/{}/'.format(MODEL_NAME)

    IS_TRAIN_FROM_CKPT = False
    IS_SEND_MSG_TO_DINGTALK = True
    IS_DEBUG = False

    if IS_DEBUG:
        TRAIN_PATH = './data/train_debug/'
        TEST_PATH = './data/test_debug/'
    else:
        TRAIN_PATH = './data/train/'
        TEST_PATH = './data/test/'
    N_CLASSES = len(os.listdir(TRAIN_PATH))

    # 利用tensorflow的preprocessing方法读取数据集
    # ---------------------
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH,
        label_mode='categorical',
        shuffle=True,
        validation_split=0.2,
        subset="training",
        seed=GLOBAL_RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH,
        label_mode='categorical',
        shuffle=True,
        subset="validation",
        validation_split=0.2,
        seed=GLOBAL_RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    train_ds = train_ds.prefetch(buffer_size=int(BATCH_SIZE * 2))
    val_ds = val_ds.prefetch(buffer_size=int(BATCH_SIZE * 2))

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype('uint8'))
    #         plt.title(int(labels[i]))
    #         plt.axis('off')
    # plt.tight_layout()

    # 构造与编译Model，并添加各种callback
    # ---------------------

    # 各种Callbacks
    # ckpt, lr schule, early stop, warm up, remote moniter
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', mode="max",
            verbose=1, patience=EARLY_STOP_ROUNDS,
            restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                CKPT_PATH,
                MODEL_NAME + '_epoch_{epoch:02d}_valacc_{val_acc:.3f}.ckpt'),
            monitor='val_acc',
            mode='max',
            save_weights_only=True,
            save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc',
                factor=0.5,
                patience=3,
                min_lr=0.0000003),
        RemoteMonitorDingTalk(
            is_send_msg=IS_SEND_MSG_TO_DINGTALK,
            model_name=MODEL_NAME,
            gpu_id=GPU_ID)
    ]

    # 训练模型
    model = build_model(
        n_classes=N_CLASSES,
        input_shape=IMAGE_SIZE + (3,),
        network_type=MODEL_NAME
    )

    # 如果模型名的ckpt文件夹不存在，创建该文件夹
    if MODEL_NAME not in os.listdir('./ckpt'):
        os.mkdir('./ckpt/' + MODEL_NAME)

    # 如果指定ckpt weights文件名，则从ckpt位置开始训练
    if IS_TRAIN_FROM_CKPT:
        latest_ckpt = tf.train.latest_checkpoint(CKPT_PATH)
        model.load_weights(latest_ckpt)
    else:
        ckpt_file_name_list = os.listdir(CKPT_PATH)

        # https://www.geeksforgeeks.org/python-os-remove-method/
        try:
            for file_name in ckpt_file_name_list:
                os.remove(os.path.join(CKPT_PATH, file_name))
        except OSError:
            print('File {} can not be deleted !'.format(file_name))

    history = model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # 生成预测结果
    # ---------------------
    test_file_name_list = os.listdir(TEST_PATH)
    test_file_name_list = sorted(test_file_name_list, key=lambda x: int(x.split('.')[0][1:]))
    test_file_fullname_list = [TEST_PATH + item for item in test_file_name_list]

    test_path_ds = tf.data.Dataset.from_tensor_slices(test_file_fullname_list)
    load_preprocess_test_image = load_preprocess_image(image_size=IMAGE_SIZE)
    test_ds = test_path_ds.map(
        load_preprocess_test_image,
        num_parallel_calls=mp.cpu_count()
    )
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(buffer_size=int(BATCH_SIZE * 2))

    test_pred_proba = model.predict(test_ds)

    test_pred_df = pd.DataFrame(
        test_file_name_list,
        columns=['image_id']
    )
    test_pred_df['category_id'] = np.argmax(test_pred_proba, axis=1)

    # test_sub_df = pd.read_csv('./data/submit_sample.csv')
    test_pred_df.to_csv('./submissions/sub.csv', index=False)
