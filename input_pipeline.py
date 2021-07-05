#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106121937
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(input_pipeline.py)构建数据读取与预处理的pipline，并训练神经网络模型。
'''

import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import load
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

GLOBAL_RANDOM_SEED = 7555
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)

TASK_NAME = 'iflytek_2021'
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

def build_model(verbose=False, is_compile=True, **kwargs):
    '''构造preprocessing与model的pipline，并返回编译过的模型。'''
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
            loss='categorical_crossentropy',
            optimizer=Adam(0.0005),
            metrics=['acc'])

    return model


def load_preprocess_train_image(image_size=None):
    '''通过闭包实现参数化的训练集的Image loading。'''

    def load_img(path=None):
        image = tf.io.read_file(path)
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, channels=3),
            lambda: tf.image.decode_gif(image)[0])

        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        image = tf.image.resize(image, image_size)
        return image

    return load_img


def load_preprocess_test_image(image_size=None):
    '''通过闭包实现参数化的测试集的Image loading。'''

    def load_img(path=None):
        image = tf.io.read_file(path)
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, channels=3),
            lambda: tf.image.decode_gif(image)[0])
        image = tf.image.resize(image, image_size)

        return image
    return load_img


class LearningRateWarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    '''
    带有Learning Rate的Warmup与Cosine自调节的Learning Rate回调类，主要参考
    文献[1]与文献[2]。

    @Args:
    ----------
    learning_rate_base: {float-like}
        基础的学习率。
    total_steps: {int-like}
        总共的训练的step的数目，#Steps = #Epochs * #Images / Batch_size。
    global_steps_initial: {int-like}
        当使用ckpt训练的时候，初始的step的数目。
    warmup_learning_rate: {int-like}
        初始的warmup学习率，一般取0。
    warmup_steps: {bool-like}
        采用的warmup的steps的数目。
    hold_steps: {str-like}
        保持learning_rate_base的steps数目。

    @References:
    ----------
    [1] https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/?t=162513696863
    [2] He, Tong, et al. "Bag of tricks for image classification with convolutional neural networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

    @Returns:
    ----------
    None，通过callback方法对Model的学习率进行设置。
    '''
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_steps_initial=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_steps=0):
        super(LearningRateWarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.warmup_learning_rate = warmup_learning_rate

        self.total_steps = total_steps
        self.current_step = global_steps_initial
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps

        self.history_learning_rates = []

    def learning_rate_cosine_decay_with_hold(
        self,
        current_step,
        learning_rate_base,
        total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=0,
        hold_base_rate_steps=0):
        '''
        带有warmup功能与learning rate holding功能的learning_rate生成方法。

        @Args:
        ----------
        current_step: {float-like}
            当前全局的训练的step数目。
        learning_rate_base: {float-like}
            基础的学习率，在warmup后保持hold_steps步数。
        total_steps: {int-like}
            总的step的数目，取值为n_epoch * n_batches_per_epoch。
        warmup_learning_rate: {float-like}
            初始warmup的学习率大小，一般取0。
        warmup_steps: {bool-like}
            warmup的步数。
        hold_steps: {str-like}
            学习率保持(hold)的步数。

        @Return:
        ----------
        调节好的学习率。
        '''
        # 修复文献[1]学习率santity check的bug
        if total_steps < (warmup_steps + hold_base_rate_steps):
            raise ValueError('total_steps must be larger',
                             'or equal to warmup_steps + hold_base_rate_steps.')

        if warmup_learning_rate > learning_rate_base:
            raise ValueError('warmup learning rate must be larger',
                             'than base learning rate.')

        # 公式来源：文献[2]
        learning_rate = 0.5 * learning_rate_base * \
            (1 + np.cos(np.pi * (current_step - warmup_steps - hold_base_rate_steps) \
            / (total_steps - warmup_steps - hold_base_rate_steps)))

        # 若当前step小于warmup_steps，计算并设置为warmup学习率
        if warmup_steps > 0 and current_step < warmup_steps:
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            current_warmup_rate = slope * current_step + warmup_learning_rate
            learning_rate = current_warmup_rate

        # 若是warmup结束，并且需要hold学习率，则设置为需要hold的学习率
        if hold_base_rate_steps > 0 and \
            current_step < (warmup_steps + hold_base_rate_steps) and \
            current_step > warmup_steps:
            learning_rate = learning_rate_base

        return learning_rate

    def on_batch_end(self, batch, logs=None):
        self.current_step = self.current_step + 1
        learning_rate = K.get_value(self.model.optimizer.lr)
        self.history_learning_rates.append(learning_rate)

    def on_batch_begin(self, batch, logs=None):
        learning_rate = self.learning_rate_cosine_decay_with_hold(
            current_step=self.current_step,
            learning_rate_base=self.learning_rate_base,
            total_steps=self.total_steps,
            warmup_learning_rate=self.warmup_learning_rate,
            warmup_steps=self.warmup_steps,
            hold_base_rate_steps=self.hold_steps
        )

        K.set_value(self.model.optimizer.lr, learning_rate)


if __name__ == '__main__':
    # 全局化的参数列表
    # ---------------------
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 16
    NUM_EPOCHS = 128
    NUM_WARMUP_EPOCHS = 6
    NUM_HOLD_EPOCHS = 5
    EARLY_STOP_ROUNDS = 6
    MODEL_NAME = 'EfficientNetB3_rtx3090'

    CKPT_DIR = './ckpt/'
    CKPT_FOLD_NAME = '{}_GPU_{}_{}'.format(TASK_NAME, GPU_ID, MODEL_NAME)

    IS_TRAIN_FROM_CKPT = False
    IS_SEND_MSG_TO_DINGTALK = True
    IS_DEBUG = False

    # 数据loading的path
    if IS_DEBUG:
        TRAIN_PATH = './data/train_debug/'
        TEST_PATH = './data/test_debug/'
    else:
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
            train_label_list.append(int(dir_name))
    train_label_oht_array = np.array(train_label_list)

    # 编码训练标签
    encoder = OneHotEncoder(sparse=False)
    train_label_oht_array = encoder.fit_transform(
        train_label_oht_array.reshape(-1, 1))

    # 按照比例划分Train与Validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_file_full_name_list, train_label_oht_array,
        train_size=0.8, random_state=GLOBAL_RANDOM_SEED,
    )

    n_train_samples, n_valid_samples = len(X_train), len(X_val)

    # 构造训练数据集的pipline
    processor_train_image = load_preprocess_train_image(image_size=IMAGE_SIZE)
    processor_valid_image = load_preprocess_test_image(image_size=IMAGE_SIZE)

    train_path_ds = tf.data.Dataset.from_tensor_slices(X_train)
    train_img_ds = train_path_ds.map(
        processor_train_image, num_parallel_calls=mp.cpu_count()
    )
    train_label_ds = tf.data.Dataset.from_tensor_slices(y_train)

    train_ds = tf.data.Dataset.zip((train_img_ds, train_label_ds))

    # 构造validation数据集的pipline
    val_path_ds = tf.data.Dataset.from_tensor_slices(X_val)
    val_img_ds = val_path_ds.map(
        processor_valid_image, num_parallel_calls=mp.cpu_count()
    )
    val_label_ds = tf.data.Dataset.from_tensor_slices(y_val)

    val_ds = tf.data.Dataset.zip((val_img_ds, val_label_ds))

    # Performance
    # train_ds = train_ds.shuffle(buffer_size=int(32 * BATCH_SIZE))
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(2 * BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(2 * BATCH_SIZE)

    # 随机可视化几张图片
    IS_RANDOM_VISUALIZING_PLOTS = False

    if IS_RANDOM_VISUALIZING_PLOTS:
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.title(int(labels[i]))
                plt.axis('off')
        plt.tight_layout()

    # 构造与编译Model，并添加各种callback
    # ---------------------

    # 各种Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', mode="max",
            verbose=1, patience=EARLY_STOP_ROUNDS,
            restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                CKPT_DIR + CKPT_FOLD_NAME,
                MODEL_NAME + '_epoch_{epoch:02d}_valacc_{val_acc:.3f}.ckpt'),
            monitor='val_acc',
            mode='max',
            save_weights_only=True,
            save_best_only=True),
        # tf.keras.callbacks.ReduceLROnPlateau(
        #         monitor='val_acc',
        #         factor=0.5,
        #         patience=2,
        #         min_lr=0.0000003),
        RemoteMonitorDingTalk(
            is_send_msg=IS_SEND_MSG_TO_DINGTALK,
            model_name=MODEL_NAME,
            gpu_id=GPU_ID),
        LearningRateWarmUpCosineDecayScheduler(
            learning_rate_base=0.0003,
            total_steps=int(n_train_samples * NUM_EPOCHS / BATCH_SIZE),
            global_steps_initial=0,
            warmup_learning_rate=0.0000001,
            warmup_steps=int(n_train_samples * NUM_WARMUP_EPOCHS / BATCH_SIZE),
            hold_steps=int(n_train_samples * NUM_HOLD_EPOCHS / BATCH_SIZE))
    ]

    # 训练模型
    model = build_model(
        n_classes=N_CLASSES,
        input_shape=IMAGE_SIZE + (3,),
        network_type=MODEL_NAME
    )

    # 如果模型名的ckpt文件夹不存在，创建该文件夹
    if CKPT_FOLD_NAME not in os.listdir(CKPT_DIR):
        os.mkdir(CKPT_DIR + CKPT_FOLD_NAME)

    # 如果指定ckpt weights文件名，则从ckpt位置开始训练
    if IS_TRAIN_FROM_CKPT:
        latest_ckpt = tf.train.latest_checkpoint(CKPT_DIR + CKPT_FOLD_NAME)
        model.load_weights(latest_ckpt)
    else:
        ckpt_file_name_list = os.listdir(CKPT_DIR + CKPT_FOLD_NAME)

        # https://www.geeksforgeeks.org/python-os-remove-method/
        try:
            for file_name in ckpt_file_name_list:
                os.remove(os.path.join(CKPT_DIR + CKPT_FOLD_NAME, file_name))
        except OSError:
            print('File {} can not be deleted !'.format(file_name))

    history = model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # 生成Test预测结果，并进行Top-1 Accuracy评估
    # ---------------------
    test_file_name_list = os.listdir(TEST_PATH)
    test_file_name_list = sorted(test_file_name_list, key=lambda x: int(x.split('.')[0][1:]))
    test_file_fullname_list = [TEST_PATH + item for item in test_file_name_list]

    test_path_ds = tf.data.Dataset.from_tensor_slices(test_file_fullname_list)
    processor_test_image = load_preprocess_test_image(image_size=IMAGE_SIZE)
    test_ds = test_path_ds.map(
        processor_test_image,
        num_parallel_calls=mp.cpu_count()
    )
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(buffer_size=int(BATCH_SIZE * 2))

    test_pred_proba = model.predict(test_ds)
    test_pred_label_list = np.argmax(test_pred_proba, axis=1)

    test_pred_df = pd.DataFrame(
        test_file_name_list,
        columns=['image_id']
    )
    test_pred_df['category_id'] = test_pred_label_list

    sub_file_name = str(len(os.list_dir('./submissions'))) + '_sub.csv'
    test_pred_df.to_csv(sub_file_name, index=False)
