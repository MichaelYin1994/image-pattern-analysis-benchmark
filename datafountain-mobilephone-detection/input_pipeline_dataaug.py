#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107180257
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(input_pipeline.py)构建数据读取与预处理的pipline，并训练神经网络模型。
其中本模块采用Mixup，Mixmatch等数据增强策略。
'''

import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from dingtalk_remote_monitor import RemoteMonitorDingTalk

from utils import LearningRateWarmUpCosineDecayScheduler, LoadSave, custom_eval_metric

GLOBAL_RANDOM_SEED = 7555
# np.random.seed(GLOBAL_RANDOM_SEED)
# tf.random.set_seed(GLOBAL_RANDOM_SEED)

TASK_NAME = 'datafountain_2021_mobilephone_detection'
GPU_ID = 1

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
def tf_f1_score(y_true, y_pred):
    return tf.py_function(custom_eval_metric, (y_true, y_pred, 0.5), tf.double)


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
            metrics=[tf_f1_score])

    return model


def load_preprocessing_img(image_size, stage):
    '''通过闭包实现参数化的Image Loading与TTA数据增强。'''
    if stage not in ['train', 'valid', 'test']:
        raise ValueError('stage must be either train, valid or test !')

    if stage == 'train' or stage == 'test':
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

            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

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


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    '''从beta分布中抽取指定size的数据'''
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    '''对输入2个tf.data.Dataset对象执行mix_up数据增强'''
    # 解压2个tf.data.Dataset实例
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # 确定lambda参数用于Mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # 进行Mixup
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)

    return (images, labels)


def get_box(lambda_value, image_size):
    '''生成切分点与切分坐标'''
    cut_rate = tf.math.sqrt(1.0 - lambda_value)

    # cutted crop的width和height
    cut_h = image_size[0] * cut_rate
    cut_h = tf.cast(cut_h, tf.int32)

    cut_w = image_size[1] * cut_rate
    cut_w = tf.cast(cut_w, tf.int32)

    # 从均匀分布生成切分crop的中心点坐标
    center_h = tf.random.uniform((1,), minval=0, maxval=image_size[0], dtype=tf.int32)
    center_w = tf.random.uniform((1,), minval=0, maxval=image_size[1], dtype=tf.int32)

    # 生成cutted crop上下前后范围坐标
    start_h = tf.clip_by_value(center_h[0] - cut_h // 2, 0, image_size[0])
    start_w = tf.clip_by_value(center_w[0] - cut_w // 2, 0, image_size[1])
    end_h = tf.clip_by_value(center_h[0] + cut_h // 2, 0, image_size[0])
    end_w = tf.clip_by_value(center_w[0] + cut_w // 2, 0, image_size[1])

    # corner case控制
    target_h = end_h - start_h
    if target_h == 0:
        target_h += 1

    target_w = end_w - start_w
    if target_w == 0:
        target_w += 1

    return start_h, start_w, target_h, target_w


# https://keras.io/examples/vision/cutmix
def cut_mix(ds_one, ds_two, image_size, alpha=0.5, beta=0.25):
    '''对输入2个tf.data.Dataset对象执行cut_mix数据增强'''
    (image_1, label_1), (image_2, label_2) = ds_one, ds_two

    # 从beta分布抽取lambda值
    lambda_value = sample_beta_distribution(1, [alpha], [beta])
    lambda_value = lambda_value[0][0]

    # 获取起始(h, w)坐标
    start_h, start_w, target_h, target_w = get_box(lambda_value, image_size)

    # 生成image_2的抠图结果，并zero padding到指定size
    crop_2 = tf.image.crop_to_bounding_box(
        image_2, start_h, start_w, target_h, target_w
    )
    zero_padded_image_2 = tf.image.pad_to_bounding_box(
        crop_2, start_h, start_w, image_size[0], image_size[1]
    )

    # 对image_1进行抠图，并补全到指定大小
    crop_1 = tf.image.crop_to_bounding_box(
        image_1, start_h, start_w, target_h, target_w
    )
    zero_padded_image_1 = tf.image.pad_to_bounding_box(
        crop_1, start_h, start_w, image_size[0], image_size[1]
    )

    # 生成cut_mix的结果
    image_1 = image_1 - zero_padded_image_1
    image = image_1 + zero_padded_image_2

    # Mixup标签信息
    label = lambda_value * label_1 + (1 - lambda_value) * label_2

    return (image, label)


if __name__ == '__main__':
    # 全局化的参数列表
    # ---------------------
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 8
    NUM_EPOCHS = 128
    EARLY_STOP_ROUNDS = 6
    TTA_ROUNDS = 25

    MODEL_NAME = 'EfficentNetB5_dataaug_rtx3090'
    MODEL_LR = 0.00001
    MODEL_LABEL_SMOOTHING = 0

    CKPT_DIR = './ckpt/'
    CKPT_FOLD_NAME = '{}_GPU_{}_{}'.format(TASK_NAME, GPU_ID, MODEL_NAME)

    IS_TRAIN_FROM_CKPT = False
    IS_SEND_MSG_TO_DINGTALK = True
    IS_DEBUG = False
    IS_RANDOM_VISUALIZING_PLOTS = False

    # 数据loading的path
    TRAIN_PATH = './data/train/'
    TEST_PATH = './data/test_images_a/'
    N_CLASSES = 2

    # 利用tensorflow的preprocessing方法读取数据集
    # ---------------------
    train_file_full_name_list = []
    train_label_list = []

    # 0 - Phone
    # *************
    full_path_name = os.path.join(TRAIN_PATH, '0_phone/JPEGImages/')
    for file_name in os.listdir(full_path_name):
        if file_name.endswith('.jpg'):
            train_file_full_name_list.append(
                os.path.join(full_path_name, file_name)
            )
            train_label_list.append(0)

    # 1 - No Phone
    # *************
    full_path_name = os.path.join(TRAIN_PATH, '1_no_phone/')
    for file_name in os.listdir(full_path_name):
        if file_name.endswith('.jpg'):
            train_file_full_name_list.append(
                os.path.join(full_path_name, file_name)
            )
            train_label_list.append(1)
    train_label_oht_array = np.array(train_label_list)

    # 编码训练标签
    # *************
    encoder = OneHotEncoder(sparse=False)
    train_label_oht_array = encoder.fit_transform(
        train_label_oht_array.reshape(-1, 1)).astype(np.float32)

    # 按照比例划分Train与Validation
    # *************
    X_train, X_val, y_train, y_val = train_test_split(
        train_file_full_name_list, train_label_oht_array,
        train_size=0.8, random_state=GLOBAL_RANDOM_SEED,
    )

    n_train_samples, n_valid_samples = len(X_train), len(X_val)

    # 构造训练数据集的pipline, 尝试使用Mixup进行数据增强
    # 参考Keras Mixup tutorial(https://keras.io/examples/vision/mixup/)
    # ************
    processor_train_image = load_preprocessing_img(
        image_size=IMAGE_SIZE, stage='train')

    train_path_ds = tf.data.Dataset.from_tensor_slices(X_train)
    train_img_ds_x = train_path_ds.map(
        processor_train_image, num_parallel_calls=mp.cpu_count()
    )
    train_img_ds_y = train_path_ds.map(
        processor_train_image, num_parallel_calls=mp.cpu_count()
    )
    train_label_ds_x = tf.data.Dataset.from_tensor_slices(y_train)
    train_label_ds_y = tf.data.Dataset.from_tensor_slices(y_train)

    train_ds_x = tf.data.Dataset.zip((train_img_ds_x, train_label_ds_x))
    train_ds_y = tf.data.Dataset.zip((train_img_ds_y, train_label_ds_y))

    # 构造validation数据集的pipeline
    # ************
    processor_valid_image = load_preprocessing_img(
        image_size=IMAGE_SIZE, stage='valid')

    val_path_ds = tf.data.Dataset.from_tensor_slices(X_val)
    val_img_ds = val_path_ds.map(
        processor_valid_image, num_parallel_calls=mp.cpu_count()
    )
    val_label_ds = tf.data.Dataset.from_tensor_slices(y_val)
    val_ds = tf.data.Dataset.zip((val_img_ds, val_label_ds))

    # 数据集性能相关参数（采用mixup进行增强）
    # ************
    train_ds_x = train_ds_x.shuffle(
        BATCH_SIZE * 32).batch(BATCH_SIZE).prefetch(2 * BATCH_SIZE)
    train_ds_y = train_ds_y.shuffle(
        BATCH_SIZE * 32).batch(BATCH_SIZE).prefetch(2 * BATCH_SIZE)
    train_ds = tf.data.Dataset.zip((train_ds_x, train_ds_y))
    train_ds_mu = train_ds.map(
        lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=1.0),
        num_parallel_calls=mp.cpu_count()
    )

    val_ds = val_ds.batch(BATCH_SIZE).prefetch(2 * BATCH_SIZE)

    # 数据集性能相关参数（采用cutmix进行增强）
    # ************
    # train_ds_x = train_ds_x.shuffle(
    #     BATCH_SIZE * 32).batch(BATCH_SIZE).prefetch(8 * BATCH_SIZE)
    # train_ds_y = train_ds_y.shuffle(
    #     BATCH_SIZE * 32).batch(BATCH_SIZE).prefetch(8 * BATCH_SIZE)
    # train_ds = tf.data.Dataset.zip((train_ds_x, train_ds_y))
    # train_ds_mu = train_ds.map(
    #     lambda ds_one, ds_two: cut_mix(ds_one, ds_two, image_size=IMAGE_SIZE),
    #     num_parallel_calls=mp.cpu_count()
    # )

    # val_ds = val_ds.batch(BATCH_SIZE).prefetch(8 * BATCH_SIZE)

    # 随机可视化几张图片
    # ************
    if IS_RANDOM_VISUALIZING_PLOTS:
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds_mu.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                # plt.title(int(labels[i]))
                plt.axis('off')
        plt.tight_layout()

    # 构造与编译Model，并添加各种callback
    # ---------------------

    # 各种Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_tf_f1_score', mode="max",
            verbose=1, patience=EARLY_STOP_ROUNDS,
            restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                CKPT_DIR + CKPT_FOLD_NAME,
                MODEL_NAME + '_latest.ckpt'),
            monitor='val_tf_f1_score',
            mode='max',
            save_weights_only=True,
            save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_tf_f1_score',
                factor=0.7,
                patience=3,
                min_lr=0.000003),
        RemoteMonitorDingTalk(
            is_send_msg=IS_SEND_MSG_TO_DINGTALK,
            model_name=CKPT_FOLD_NAME,
            gpu_id=GPU_ID),
    ]

    # 训练模型
    model = build_efficentnet_model(
        n_classes=N_CLASSES,
        input_shape=IMAGE_SIZE + (3,),
        network_type=MODEL_NAME,
        model_name=MODEL_NAME,
        model_lr=MODEL_LR,
        model_label_smoothing=MODEL_LABEL_SMOOTHING,
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
        train_ds_mu,
        epochs=NUM_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # 生成Test预测结果，并进行Top-1 Accuracy评估
    # ---------------------
    test_file_name_list = os.listdir(TEST_PATH)
    test_file_fullname_list = [TEST_PATH + item for item in test_file_name_list]

    test_path_ds = tf.data.Dataset.from_tensor_slices(test_file_fullname_list)
    processor_test_image = load_preprocessing_img(
        image_size=IMAGE_SIZE, stage='test')
    test_ds = test_path_ds.map(
        processor_test_image,
        num_parallel_calls=mp.cpu_count()
    )
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(buffer_size=int(BATCH_SIZE * 32))

    # 进行TTA强化
    test_pred_proba_list = []
    for i in tqdm(range(TTA_ROUNDS)):
        test_pred_proba_list.append(model.predict(test_ds))
    test_pred_proba = np.mean(test_pred_proba_list, axis=0)
    test_pred_label_list = np.argmax(test_pred_proba, axis=1)

    test_pred_df = pd.DataFrame(
        test_file_name_list,
        columns=['image_name']
    )
    test_pred_df['class_id'] = test_pred_label_list

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