#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106112134
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(visualizing_data.py)对图片数据进行基础的一些统计分析：
- 标签的分布情况
- 对训练与验证中的图像进行可视化
'''

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)

# ----------------------------------------------------------------------------
def load_raw_labels():
    '''从原始数据中载入并解析标签数据，返回train与val的标签数据。'''
    PATH = './data/Label/label/'

    # 载入训练集标签
    with open(PATH+'train_finetune.txt', 'r') as f:
        train_labels = f.readlines()
    train_labels = [item.split("/") for item in train_labels]

    train_labels_list = []
    for item in train_labels:
        tmp_list = []
        tmp_list.append(item[0])
        tmp_list.extend(item[1].strip('\n').split(' '))

        train_labels_list.append(tmp_list)

    train_labels_df = pd.DataFrame(
        train_labels_list, columns=['dir_id', 'file_name', 'label_id'])
    train_labels_df['dir_id'] = train_labels_df['dir_id'].astype('int')
    train_labels_df['label_id'] = train_labels_df['label_id'].astype('int')

    # 载入验证集标签
    with open(PATH+'val_finetune.txt', 'r') as f:
        valid_labels = f.readlines()
    valid_labels = [item.split("/") for item in valid_labels]

    valid_labels_list = []
    for item in valid_labels:
        tmp_list = []
        tmp_list.append(item[0])
        tmp_list.extend(item[1].strip('\n').split(' '))

        valid_labels_list.append(tmp_list)

    valid_labels_df = pd.DataFrame(
        valid_labels_list, columns=['dir_id', 'file_name', 'label_id'])
    valid_labels_df['dir_id'] = valid_labels_df['dir_id'].astype('int')
    valid_labels_df['label_id'] = valid_labels_df['label_id'].astype('int')

    return train_labels_df, valid_labels_df


def load_train_class_k_img(k=0):
    '''从train文件夹中读取类标签为k的所有图片。'''
    PATH = './data/Train/'

    if k < 0 or k > 999:
        raise ValueError('k range from 0 to 999')

    # load文件夹下的所有图像
    PATH = '{}/{}/'.format(PATH, k)
    file_names = os.listdir(PATH)

    images_list = []
    for file_name in file_names:
        img = cv2.imread('{}/{}'.format(PATH, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_list.append(img)

    return images_list


def load_valid_class_k_img(k=0):
    '''从Val文件夹中读取类标签为k的所有图片。'''
    PATH = './data/Val/'

    if k < 0 or k > 999:
        raise ValueError('k range from 0 to 999')

    # load文件夹下的所有图像
    PATH = '{}/{}/'.format(PATH, k)
    file_names = os.listdir(PATH)

    images_list = []
    for file_name in file_names:
        img = cv2.imread('{}/{}'.format(PATH, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_list.append(img)

    return images_list


if __name__ == '__main__':
    # 载入训练与测试的类别标签
    # ---------------------
    train_labels_df, valid_labels_df = load_raw_labels()

    train_labels_feat = train_labels_df.groupby(
        ['label_id'])[['file_name']].count()
    train_labels_feat.rename(
        {'file_name': 'label_count'}, axis=1, inplace=True)
    train_labels_feat.reset_index(inplace=True)
    train_labels_feat.sort_values(
        by=["label_id"], ascending=True, inplace=True)
    train_labels_feat.reset_index(inplace=True, drop=True)

    valid_labels_feat = valid_labels_df.groupby(
        ['label_id'])[['file_name']].count()
    valid_labels_feat.rename(
        {'file_name': 'label_count'}, axis=1, inplace=True)
    valid_labels_feat.reset_index(inplace=True)
    valid_labels_feat.sort_values(
        by=["label_id"], ascending=True, inplace=True)
    valid_labels_feat.reset_index(inplace=True, drop=True)

    # 可视化类别的分布
    # ---------------------
    fig, ax_objs = plt.subplots(3, 1, figsize=(16, 12))
    ax_objs = ax_objs.ravel()

    # train与valid各个类别占总样本的比例
    ax = ax_objs[0]
    ax.plot(
        train_labels_feat['label_count'].values / train_labels_feat['label_count'].sum(),
        color='b', label='train', linewidth=1.5,
        marker='s', markersize=3, linestyle='--')
    ax.plot(
        valid_labels_feat['label_count'].values / valid_labels_feat['label_count'].sum(),
        color='k', label='valid', linewidth=1.5,
        marker='o', markersize=3, linestyle='--')

    ax.set_xlim(0, 999+1)
    ax.set_ylim(0,)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True)
    ax.legend(fontsize=10, loc='best')

    # train各个类别样本的个数
    ax = ax_objs[1]
    ax.plot(
        train_labels_feat['label_count'].values,
        color='b', label='train', linewidth=1.5,
        marker='s', markersize=3, linestyle='--')

    ax.set_xlim(0, 999+1)
    ax.set_ylim(0,)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True)
    ax.legend(fontsize=10, loc='best')

    # valid各个类别样本的个数
    ax = ax_objs[2]
    ax.plot(
        valid_labels_feat['label_count'].values,
        color='k', label='valid', linewidth=1.5,
        marker='s', markersize=3, linestyle='--')

    ax.set_xlim(0, 999+1)
    ax.set_ylim(0,)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()

    # 载入train与valid中的第k类
    # ---------------------
    k = 233
    train_images_list = load_train_class_k_img(k=k)
    valid_images_list = load_valid_class_k_img(k=k)

    # 随机可视化train中的一张图片
    # ---------------------
    fig, ax = plt.subplots()
    rank_idx = np.random.randint(0, len(train_images_list))
    ax.imshow(train_images_list[rank_idx])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # 随机可视化valid中的一张图片
    # ---------------------
    fig, ax = plt.subplots()
    rank_idx = np.random.randint(0, len(valid_images_list))
    ax.imshow(valid_images_list[rank_idx])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
