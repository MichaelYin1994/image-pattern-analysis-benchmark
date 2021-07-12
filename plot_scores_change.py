#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106242058
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(plot_score_change.py)绘制赛程期间主要分数变化情况。
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # 提交的基础信息
    col_names = ['sub_date', 'val_score', 'online_score', "description"]
    score_list = [['2021-06-26 02:56:56', 0.22, 0.08785, '第一次正常提交分数，没有采用任何数据增强方法。'],
                  ['2021-06-26 22:50:24', 0.332, 0.11234, '在第一次提交的基础上，在小水管上多跑了一些轮次'],
                  ['2021-06-28 18:26:24', 0.7028, 0.13996, '采用ImageNet预训练的EfficentNet做训练，发现了Bug所在之处：train与val划分有Bug'],
                  ['2021-06-28 22:50:24', 0.7755, 0.14729, '把validation与train的比例做对了，猜测testing的预测顺序有问题'],
                  ['2021-06-29 09:57:24', 0.820, 0.15197, '需要再次check预测程序'],
                  ['2021-07-02 14:23:00', 0.829, 0.8335, '修复了标签loading的bug，采用EfficentNetB3作为base，224*224输入'],
                  ['2021-07-04 10:22:00', 0.835, 0.82696, '采用cosine学习率与简单数据增强，效果似乎不如ReduceLROnPlateau好'],
                  ['2021-07-09 23:25:00', 0.854, 0.85973, '采用更大的EfficentNetB5预训练模型']，
                  ['2021-07-12 23:15:00', 0.842, 0.84974, '采用EfficentNetB5的模型，使用label_smoothing=0.1']]

    # 分数数据预处理
    df = pd.DataFrame(score_list, columns=col_names)
    df['sub_date'] = pd.to_datetime(df['sub_date'])
    df = df.query('online_score > 0.8').reset_index(drop=True)

    # 分数随时间变化图
    # ---------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['val_score'].values, linestyle="--", marker="s", color="k",
            linewidth=2, markersize=4.5, label='validation')
    ax.plot(df['online_score'].values, linestyle="-", marker="o", color="b",
            linewidth=2, markersize=4.5, label='round 1 testing')

    ax.grid(True)
    # ax.set_xlim(0.6, )
    ax.set_ylim(0.8, 0.95)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Top-1 Accuracy", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10)
    plt.tight_layout()
