#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107061103
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(utils.py)为部分深度学习工具的实现。
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


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
