#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106112134
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(models.py)构建并编译各种类型的神经网络模型。此模块大部分代码来自keras application的
部分模块[1]，但是在细节上做了适应。

@References:
----------
[1] https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py
[2] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
[3] He, Kaiming, et al. "Identity mappings in deep residual networks." European conference on computer vision. Springer, Cham, 2016.
'''

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
def build_model_resnet101_v2(layer_input, is_use_bias=True):
    '''
    用Residual Module V2构造一个未编译的Resnet 101模型。

    @Args:
    ----------
    layer_input: {tensor-like}
        输入的tensor，是keras的layer。
    is_use_bias: {bool-like}
        是否在特征抽取的conv层使用bias。

    @Returns:
    ----------
    构造好的未编译的Resnet101 V2模型。
    '''
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # Conv特征抽取层
    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(layer_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=is_use_bias, name='conv1_conv')(x)

    # Pre-activation && Pooling
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    # 残差组件堆叠
    x = residual_module_v2(x, n_blocks=3, n_filters=64, name='conv2')
    x = residual_module_v2(x, n_blocks=4, n_filters=128, name='conv3')
    x = residual_module_v2(x, n_blocks=23, n_filters=256, name='conv4')
    x = residual_module_v2(x, n_blocks=3, n_filters=512, name='conv5')

    return x


def build_model_resnet50_v2(layer_input, is_use_bias=False):
    '''
    用Residual Module V2构造一个未编译的Resnet 50模型。

    @Args:
    ----------
    layer_input: {tensor-like}
        输入的tensor，是keras的layer。
    is_use_bias: {bool-like}
        是否在特征抽取的conv层使用bias。

    @Returns:
    ----------
    构造好的未编译的Resnet50 V2模型。
    '''
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # Conv特征抽取层
    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(layer_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=is_use_bias, name='conv1_conv')(x)

    # Pre-activation && Pooling
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    # 残差组件堆叠
    x = residual_module_v2(x, n_blocks=3, n_filters=64, name='conv2')
    x = residual_module_v2(x, n_blocks=4, n_filters=128, name='conv3')
    x = residual_module_v2(x, n_blocks=6, n_filters=256, name='conv4')
    x = residual_module_v2(x, n_blocks=3, n_filters=512, name='conv5')

    return x


def residual_block_v1(
        x, n_filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    '''
    一个基础的残差模块，其中输入默认为channel last模式，结构来源于文献[2]，采用Bottleneck结构。

    @Args:
    ----------
    x: {tensor-like}
        输入的tensor，可以是keras的layer。
    n_filters: {int-like}
        残差模块Bottleneck结构的filters的数量。
    kernel_size: {int-like}
        残差模块Bottleneck结构的kernel的数量。
    stride: {int-like}
        第一层的stride的大小。
    conv_shortcut: {bool-like}
        是否使用1 * 1的conv层作为short-cut用于升维，对应于论文[2]中的conv通道连接方式。
    name: {str-like}
        block的每一层layer的name prefix。

    @Returns:
    ----------
    构造好的一个残差模块。
    '''
    if stride != 1 and conv_shortcut == False:
        raise ValueError('Input shape mismatch with the shortcut shape !')

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if conv_shortcut:
        # 残差通道(1 * 1的层做维度适配，n_filters * 4)
        shortcut = layers.Conv2D(
            4 * n_filters, 1, strides=stride, name=name + '_shortcut_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_shortcut_0_bn')(shortcut)
    else:
        shortcut = x

    # 降维(1 * 1的层，n_filters不变)
    x = layers.Conv2D(n_filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    # 特征抽取(3 * 3的层，n_filters不变)
    x = layers.Conv2D(
        n_filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    # 升维 + 残差连接(1 * 1的层，n_filters * 4)
    x = layers.Conv2D(4 * n_filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x


def residual_block_v2(
        x, n_filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    '''
    一个基础的残差模块，其中输入默认为channel last模式，结构来源于文献[3]，采用Bottleneck结构。

    @Args:
    ----------
    x: {tensor-like}
        输入的tensor，可以是keras的layer。
    n_filters: {int-like}
        残差模块Bottleneck结构的filters的数量。
    kernel_size: {int-like}
        残差模块Bottleneck结构的kernel的数量。
    stride: {int-like}
        第一层的stride的大小。
    conv_shortcut: {bool-like}
        是否使用1 * 1的conv层作为short-cut用于升维，对应于论文[2]中的conv通道连接方式。
    name: {str-like}
        block的每一层layer的name prefix。

    @Returns:
    ----------
    构造好的一个残差模块。
    '''
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    preact = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        # 残差通道(1 * 1的层做维度适配，stride == 2，n_filters * 4)
        shortcut = layers.Conv2D(
            4 * n_filters, 1, strides=stride, name=name + '_shortcut_0_conv')(preact)
    else:
        # FIXME(zhuoyin94@163.com): 此处的MaxPooling2D的作用是？
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(
        n_filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(
        n_filters,
        kernel_size,
        strides=stride,
        use_bias=False,
        name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * n_filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])

    return x


def residual_module_v1(x, n_filters, n_blocks, stride=2, name=None):
    '''
    一个基础的残差组件，由一系列的残差模块组成，其中输入默认为channel last模式。

    @Args:
    ----------
    x: {tensor-like}
        输入的tensor，可以是keras的layer。
    n_filters: {int-like}
        残差组件Bottleneck结构的filters的数量。
    n_blocks: {int-like}
        残差组件的block的数量。
    stride: {int-like}
        残差组件的第一层的stride的大小。
    name: {str-like}
        block的每一层layer的name prefix。

    @Returns:
    ----------
    一个构造好的残差组件。
    '''
    # 基础特征图抽取
    x = residual_block_v1(x, n_filters, conv_shortcut=True, stride=stride, name=name + '_block1')

    # Residual block叠加，残差连接没有升维操作
    for i in range(2, n_blocks + 1):
        x = residual_block_v1(
            x, n_filters, stride=1, conv_shortcut=False,
            name=name + '_block' + str(i))

    return x


def residual_module_v2(x, n_filters, n_blocks, stride=2, name=None):
    '''
    一个基础的残差组件，由一系列的残差模块组成，其中输入默认为channel last模式。

    @Args:
    ----------
    x: {tensor-like}
        输入的tensor，可以是keras的layer。
    n_filters: {int-like}
        残差组件Bottleneck结构的filters的数量。
    n_blocks: {int-like}
        残差组件的block的数量。
    stride: {int-like}
        残差组件的第一层的stride的大小。
    name: {str-like}
        block的每一层layer的name prefix。

    @Returns:
    ----------
    一个构造好的残差组件。
    '''
    # 基础特征图抽取
    x = residual_block_v2(x, n_filters, conv_shortcut=True, name=name + '_block1')

    for i in range(2, n_blocks):
        x = residual_block_v2(x, n_filters, name=name + '_block' + str(i))

    x = residual_block_v2(x, n_filters, stride=stride, name=name + '_block' + str(n_blocks))

    return x
