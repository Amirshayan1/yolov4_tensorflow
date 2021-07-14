# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:32:38 2021

@author: Amirshayan Tatari
@Email: amirshayan.tatari@tuh.de
"""

# Libraries
import tensorflow as tf
from tensorflow.keras import layers


# Mish activation function 
def mish(x):
    """
    Parameters
    ----------
    x : output of BN
        Perform the mish activation function.

    Returns
    -------
    tf.tensor
    """
    return x * tf.math.tanh(tf.math.softplus(x))


# layer function
def convolutional_layer(input_tensor, filters, bn=True, downsample=False, activate=True, activation='leaky'):
    """
    Parameters
    ----------
    input_tensor : (N,H,W,C)
    filters: tuple - (kernel size, ?, in_num_filter, out_num_filter)
    bn: boolean -  Batch normalization
    downsample: boolean - checks if downsample should apply
    activate: boolean - check if activation function is needed
    activation: str - type of activation function LeakyReLU or Mish

    Returns
    -------
    tf.tensor
    """
    # Zero padding
    if downsample:
        input_tensor = layers.ZeroPadding2D(((1, 0), (1, 0)))(input_tensor)
        padding = 'valid'
        strides = 2

    else:
        strides = 1
        padding = 'same'

    # Conv2D layer 
    conv = layers.Conv2D(filters=filters[-1], kernel_size=filters[0],
                         strides=strides, padding=padding, use_bias=False,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                         bias_initializer=tf.constant_initializer(0.))(input_tensor)

    # Batch Normalization BN 
    if bn: conv = layers.BatchNormalization()(conv)

    # Activation Function (mish or leaky)
    if activate:
        if activation == 'mish':
            conv = mish(conv)
        elif activation == 'leaky':
            conv = layers.LeakyReLU(conv, alpha=0.1)

    return conv


def resblock(input_tensor, input_filter, n_filter_1, n_filter_2, activation='leaky'):
    """
    Parameters
    ----------
    input_tensor : (N,H,W,C)
    input_filter: int - number of input of filters for first layer
    n_filter_1: int - number of output filters for first layer
    n_filter_2: int - number of output filters for second layer
    activation: str - type of activation function LeakyReLU or Mish

    Returns
    -------
    tf.tensor
    """
    residual = input_tensor
    conv = convolutional_layer(input_tensor, filters=(1, 1, input_filter, n_filter_1), activation=activation)
    conv = convolutional_layer(conv, filters=(3, 3, n_filter_1, n_filter_2), activation=activation)
    output = conv + residual
    return output


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
