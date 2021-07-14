# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 12:03:07 2021

@author: Amirshayan Tatari
@Email: amirshayan.tatari@tuhh.de

"""
# Libraries
import tensorflow as tf
from tensorflow.keras import layers
import layer_module


def layerblock(input_layer, n_filter, iteration, activation):
    """
            Parameters
            ----------
            input_layer: (N,H,W,C)
            n_filter: int - input channel
            iteration: int - number of using resblock
            activation: str - type of activation function

            Returns
            -------
            tf.tensor: out, route
    """
    double_filter = 2 * n_filter
    out = layer_module.convolutional_layer(input_layer, (1, 1, n_filter, n_filter), activation=activation)
    route = out
    # Downsample
    out = layer_module.convolutional_layer(out, (3, 3, n_filter, double_filter), downsample=True, activation=activation)

    x = out
    x = layer_module.convolutional_layer(x, (1, 1, double_filter, n_filter), activation=activation)

    out = layer_module.convolutional_layer(out, (1, 1, double_filter, n_filter), activation=activation)
    # Resblock
    for _ in range(iteration):
        out = layer_module.resblock(out, n_filter, n_filter, n_filter, activation=activation)

    out = layer_module.convolutional_layer(out, 1, 1, n_filter, n_filter, activation=activation)
    # Concatenating
    out = tf.concat([out, x], axis=-1)

    return out, route


def cspdarknet53(input_tensor):
    """
            Parameters
            ----------
            input_tensor: (N,H,W,C)

            Returns
            -------
            tf.tensor: route_1, route_2, out
    """
    out = layer_module.convolutional_layer(input_tensor, (3, 3, 3, 32), activation="mish")
    # First downsample
    out = layer_module.convolutional_layer(out, (3, 3, 32, 64), downsample=True, activation="mish")

    x = out
    x = layer_module.convolutional_layer(x, (1, 1, 64, 64), activation='mish')

    out = layer_module.convolutional_layer(out, (1, 1, 64, 64), activation='mish')
    # Resblock
    for _ in range(1):
        out = layer_module.resblock(out, 64, 32, 64, activation="mish")

    out = layer_module.convolutional_layer(out, (1, 1, 64, 64), activation='mish')
    # Concatenating
    out = tf.concat([out, x], axis=-1)

    out, _ = layerblock(out, 64, 2, activation='mish')

    out, _ = layerblock(out, 128, 8, activation='mish')

    out, route_1 = layerblock(out, 256, 8, activation='mish')

    out, route_2 = layerblock(out, 512, 4, activation='mish')

    out = layer_module.convolutional_layer(out, (1, 1, 1024, 1024), activation="mish")
    out = layer_module.convolutional_layer(out, (1, 1, 1024, 512))
    out = layer_module.convolutional_layer(out, (3, 3, 512, 1024))
    out = layer_module.convolutional_layer(out, (1, 1, 1024, 512))

    max_pooling_1 = layers.MaxPooling2D(pool_size=13, padding='same', strides=1)(out)
    max_pooling_2 = layers.MaxPooling2D(pool_size=9, padding='same', strides=1)(out)
    max_pooling_3 = layers.MaxPooling2D(pool_size=5, padding='same', strides=1)(out)

    out = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, out], axis=-1)

    out = layer_module.convolutional_layer(out, (1, 1, 2048, 512))
    out = layer_module.convolutional_layer(out, (3, 3, 512, 1024))
    out = layer_module.convolutional_layer(out, (1, 1, 1024, 512))

    return route_1, route_2, out
