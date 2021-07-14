# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 09:12:58 2021

@author: HP01
"""
import numpy as np
import tensorflow as tf
import backbone
import layer_module
import setup
from setup import *
import utils
from tensorflow.keras import layers

STRIDES = np.array(setup.STRIDES)
ANCHORS = (np.array(setup.ANCHORS).T / STRIDES).T


def yolov4(input_tensor, num_classes):
    """
    Yolov4 head network
    Parameters
    ----------
    input_tensor: tuple (n, h, w ,c)
    num_classes

    Returns
    -------
    conv_sbbox: tuple(n, h, w, c)
    conv_mbbox: tuple(n, h, w, c)
    conv_lbbox: tuple(n, h, w, c)
    """
    route_1, route_2, conv = backbone.cspdarknet53(input_tensor)

    route = conv
    out = layer_module.convolutional_layer(conv, (1, 1, 512, 256))
    out = layer_module.upsample(out)

    route_2 = layer_module.convolutional_layer(route_2, (1, 1, 512, 256))
    out = tf.concat([route_2, out], axis=-1)

    out = layer_module.convolutional_layer(out, (1, 1, 512, 256))
    out = layer_module.convolutional_layer(out, (3, 3, 256, 512))
    out = layer_module.convolutional_layer(out, (1, 1, 512, 256))
    out = layer_module.convolutional_layer(out, (3, 3, 256, 512))
    out = layer_module.convolutional_layer(out, (1, 1, 512, 256))

    route_2 = out
    out = layer_module.convolutional_layer(out, (1, 1, 256, 128))
    out = layer_module.upsample(out)
    route_1 = layer_module.convolutional_layer(route_1, (1, 1, 256, 128))
    out = tf.concat([route_1, out], axis=-1)

    out = layer_module.convolutional_layer(out, (1, 1, 256, 128))
    out = layer_module.convolutional_layer(out, (3, 3, 128, 256))
    out = layer_module.convolutional_layer(out, (1, 1, 256, 128))
    out = layer_module.convolutional_layer(out, (3, 3, 128, 256))
    out = layer_module.convolutional_layer(out, (1, 1, 256, 128))

    route_1 = out
    out = layer_module.convolutional_layer(out, (3, 3, 128, 256))
    conv_sbbox = layer_module.convolutional_layer(out, (1, 1, 256, 3 * (num_classes + 5)), activate=False, bn=False)

    out = layer_module.convolutional_layer(route_1, (3, 3, 128, 256), downsample=True)
    out = tf.concat([out, route_2], axis=-1)

    out = layer_module.convolutional_layer(out, (1, 1, 512, 256))
    out = layer_module.convolutional_layer(out, (3, 3, 256, 512))
    out = layer_module.convolutional_layer(out, (1, 1, 512, 256))
    out = layer_module.convolutional_layer(out, (3, 3, 256, 512))
    out = layer_module.convolutional_layer(out, (1, 1, 512, 256))

    # route_2 = out
    out = layer_module.convolutional_layer(out, (3, 3, 256, 512))
    conv_mbbox = layer_module.convolutional_layer(out, (1, 1, 512, 3 * (num_classes + 5)), activate=False, bn=False)

    out = layer_module.convolutional_layer(route_1, (3, 3, 256, 512), downsample=True)
    out = tf.concat([out, route], axis=-1)

    out = layer_module.convolutional_layer(out, (1, 1, 1024, 512))
    out = layer_module.convolutional_layer(out, (3, 3, 512, 1024))
    out = layer_module.convolutional_layer(out, (1, 1, 1024, 512))
    out = layer_module.convolutional_layer(out, (3, 3, 512, 1024))
    out = layer_module.convolutional_layer(out, (1, 1, 1024, 512))

    out = layer_module.convolutional_layer(out, (3, 3, 512, 1024))
    conv_lbbox = layer_module.convolutional_layer(out, (1, 1, 1024, 3 * (num_classes + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(output_tensor, num_classes, i=0):
    """
    This function decode the channel information of the feature map(Output of the Yolov4)
    Parameters
    ----------
    output_tensor
    num_classes
    i

    Returns
    -------
    pred_xywh: prediction coordinates of the prediction box (x, y, w, h)
    pred_conf: prediction confidence
    pred_prob: prediction probability
    """
    conv_shape = tf.shape(output_tensor)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    output_tensor = tf.reshape(output_tensor, (batch_size, output_size, output_size, 3, 5 + num_classes))
    """
    #conv_raw_dxdy: offset of center position     
    #conv_raw_dwdh: Prediction box length and width offset
    #conv_raw_conf: confidence of the prediction box
    #conv_raw_prob: category probability of the prediction box
    """
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(output_tensor, (2, 2, 1, num_classes),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    # object box calculates the predicted confidence
    pred_conf = tf.sigmoid(conv_raw_conf)
    # calculating the predicted probability category box object
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def create_yolo(class_path, input_size=416, channels=3, training=False):
    """

    Parameters
    ----------
    class_path: str - class directory
    input_size: int - height and width of the input image
    channels: int - number of input channels
    training: bool

    Returns
    -------
    yolo: tensorflow model
    """
    num_classes = utils.read_class_names(class_path)
    input_layer = layers.Input([input_size, input_size, channels])
    conv_tensors = yolov4(input_layer, num_classes)
    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, num_classes, i)
        if training:
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    yolo = tf.keras.Model(input_layer, output_tensors)
    return yolo


def bbox_iou(boxes1, boxes2):
    """

    Parameters
    ----------
    boxes1: boundary box 1
    boxes2: boundary box 2

    Returns
    -------

    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area
    return iou


def bbox_giou(boxes1, boxes2):
    # Calculate the iou value between the two bounding boxes
    iou, union_area = bbox_iou(boxes1, boxes2)

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex
    # surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bbox_iou(boxes1, boxes2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (boxes1[..., 1] - boxes2[..., 1]) * (
            boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]

    ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term


def loss(pred, conv, label, bbox, classes, i=0):
    num_classes = len(utils.read_class_names(classes))
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_classes))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bbox[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    # Find the value IoU with the real box the largest prediction box
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    # If the largest IoU is less than the threshold, it will be considered as contained no objects
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_THRESH, tf.float32)
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Confidence Loss
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
