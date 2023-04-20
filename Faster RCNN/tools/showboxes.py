# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys
import numpy as np
import time
sys.path.append("../")
from libs.configs import cfgs
from libs.networks import build_whole_network_boxes
from data.io.read_tfrecord import next_batch
from libs.box_utils import show_box_in_tensor
from help_utils import tools
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP


def train():

    faster_rcnn = build_whole_network_boxes.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=True)

    with tf.name_scope('get_batch'):
        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                       batch_size=cfgs.BATCH_SIZE,
                       shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                       is_training=True)
        gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 5])

    biases_regularizer = tf.no_regularizer
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfgs.WEIGHT_DECAY)

    # list as many types of layers as possible, even if they are not used now
    with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                         slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer,
                        biases_initializer=tf.constant_initializer(0.0)):
        all_anchors, posi_boxes = faster_rcnn.build_whole_detection_network(
            input_img_batch=img_batch,
            gtboxes_batch=gtboxes_and_label)



    gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=img_batch,
                                                                   boxes=gtboxes_and_label[:, :-1],
                                                                   labels=gtboxes_and_label[:, -1])



    # ___________________________________________________________________________________________________add summary



    # ---------------------------------------------------------------------------------------------compute gradients


    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = faster_rcnn.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)



        for step in range(cfgs.MAX_ITERATION):
            a_img_name,o_boxes, m_boxes,gt_img = \
                sess.run(
                    [img_name_batch,all_anchors, posi_boxes,gtboxes_in_img])
            a_img_name=str(a_img_name)
            a_img_name=a_img_name.strip("[b\\'").strip("\\']")
            m_boxes=m_boxes[0,:,:,:]
            m_boxes=m_boxes[:,:,::-1]
            o_boxes=o_boxes[0,:,:,:]
            o_boxes=o_boxes[:,:,::-1]
            gt_img = gt_img[0, :, :, :]
            gt_img = gt_img[:, :, ::-1]
            if not os.path.exists(cfgs.BOXES_SAVE_PATH):
                    os.makedirs(cfgs.BOXES_SAVE_PATH)
            Call_path=os.path.join(cfgs.BOXES_SAVE_PATH,'po_rois')
            if not os.path.exists(Call_path):
                    os.makedirs(Call_path)
            gt_path = os.path.join(cfgs.BOXES_SAVE_PATH, 'gt')
            if not os.path.exists(gt_path):
                os.makedirs(gt_path)
            Ori_path = os.path.join(cfgs.BOXES_SAVE_PATH, 'anchors')
            if not os.path.exists(Ori_path):
                os.makedirs(Ori_path)
            cv2.imwrite(Call_path + '/' + a_img_name,
                           m_boxes)
            cv2.imwrite(Ori_path + '/' + a_img_name,
                        o_boxes)
            cv2.imwrite(gt_path + '/' + a_img_name,
                        gt_img)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    train()

#
















