#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import numpy as np
import os, cv2, glob
import argparse
from os import path as osp

from nets.vgg16 import vgg16

CLASSES = ('__background__',
           'V', 'WJ_7', 'W300-1', 'negative')
COLOR = {'V': (125,125,0), 'WJ_7': (125,125,0), 'W300-1': (255,255,255), 'yz': (125,125,0), 'negative': (0,255,255)}

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, det_info, thresh=0.5):
    # im = im[:, :, (2, 1, 0)]
    # im = cv2.imread(im_name)
    for class_name, dets in det_info:
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue


        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            lx, ly, rx, ry = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            text_show = '%s %.4f' % (class_name, score)
            # img name, left-top, right-down, color, line-width
            color = COLOR.get(class_name, (255, 255, 255))
            cv2.rectangle(im, (lx, ly), (rx, ry), color, 2)
            cv2.putText(im, text_show, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness=1)
    return im

def _vis_detections(im, det_info, thresh=0.5):
    for class_name, dets in det_info:
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue


        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            lx, ly, rx, ry = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            center_w, center_h = int((lx + rx) / 2), int((ly + ry) / 2)

            text_show = '%s %.4f' % (class_name, score)
            # img name, left-top, right-down, color, line-width
            cv2.rectangle(im, (center_w-36, center_h-52), (center_w+36, center_h+52), (0, 255, 255), 2)
            cv2.putText(im, text_show, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness=1)
    return im

def demo(sess, net, image_name, CONF_THRESH=0.5, save_path=''):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    det_info = []
    # Visualize detections for each class
    # CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        det_info += [[cls, dets]]
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)

    save_im = vis_detections(im, det_info, thresh=CONF_THRESH)
    basename = osp.basename(image_name)
    cv2.imwrite(filename=osp.join(save_path, basename), img=save_im)
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    detpath = '/home/wanghao/tf-faster-rcnn/data/demo/new-data'
    reg = '*.JPG'
    save_path = 'data/draw/'
    # model path
    demonet = args.demo_net
    # dataset = args.dataset
    # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
    #                           NETS[demonet][0])
    tfmodel = 'output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_20000.ckpt'


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        raise NotImplementedError
    net.create_architecture(sess, "TEST", len(CLASSES),
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    im_names = glob.glob(osp.join(detpath, reg))
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('detect for {}'.format(im_name))
        demo(sess, net, im_name, CONF_THRESH=0.5, save_path=save_path)
