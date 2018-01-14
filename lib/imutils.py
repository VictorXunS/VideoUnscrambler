#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import numpy as np
import cv2

def resizeFrames(frame_list, target_size):

    '''
    Resize all images in list so that max dimension
    is less than target
    '''

    im_size_max = np.max(frame_list[0].shape[:2])
    im_scale = float(target_size) / float(im_size_max)

    frame_list_resized = []
    for frame in frame_list:
        frame_list_resized.append(cv2.resize(frame, None, None, fx=im_scale,
                                fy=im_scale, interpolation=cv2.INTER_LINEAR))

    return frame_list_resized


def convertFramesToGrayscale(frame_list):

    '''
    Convert all images in list to grayscale
    '''

    frame_list_gray = []
    for frame in frame_list:
        frame_list_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    return frame_list_gray

