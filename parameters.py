#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import os
import cv2

class Object(object):
    pass

cfg = Object()

##### DATA PATH #####
current_path = os.path.dirname(os.path.realpath(__file__))
cfg.DATA_FOLDER = os.path.join(current_path, 'Data/')
cfg.OUTPUT_VIDEOS_FOLDER = os.path.join(current_path, 'Output/')

##### GENERAL #####
cfg.HASH_NAME = 'aHash' # aHash, pHash, dHash, wHash
cfg.ARCHIVE_PREVIOUS_RUN = False
cfg.SHUFFLE_INPUT_FRAMES = False # Further shuffle input frames
cfg.SHUFFLE_SEED = 42
cfg.REMOVE_BAD_SCENES = True # Remove scenes that have to few frames
cfg.MINIMAL_NB_FRAMES_PER_SCENE = 5

##### TSP SOLVER #####
cfg.TSP_SOLVER = 'greedy' # greedy, 2opt
cfg.USE_DUMMY_NODE = False # Dummy node to simulate acyclic TSP
cfg.REFINE_GREEDY_WITH_2OPT = True # Use greedy, then 2opt

##### OPTICAL FLOW #####
cfg.RESIZE_FRAMES_FOR_CORNERS = False
cfg.FRAMES_SIZE_FOR_CORNERS = 600

# params for ShiTomasi/GFTT corner detection
cfg.FEATURE_PARAMS = dict(maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)

# Parameters for Lucas-Kanade optical flow
cfg.LK_PARAMS = dict(winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

cfg.MAX_OPTICAL_FLOW_ERR = 10000
cfg.USE_DIRECTED_GRAPH = False
