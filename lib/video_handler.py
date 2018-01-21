#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import os
import cv2
import numpy as np

from parameters import cfg

def readFramesFromVideo(video_filename):

    '''
    Read video and store frames into list of images
    '''

    corrupted_video_path = os.path.join(cfg.DATA_FOLDER,\
                            os.path.splitext(video_filename)[0], video_filename)
    print "Loading frames from video: ", corrupted_video_path

    cap = cv2.VideoCapture(corrupted_video_path)

    frame_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        frame_list.append(frame)

    nbFrames = len(frame_list)
    print "     Frames extracted: ", nbFrames
    return frame_list, nbFrames

def storeFramesAsVideo(video_filename, final_permutation, frame_list):

    '''
    Store list of images as video
    '''

    if not os.path.exists(cfg.OUTPUT_VIDEOS_FOLDER):
        os.mkdir(cfg.OUTPUT_VIDEOS_FOLDER)

    unscrambled_video_path = cfg.OUTPUT_VIDEOS_FOLDER + video_filename
    # Change file extension to avi
    unscrambled_video_path = os.path.splitext(unscrambled_video_path)[0] + '.avi'

    width, height = frame_list[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(unscrambled_video_path, fourcc, 24.0, (height,width))

    for frame_id in final_permutation:
        video.write(frame_list[frame_id])

    cv2.destroyAllWindows()
    video.release()

    print "\nComputation done, video saved at: ", unscrambled_video_path
