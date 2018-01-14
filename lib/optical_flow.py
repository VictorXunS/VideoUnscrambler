#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import numpy as np
import cv2
import os
import sys
from parameters import cfg
from lib.imutils import convertFramesToGrayscale, resizeFrames


def getCorners(video_filename, frame_list_gray):

    '''
    Precompute good features to track keypoints
    to speed up double loop in computeOpticalFlowDistances
    '''

    print "Precomputing corners"

    corners_filepath = os.path.join(cfg.DATA_FOLDER,\
                            os.path.splitext(video_filename)[0], 'cornersGFTT.npy')

    if os.path.exists(corners_filepath):
        corners_list = np.load(corners_filepath)
        print "Loaded corners from: ", corners_filepath
    else:
        corners_list = [cv2.goodFeaturesToTrack(frame, mask = None, \
                **cfg.FEATURE_PARAMS) for frame in frame_list_gray]
        np.save(corners_filepath, corners_list)
        print "Computed and saved corners in file: ", corners_filepath

    return corners_list


def computeOpticalFlowDistances(video_filename, scenes_dict, frame_list):

    '''
    Computes pairwise optical flow error between frames of a scene
    '''

    # Precompute frames in grayscale for corner detection
    frame_list_gray = convertFramesToGrayscale(frame_list)

    if cfg.RESIZE_FRAMES_FOR_CORNERS:
        frame_list_gray = resizeFrames(frame_list_gray, cfg.FRAMES_SIZE_FOR_CORNERS)

    # Load or compute list of corners
    corners_list = getCorners(video_filename, frame_list_gray)

    nbScenes = len(scenes_dict)
    scenes_OF_distances_dict = {}
    # Loop over scenes
    for scene in scenes_dict:
        frames_in_scene = scenes_dict[scene]
        nbFramesInScene = len(frames_in_scene)

        print "Computing optical flows for scene: ", scene, ' / ', nbScenes
        OF_distances_frame = np.zeros((nbFramesInScene, nbFramesInScene))
        # Loop over pair of frames in scene:
        for frame_i in range(nbFramesInScene):

            # Print computation advancement
            pctg = int(float(frame_i+1) / nbFramesInScene * 20)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*pctg, 5*pctg))
            sys.stdout.flush()

            # Optical flow is not symmetric so need full double loop
            for frame_ii in range(nbFramesInScene):

                old_gray = frame_list_gray[frames_in_scene[frame_i]]
                frame_gray = frame_list_gray[frames_in_scene[frame_ii]]

                # calculate optical flow error between frames
                _, _, err_list = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray,\
                                            corners_list[frames_in_scene[frame_i]],\
                                            None, **cfg.LK_PARAMS)
                err = np.mean(err_list)
                if err > cfg.MAX_OPTICAL_FLOW_ERR:
                    err = cfg.MAX_OPTICAL_FLOW_ERR
                OF_distances_frame[frame_i, frame_ii] = err

        # Symmetrize optical flow matrix by averaging OF i->ii and ii->i
        if not cfg.USE_DIRECTED_GRAPH:
            OF_distances_frame = np.add(OF_distances_frame, OF_distances_frame.T) / 2
        scenes_OF_distances_dict[scene] = OF_distances_frame


    return scenes_OF_distances_dict


def getOpticalFlowDistances(video_filename, scenes_dict, frame_list):

    '''
    Either load optical flow pairwise distances matrices from file
    or compute them
    '''

    of_matrices_filepath = os.path.join(cfg.DATA_FOLDER,\
                            os.path.splitext(video_filename)[0],\
                            'LK_optical_flow_distances.npy')
    if os.path.exists(of_matrices_filepath):
        scenes_OF_distances_dict = np.load(of_matrices_filepath).item()
        print "Loaded optical flow distance matrices"
    else:

        scenes_OF_distances_dict = computeOpticalFlowDistances(video_filename,\
                                                scenes_dict, frame_list)
        # Create directory if it doesn't exist
        np.save(of_matrices_filepath, scenes_OF_distances_dict)
        print "Computed and saved optical flow distance matrices"

    return scenes_OF_distances_dict