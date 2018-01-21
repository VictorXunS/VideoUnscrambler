#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import sys
import cv2
import random
import numpy as np

from parameters import cfg
from lib.file_manager import archivePreviousRun
from lib.video_handler import readFramesFromVideo, storeFramesAsVideo
from lib.visual_hash import getVisualHashes, clusterScenesUsingHash
from lib.optical_flow import getOpticalFlowDistances
from lib.permutate_frames import permutateFramesWithinScenes
from lib.permutate_scenes import permutateScenes


def unscrambleVideo(corrupted_video_filename, verbose):

    ''' Main function '''

    # Copy previous run data into an archive folder
    if cfg.ARCHIVE_PREVIOUS_RUN:
        archivePreviousRun(corrupted_video_filename)

    # Read video from file and store frames in a numpy array
    frame_list, nbFrames = readFramesFromVideo(corrupted_video_filename)

    if cfg.SHUFFLE_INPUT_FRAMES:
        random.seed(cfg.SHUFFLE_SEED)
        random.shuffle(frame_list)

    ##### CLUSTERING FRAMES INTO SCENES
    # Get perceptual hashes of each frame in video
    visual_hash_list = getVisualHashes(corrupted_video_filename, frame_list)

    # Hierarchical clustering of frames in scenes based on hash distances
    scenes_dict, nbScenes, hash_distances =\
        clusterScenesUsingHash(visual_hash_list, nbFrames)
    if verbose:
        print "Clusterized scenes:\n", scenes_dict


    # # Result when only hash distance matrix is used
    # # Comment next two function if you want to test this
    # hash_distances_dict = {}
    # hash_distances_dict[1] = hash_distances
    # permutated_frames_dict = permutateFramesWithinScenes(hash_distances_dict)
    # scenes_dict = {}
    # scenes_dict[1] = range(len(frame_list))


    ##### OPTIMIZING FRAMES PERMUTATION WITHIN EACH SCENE
    # For each scene, compute the matrix of Lukas-Kanade optical
    # flow distances between frames
    OF_distances_dict = getOpticalFlowDistances(corrupted_video_filename,
                                                scenes_dict, frame_list)
    if verbose:
        print "Optical flow distances:\n", OF_distances_dict

    # Optimize permutation of frames within each scene
    permutated_frames_dict = permutateFramesWithinScenes(OF_distances_dict)
    if verbose:
        print "Permutated frames:\n", permutated_frames_dict


    ##### ARRANGING SCENES ORDER
    # Permutate scenes in order to get final solution
    # Hard problem (use semantic etc...)
    final_permutation = permutateScenes(scenes_dict, permutated_frames_dict)
    if verbose:
        print "Final permutation:\n", final_permutation


    # # Create some random colors
    # color = np.random.randint(0,255,(100,3))
    # old_frame = frame_list[final_permutation[80]]
    # old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, \
    #             **cfg.FEATURE_PARAMS)
    # # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)
    # # Show result
    # for frame_id in final_permutation[80:100]:
    #     # cv2.imshow('frame', frame_list[frame_id])
    #     # cv2.waitKey(0)

    #     frame = frame_list[frame_id]
    #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # calculate optical flow
    #     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0,\
    #                                                   None, **cfg.LK_PARAMS)
    #     print np.mean(err)
    #     # Select good points
    #     good_new = p1[st==1]
    #     good_old = p0[st==1]

    #     # draw the tracks
    #     for i,(new,old) in enumerate(zip(good_new,good_old)):
    #         a,b = new.ravel()
    #         c,d = old.ravel()
    #         mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #         frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    #     img = cv2.add(frame,mask)

    #     cv2.imshow('frame',img)
    #     cv2.waitKey(0)


    # Save new frames permutation as video
    storeFramesAsVideo(corrupted_video_filename, final_permutation, frame_list)


if __name__ == '__main__':
    nbArguments = len(sys.argv)
    corrupted_video_filename = sys.argv[1]
    if nbArguments == 3:
        verbose = int(sys.argv[2])
    else:
        verbose = False
    unscrambleVideo(corrupted_video_filename, verbose)
