#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import sys
import cv2
import random

from parameters import cfg
from lib.video_handler import readFramesFromVideo, storeFramesAsVideo
from lib.visual_hash import getVisualHashes, clusterScenesUsingHash, removeBadScenes
from lib.optical_flow import getOpticalFlowDistances
from lib.permutate_frames import permutateFramesWithinScenes
from lib.permutate_scenes import permutateScenes


def unscrambleVideo(corrupted_video_filename, verbose):

    # Read video from file and store frames in a numpy array
    frame_list, nbFrames = readFramesFromVideo(corrupted_video_filename)

    if cfg.SHUFFLE_INPUT_FRAMES:
        random.seed(cfg.SHUFFLE_SEED)
        random.shuffle(frame_list)
        print frame_list

    ##### CLUSTERING FRAMES INTO SCENES
    # Get perceptual hashes of each frame in video
    aHash_list, pHash_list, dHash_list, wHash_list =\
                        getVisualHashes(corrupted_video_filename, frame_list)

    # Hierarchical clustering of frames in scenes based on hash distances
    scenes_dict, nbScenes = clusterScenesUsingHash(aHash_list, nbFrames)
    # scenes_dict, nbScenes = clusterScenesUsingHash(pHash_list, nbFrames)
    # scenes_dict, nbScenes = clusterScenesUsingHash(dHash_list, nbFrames)
    # scenes_dict, nbScenes = clusterScenesUsingHash(wHash_list, nbFrames)
    if verbose:
        print "Clusterized scenes:\n", scenes_dict


    ##### OPTIMIZING FRAMES PERMUTATION WITHIN EACH SCENE
    # For each scene, compute the matrix of Lukas-Kanade optical flow distances between frames
    OF_distances_dict = getOpticalFlowDistances(corrupted_video_filename,\
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

    # # Show result
    # for frame_id in final_permutation:
    #     cv2.imshow('frame', frame_list[frame_id])
    #     cv2.waitKey(0)

    # Save new frames permutation as video
    storeFramesAsVideo(corrupted_video_filename, final_permutation, frame_list)

    print "Computation done"

if __name__ == '__main__':
    nbArguments = len(sys.argv)
    corrupted_video_filename = sys.argv[1]
    if nbArguments==3:
        verbose = int(sys.argv[2])
    else:
        verbose = False
    unscrambleVideo(corrupted_video_filename, verbose)