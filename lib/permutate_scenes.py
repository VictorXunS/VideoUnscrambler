#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import numpy as np
import cv2

def permutateScenes(scenes_dict, permutated_frames_dict):

	'''
	This function doesn't do anything special
	Finding scene order is a very hard problem in general
	'''

	final_frame_permutation = []
	for scene in permutated_frames_dict:
		frames_permut_in_scene = permutated_frames_dict[scene]
		actual_frames = scenes_dict[scene]
		final_frame_permutation.extend([actual_frames[frame_per_id]\
						for frame_per_id in frames_permut_in_scene])

	return final_frame_permutation




