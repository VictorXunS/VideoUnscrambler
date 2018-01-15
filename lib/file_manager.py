#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import numpy as np
import os
import time

from parameters import cfg

def archivePreviousRun(video_filename):

    '''
    Save previous computation files in a separate folder so that they are not reused
    '''

    archive_name = time.strftime("%d_%m_%Y;%H_%M_%S")

    folder_name = os.path.join(cfg.DATA_FOLDER, os.path.splitext(video_filename)[0])
    data_filepath = os.path.join(folder_name, archive_name)

    if not os.path.exists(data_filepath):
        os.makedirs(data_filepath)

    aHash_filename = 'aHash.npy'
    pHash_filename = 'pHash.npy'
    dHash_filename = 'dHash.npy'
    wHash_filename = 'wHash.npy'
    corners_filename = 'cornersGFTT.npy'
    of_matrices_filename = 'LK_optical_flow_distances.npy'

    files_to_archive = [aHash_filename, pHash_filename, dHash_filename, wHash_filename, \
                        corners_filename, of_matrices_filename]

    for file in files_to_archive:
        if os.path.exists(os.path.join(folder_name, file)):
            os.rename(os.path.join(folder_name, file), os.path.join(data_filepath, file))

    print "Previous data files archived, ready for new run"