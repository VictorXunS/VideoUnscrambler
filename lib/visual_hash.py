#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import numpy as np
import os
from PIL import Image
import imagehash as imgh
import scipy
import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

from parameters import cfg

def getVisualHashes(video_filename, frame_list):

    ''' 
    Compute perceptual hashes for each frame using 4 different methods
    Can load hashes if already precomputed
    '''

    hash_folder = cfg.DATA_FOLDER + os.path.splitext(video_filename)[0]
    # hash_folder = cfg.HASHES_FOLDER + os.path.splitext(video_filename)[0]

    # Try to load the files if they are saved
    if os.path.exists(hash_folder + '/aHash.npy'):
        aHash_list = np.load(hash_folder + '/aHash.npy')
        print 'aHash loaded from file'
    else: # Or compute them
        aHash_list = [imgh.average_hash(Image.fromarray(frame)) for frame in frame_list]
        np.save(hash_folder + '/aHash', aHash_list)
        print "aHash computed"
    
    if os.path.exists(hash_folder + '/pHash.npy'):
        pHash_list = np.load(hash_folder + '/pHash.npy')
        print 'pHash loaded from file'
    else:
        pHash_list = [imgh.phash(Image.fromarray(frame)) for frame in frame_list]
        np.save(hash_folder + '/pHash', pHash_list)
        print "pHash computed"
    
    if os.path.exists(hash_folder + '/dHash.npy'):
        dHash_list = np.load(hash_folder + '/dHash.npy')
        print 'dHash loaded from file'
    else:
        dHash_list = [imgh.dhash(Image.fromarray(frame)) for frame in frame_list]
        np.save(hash_folder + '/dHash', dHash_list)
        print "dHash computed"
    
    if os.path.exists(hash_folder + '/wHash.npy'):
        wHash_list = np.load(hash_folder + '/wHash.npy')
        print 'wHash loaded from file'
    else:
        wHash_list = [imgh.whash(Image.fromarray(frame)) for frame in frame_list]
        np.save(hash_folder + '/wHash', wHash_list)
        print "wHash computed"

    return aHash_list, pHash_list, dHash_list, wHash_list

def clusterScenesUsingHash(list_hashes, nbFrames):

    '''
    Clusters scenes based on visual hash distance matrix
    '''

    # Constitute distance matrix
    hash_distances = np.zeros((nbFrames, nbFrames))
    for i in range(nbFrames):
        for ii in range(nbFrames):
            hash_distances[i,ii] = list_hashes[i] - list_hashes[ii]

    # Convert the redundant n*n square matrix form into a condensed nC2 array
    distArray = ssd.squareform(hash_distances)

    # Hierarchical clustering
    z = hac.linkage(distArray, method='single', metric='euclidean')
    knee = np.diff(z[::-1, 2], 2)
    num_clust = knee.argmax() + 2
    part = hac.fcluster(z, num_clust, 'maxclust')

    # fig = plt.figure(figsize=(15, 9))
    # dn = dendrogram(z)
    # plt.show(block=True)

    # Store scene labels in a more handy dictionary
    scenes_dict = convertClusterToDict(part, nbFrames)

    # Remove scenes that have too few number of frames
    if cfg.REMOVE_BAD_SCENES:
        scenes_dict, nbScenes = removeBadScenes(scenes_dict)

    return scenes_dict, nbScenes

def convertClusterToDict(clusters, nbFrames):
    
    '''
    Convert scenes cluster array into a dictionary of frames
    '''

    clusters_dict = {}
    for frame_id in range(nbFrames):
        if clusters[frame_id] not in clusters_dict:
            clusters_dict[clusters[frame_id]] = []
        clusters_dict[clusters[frame_id]].append(frame_id)

    return clusters_dict

def removeBadScenes(scenes_dict):

    '''
    Remove scenes with less than a few frames
    '''
    
    new_scenes_dict = {}
    nbScenes = 0
    for scene in scenes_dict:
        if len(scenes_dict[scene]) >= cfg.MINIMAL_NB_FRAMES_PER_SCENE:
            new_scenes_dict[scene] = scenes_dict[scene]
            nbScenes += 1

    print "Number of scenes detected: ", nbScenes
    return new_scenes_dict, nbScenes