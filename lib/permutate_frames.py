#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

import numpy as np

from tsp_solver.greedy import solve_tsp
from parameters import cfg


def evaluate_distance(distance_matrix, permutation):
    distance = 0
    nbNodes = len(permutation)
    for i in range(nbNodes-1):
        distance += distance_matrix[permutation[i], permutation[i+1]]
    # distance += distance_matrix[permutation[nbNodes-1], permutation[0]]
    return distance


def two_opt_refine(distance_matrix, permutation):
    ref_perm = permutation
    nbNodes = len(ref_perm)

    bestChange = 1
    while bestChange > 0:
        best_i = 0
        best_j = 0
        bestChange = 0
        for i in range(1, nbNodes-2):
            for j in range(i+1, nbNodes-1):
                distIni1 = distance_matrix[ref_perm[i-1], ref_perm[i]]
                distIni2 = distance_matrix[ref_perm[j], ref_perm[j+1]]
                swapped1 = distance_matrix[ref_perm[i], ref_perm[j+1]]
                swapped2 = distance_matrix[ref_perm[i-1], ref_perm[j]]

                change = distIni1 + distIni2 - (swapped1 + swapped2)
                if change > bestChange:
                    bestChange = change
                    best_i = i
                    best_j = j

        if bestChange > 0:
            # Swap i and j
            temp = ref_perm[best_i]
            ref_perm[best_i] = ref_perm[best_j]
            ref_perm[best_j] = temp
            ref_perm[best_i+1:best_j] = ref_perm[best_i+1:best_j][::-1]

    return ref_perm


def permutateFramesWithinScenes(distances_dict):

    '''
    Finds permutation of frames within a scene
    that minimizes the total optical flow distance
    Solver used is greedy traveling salesman
    or 2-opt heuristic
    '''
    solution = {}
    for scene in distances_dict:

        nbNodes = len(distances_dict[scene])

        if cfg.USE_DUMMY_NODE:
            scratch = range(nbNodes+1)
            distance_matrix = np.zeros((nbNodes+1, nbNodes+1))
            distance_matrix[:nbNodes, :nbNodes] = distances_dict[scene]
        else:
            scratch = range(nbNodes)
            distance_matrix = distances_dict[scene]

        if cfg.TSP_SOLVER == 'greedy':
            solution[scene] = solve_tsp(distance_matrix)
        elif cfg.TSP_SOLVER == '2opt':
            solution[scene] = two_opt_refine(distance_matrix, scratch)

        if cfg.USE_DUMMY_NODE:
            solution[scene].remove(nbNodes)

    for scene in distances_dict:
        print "TSP distances for scene", scene, ":",\
            evaluate_distance(distances_dict[scene], solution[scene])

    # Further refine:
    if cfg.TSP_SOLVER == 'greedy' and cfg.REFINE_GREEDY_WITH_2OPT:
        solution[scene] = two_opt_refine(distance_matrix, solution[scene])
        for scene in distances_dict:
            print "TSP distances for scene", scene, ":",\
                evaluate_distance(distances_dict[scene], solution[scene])

    return solution
