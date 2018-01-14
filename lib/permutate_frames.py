#!/usr/bin/env python
# -*- coding: utf-8 -*-
# January 2017
# Written by Victor SUO
# Email: xunvictor.suo@gmail.com

from tsp_solver.greedy import solve_tsp

def permutateFramesWithinScenes(distances_dict):

    '''
    Finds permutation of frames within a scene
    that minimizes the total optical flow distance
    Solver used is greedy traveling salesman
    '''

    solution = {}
    for scene in distances_dict:
        solution[scene] = solve_tsp(distances_dict[scene])

    return solution