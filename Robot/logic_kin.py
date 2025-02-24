#!/usr/bin/env python

import numpy as np
from robot_kin import inverse

def vector_subtraction(A, B):
    """Subtracts vector B from A."""
    return np.subtract(A, B)

def vector_norm(V):
    """Computes the Euclidean norm of a 3D vector."""
    return np.linalg.norm(V)

def get_transformation(pose):
    """
    Convert pose (XYZ + Euler angles) into a 4x4 transformation matrix.
    :param pose: List of 6 values (x, y, z, gamma, beta, alpha)
    :return: 4x4 transformation matrix
    """
    x, y, z, gamma, beta, alpha = pose

    T = np.array([
        [np.cos(alpha) * np.cos(beta), np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
         np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma), x],
        [np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
         np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma), y],
        [-np.sin(beta), np.cos(beta) * np.sin(gamma), np.cos(beta) * np.cos(gamma), z],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return T

def move_base_arm(chaser_position, base_position, arm_pose):
    """Determines whether the base should move based on arm reachability."""
    base_move = False
    min_base_arm_position = np.array([0.3, 0.2, 0.0])
    arm_base_position = vector_subtraction(chaser_position[:3], base_position)

    if vector_norm(arm_base_position) > 1.0:
        base_position[:] = vector_subtraction(chaser_position[:3], min_base_arm_position)
        base_move = True

    arm_pose[:] = chaser_position[:]
    return base_move

def base_arm_variables(base_position, arm_pose):
    """Computes joint variables for the base and arm."""
    base_variables = np.array([base_position[0], base_position[1], 0.0])
    T = get_transformation(arm_pose)
    solutions = inverse(T)

    if solutions is not None and len(solutions) > 0:
        for sol in solutions:
            if sol[1] < 0:
                arm_variables = sol
                return base_variables, arm_variables
    return None, None