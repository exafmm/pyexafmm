"""
3D Cartesian rotation matrices
"""
import numpy as np


X_ROT_90 = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])

X_REF = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

Y_ROT_90 = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
])

Y_REF = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])

Z_ROT_90 = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

Z_REF = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])
