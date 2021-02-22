"""
CPU based kernels for, accelerated with Numba.
"""
import math

import numba
from numba import cuda
import numpy as np

TOL = 1e-6

def laplace_scale(level):
    return 1/(2**level)


@numba.njit(cache=True)
def laplace(x, y):
    diff = x-y
    diff2 = diff*diff

    if np.all(diff2 < TOL):
        return np.float64(0)

    diff2 = np.sqrt(np.sum(diff2))

    return np.reciprocal(4*np.pi*diff2)


@cuda.jit(device=True)
def laplace_cuda(ax, ay, az, bx, by, bz):
    """
    Numba-Cuda Laplace device kernel.

    Parameters:
    -----------
    ax : np.float32
        'x' coordinate of a point.
    ay : np.float32
        'y' coordinate of a point.
    az : np.float32
        'z' coordinate of a point.
    bx : np.float32
        'x' coordinate of a point.
    by : np.float32
        'y' coordinate of a point.
    bz : np.float32
        'z' coordinate of a point.
    """
    rx = ax-bx
    ry = ay-by
    rz = az-bz

    dist = rx**2+ry**2+rz**2
    dist_sqr = math.sqrt(dist)
    inv_dist_sqr =  1./(4*math.pi*dist_sqr)

    if math.isinf(inv_dist_sqr):
        return 0.

    return inv_dist_sqr


KERNELS = {
    'laplace': {
        'eval': laplace,
        'scale': laplace_scale
    },
}
