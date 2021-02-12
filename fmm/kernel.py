"""
CPU based kernels for, accelerated with Numba.
"""
import numba
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


KERNELS = {
    'laplace': {
        'eval': laplace,
        'scale': laplace_scale
    },
}
