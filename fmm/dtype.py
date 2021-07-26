"""
Supported datatypes.
"""
import numba
import numpy as np


NUMPY = {
    'single': np.float32,
    'double': np.float64
}

NUMBA = {
    'single': numba.types.float32,
    'double': numba.types.float64
}
