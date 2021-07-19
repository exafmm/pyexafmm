"""
Supported datatypes, and mapping between Numpy and Numba types
"""
import numpy as np

FLOATING_POINT = {
    'f': np.float32,
    'd': np.float64
}

INTEGER = {
    'i': np.int32,
    'l': np.int64
}

DTYPE = {
    'single': {
        'float': FLOATING_POINT['f'],
        'int': INTEGER['i']
    },
    'double': {
        'float': FLOATING_POINT['d'],
        'int': INTEGER['l']
    }
}
