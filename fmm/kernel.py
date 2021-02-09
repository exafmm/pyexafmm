"""
Kernel class
"""
import abc

import numba
import numpy as np


class Kernel(abc.ABC):
    """
    Abstract callable Kernel Class

    Parameters:
    -----------
    x : np.array(shape=(n))
        An n-dimensional vector corresponding to a point in n-dimensional space.
    y : np.array(shape=(n))
        Different n-dimensional vector corresponding to a point in n-dimensional
        space.

    Returns:
    --------
    float
        Operator value (scaled by 4pi) between points x and y.
    """

    @abc.abstractstaticmethod
    def scale(level):
        """Implement level dependent scale.
        """
        raise NotImplementedError

    @abc.abstractstaticmethod
    def eval(x, y):
        """ Implement static kernel function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x, y):
        raise NotImplementedError


class Identity(Kernel):
    """Identity operation
    """

    @staticmethod
    def scale(level):
        return level

    @staticmethod
    def eval(x, y):
        return np.dot(x, y)

    def __call__(self, x, y):
        return self.eval(x, y)


class Laplace(Kernel):
    """Single layer Laplace kernel
    """

    @staticmethod
    def scale(level):
        return 1/(2**level)

    @staticmethod
    @numba.njit(cache=True)
    def eval(x, y):
        diff = x-y
        diff2 = diff*diff
        tol = np.float64(1e-6)

        if np.all(diff2 < tol):
            return np.float64(0)
        
        diff2 = np.sqrt(np.sum(diff2))
        
        return np.reciprocal(4*np.pi*diff2)

    def __call__(self, x, y):
        return self.eval(x, y)


KERNELS = {
    'laplace': Laplace,
    'identity': Identity
}
