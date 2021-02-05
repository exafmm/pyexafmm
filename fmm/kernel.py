"""
Kernel class
"""
import abc

import numpy as np


class SingletonDecorator:
    """
    Decorate instance to enforce Singleton
    """
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance == None:
            self.instance = self.klass(*args, **kwargs)
        return self.instance


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
    def kernel_function(x, y):
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
    def kernel_function(x, y):
        return np.dot(x, y)

    def __call__(self, x, y):
        return self.kernel_function(x, y)


class Laplace(Kernel):
    """Single layer Laplace kernel
    """

    @staticmethod
    def scale(level):
        return 1/(2**level)

    @staticmethod
    def kernel_function(x, y):
        r = np.linalg.norm(x-y)

        if np.isclose(r, 0, rtol=1e-12):
            return 0

        return 1/(4*np.pi*r)

    def __call__(self, x, y):
        return self.kernel_function(x, y)


KERNELS = {
    'laplace': Laplace,
    'identity': Identity
}
