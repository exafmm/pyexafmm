"""
Base class to store results
"""
import abc

import numpy as np


class AbstractDensity(abc.ABC):
    """Base Return Object for calculations"""
    def __init__(self, surface, density):
        """
        Parameters:
        -----------
        surface : np.array(shape=(n, 3))
            `n` quadrature points discretising surface.
        density : np.array(shape=(n))
            `n` densities, corresponding to each quadrature point.
        """
        if isinstance(surface, np.ndarray) and isinstance(density, np.ndarray):
            self.surface = surface
            self.density = density
        else:
            raise TypeError("`surface` and `density` must be numpy arrays")

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError


class Charge(AbstractDensity):
    """
    Return object bundling computed charge, at corresponding points.
    """
    def __init__(self, surface, density):
        super().__init__(surface, density)

    def __repr__(self):
        return str((self.surface, self.density))


class Potential(AbstractDensity):
    """
    Return object bundling computed potential, at corresponding points.
    """
    def __init__(self, surface, density, indices=None):
        super().__init__(surface, density)

        if indices is None:
            self.indices = set()
        else:
            self.indices = indices

    def __repr__(self):
        return str((self.surface, self.density))