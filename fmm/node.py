"""
Abstraction for each tree node
"""
import numpy as np


class Node:
    """Holds expansion and source/target indices for each tree node"""
    def __init__(self, key, ncoefficients, indices=None):
        """
        Parameters:
        ----------
        key : int
            Hilbert key for a node.
        ncoefficients: int
            Number of expansion coefficients, corresponds to discrete points on
            surface of box for this node.
        indices : set
            Set of indices.
        """
        self.key = key
        self.expansion = np.zeros(ncoefficients, dtype='float64')

        if indices is None:
            self.indices = set()
        else:
            self.indices = indices

    def __repr__(self):
        return str((self.key, self.expansion, self.indices))