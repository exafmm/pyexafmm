"""
Quadtree datastructure, for 2D algorithms
"""
import numpy as np


class Quadtree:
    """
    Defined over R^2, Consider only square domains for now
    """

    def __init__(self, sources, targets, max_level):
        """
        Parameters
        ----------
        sources : np.array(shape=(N, 2))
            Coordinates of source points
        targets : np.array(shape=(N, 2))
            Coordinates of target points
        max_level : int
            Max height of tree to build
        """
        self.sources = sources
        self.targets = targets
        self.max_level = max_level

    @property
    def bounds(self):
        xs = self.targets[:, 0:1] + self.sources[:, 0:1]
        ys = self.targets[:, 1:] + self.sources[:, 1:]

        l, r, b, t = xs.min(), xs.max(), ys.min(), ys.max()

        return l, r, b, t

    def partition(level):
        pass

    def _generate_tree(self):
        pass