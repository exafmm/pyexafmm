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
        """
        Calculate the box bounds of a domain containing sources/targets.
        """
        xvec = np.vstack((self.targets[:, 0:1], self.sources[:, 0:1]))
        yvec = np.vstack((self.targets[:, 1:], self.sources[:, 1:]))

        left, right, bottom, top = xvec.min(), xvec.max(), yvec.min(), yvec.max()

        return left, right, bottom, top

    @staticmethod
    def partition(bounds):
        """
        Partition into four quadrants from bounds
        """
        left, right, bottom, top = bounds
        
        center = (left+right)/2, (top+bottom)/2

        northWest = (left, center[0], center[1], top)
        northEast = (center[0], right, center[1], top)
        southWest = (left, center[0], bottom, center[1])
        southEast = (center[0], right, bottom, center[1])

        return [northWest, northEast, southWest, southEast]



