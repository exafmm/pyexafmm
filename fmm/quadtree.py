"""
Quadtree datastructure, for 2D algorithms
"""
import numpy as np


class Node:
    """
    Defined over R^2, Consider only square domains for now
    """

    def __init__(self, sources, targets, parent):
        """
        Parameters
        ----------
        sources : np.array(shape=(N, 2))
            Coordinates of source points
        targets : np.array(shape=(N, 2))
            Coordinates of target points
        """
        self.sources = sources
        self.targets = targets
        self.parent = parent

    @property
    def bounds(self):
        """Find the bounds from the sources and targets in this Node"""
        return find_bounds(self.targets, self.sources)

    @property
    def children(self):
        _children = []
        quadrants = partition(self.bounds)
        for quadrant in quadrants:
            left, right, bottom, top = quadrant
            child_sources = []
            child_targets = []
            
            for source in self.sources:
                x, y = source
                if left <= x < right and bottom<= y < top:
                    child_sources.append(source)

            for target in self.targets:
                x, y = target
                if left <= x < right and bottom <= y < top:
                    child_targets.append(target)

            child_node = Node(child_sources, child_targets, self)

            _children.append(child_node)

        return _children


def find_bounds(targets, sources):
    """
    Calculate the box bounds of a domain containing sources/targets.
    """
    xvec = np.vstack((targets[:, 0:1], sources[:, 0:1]))
    yvec = np.vstack((targets[:, 1:], sources[:, 1:]))

    left, right, bottom, top = xvec.min(), xvec.max(), yvec.min(), yvec.max()

    return left, right, bottom, top

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


