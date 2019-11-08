"""
Quadtree datastructure, for 2D algorithms
"""
import numpy as np


class Node:
    """
    Defined over R^2, Consider only square domains for now
        [0, 1] x [0, 1] for simplicity
    """

    def __init__(self, sources, targets, parent=None, bounds=(-0.1, 1.1, -0.1, 1.1)):
        """
        Parameters
        ----------
        sources : np.array(shape=(N, 2))
            Coordinates of source points
        targets : np.array(shape=(N, 2))
            Coordinates of target points
        parent: pointer/None
            Pointer to parent node, None it is the top of tree.
        bounds: List()/ default [-0.1, 1.1, -0.1, 1.1]
            Default domain
        """
        self.sources = sources
        self.targets = targets
        self.parent = parent
        self.bounds = bounds

    @property
    def children(self):
        _children = []
        quadrants = partition(self.bounds)


        for quadrant in quadrants:
            left, right, bottom, top = quadrant

            child_sources = []
            child_targets = []

            for source in self.sources:

                if left <= source[0] < right and bottom <= source[1] < top:
                    child_sources.append(source)

            for target in self.targets:
                
                if left <= target[0] < right and bottom <= target[1] < top:
                    child_targets.append(target)

            n_sources = len(child_sources)
            n_targets = len(child_targets)
            child_sources = np.array(child_sources).reshape((n_sources, 2))
            child_targets = np.array(child_targets).reshape((n_targets, 2))

            child_node = Node(child_sources, child_targets, self, quadrant)

            _children.append(child_node)

        return _children


class Quadtree:

    def __init__(self, max_levels):
        self.max_levels = max_levels

    def generate(self):
        pass


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


def find_vertices(partition):

    left, right, bottom, top = partition
    x_coords = [left, right]
    y_coords = [bottom, top]

    return np.array([(x, y) for x in x_coords for y in y_coords])

    
