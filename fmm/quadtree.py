"""
Simple Quadtree structure based on discontinuous Hilbert-like Curve.

References:
-----------
[1] J. Skilling (2004) 10.1063/1.1751381
"""
import numpy as np


def point_to_hilbert(y, x, p):
    """
    Transform 2D coordinates to distance along Hilbert-like curve.
    This method works by examining the bits of y, x from the highest to lowest
    order. Determining which quadrant (in the Quadtree upon which this curve is
    being drawn) in which the point lands, and assigning this to a binary
    Hilbert-like value.

    Parameters:
    -----------
    y : float
    x : float
    p : int
        Precision of data, here x,y \in [0, 2*p-1]. Quadtree has a maximum of
        (2*p)**2 leaf nodes.

    Returns:
    --------
    key : int
        Distance along discontinuous Hilbert-like curve.
    """
    max_value = 2*p-1
    if (x < 0 or x > max_value) or (y < 0 or y > max_value):
        raise ValueError('Must pick in valid range')

    key = 0
    for i in range(p-1, -1, -1):
        mask = 2**i
        # Start with highest order bits
        quad_x = 1 if mask & x else 0
        quad_y = 1 if mask & y else 0

        # Concatenate
        key <<= 1
        key |= quad_y
        key <<= 1
        key |= quad_x

    return key


def hilbert_to_point(key, p):
    """
    Transform distance along Hilbert-like curve back to 2D coordinates.

    Parameters:
    -----------
    key : int
        Distance along discontinuous Hilbert-like curve, indicates a nodal
        position.
    p : int
        Precision of data, here x,y \in [0, 2*p-1]. Quadtree has a maximum of
        (2*p)**2 leaf nodes.

    Returns:
    --------
    (int, int)
        2D spatial coordinates of Hilbert-like key.
    """
    max_value = (2*p)**2

    if key < 0 or key > max_value:
        raise ValueError('Must pick in range')

    y, x = 0, 0

    for i in range(p-1, -1, -1):

        y_mask = 1 << ((2*i) + 1)
        x_mask = y_mask >> 1

        xi = (x_mask & key) >> (2*i)
        yi = (y_mask & key) >> ((2*i)+1)

        y <<= 1
        y |= yi
        x <<= 1
        x |= xi

    return y, x


def find_parent(key):
    """
    Find the parent quadrant of a Hilbert-like key.

    Parameters:
    -----------
    key : int
        Distance along discontinuous Hilbert-like curve, indicates a nodal
        position.

    Returns:
    --------
    int
        Parent quadrant Hilbert-like key.
    """
    return key >> 2


def find_children(key):
    """
    Find the children of a given quadrant.

    Parameters:
    -----------
    key : int
        Distance along discontinuous Hilbert-like curve, indicates a nodal
        position.

    Returns:
    --------
    [int]
        Child keys of this node.
    """
    offset = (key << 2)

    children = []
    for i in range(4):
        children.append(offset | i)

    return children


class Quadtree:
    """
    Simple Quadtree class.
    """
    def __init__(self, sources, targets, precision=None):
        """
        Parameters:
        -----------
        sources : Points
            A Points class of sources.
        targets : Points
            A points class of targets.
        precision : None/int.
            Desired precision of quadtree, defaults to that defined by data if
            not provided.
        """
        self.sources = sources
        self.targets = targets

        # Estimate precision required of tree from data
        if precision is None:
            self.precision = (self.sources.int_max // 2) + 1
        else:
            self.precision = precision

        # Assign sources to leaf nodes
        self._assign_points_to_leaf_nodes(self.sources)

    @property
    def leaf_nodes(self):
        """
        Leaf nodes available for this precision.
        """
        return np.array(
            [
                hilbert_to_point(i, self.precision)
                for i in range((2*self.precision)**2)
            ]
        )

    @property
    def leaf_node_keys(self):
        """
        Distances along Hilbert-like curve corresponding to each leaf-node.
        """
        return np.array(
            [
                point_to_hilbert(*node, self.precision)
                for node in self.leaf_nodes
            ]
        )

    def _assign_points_to_leaf_nodes(self, points):
        """
        Sift through all points and all possible leaf nodes, and figure out
        the assignment of points to leaf nodes.

        Parameters:
        -----------
        points : Points
        """
        # Width of leaf node
        delta = 1

        # For each point, check each leaf node to see where it goes
        for idx, point in enumerate(points):
            for jdx, node in enumerate(self.leaf_nodes):
                ny, nx = node[0], node[1]
                y, x = point[0], point[1]

                # Assign leaf nodes to each source
                if ny <= y < ny+delta and nx <= x < nx+delta:
                    points[idx: idx+1, 2] = self.leaf_node_keys[jdx]


class Points:
    """
    Helper class to enforce safety when dealing with numpy arrays containing
    coordinate data.
    """
    def __init__(self, points):
        """
        Parameters:
        -----------
        points : np.ndarray(shape=(N, 3))
            N is the number of points, the first two dimensions must correspond
            to x and y coordinate resp, and the third dimension indicates the
            leaf node this point has been assigned to.
        """
        if not isinstance(points, np.ndarray):
            raise TypeError('`points` must be a numpy array')

        if points.ndim != 2 or points.shape[1] != 3:
            raise TypeError('`points` must by an (N,3) array')

        self.points = points
        self.ys = self.points[:, 0]
        self.xs = self.points[:, 1]

    def __iter__(self):
        return iter(self.points)

    def __setitem__(self, k, v):
        self.points[k] = v

    def __getitem(self, k):
        return self.points[k]

    @property
    def leaf_nodes(self):
        """Leaf nodes as coordinates np.array(shape=(N, 2))"""
        return self.points[:, 2]

    @property
    def x_max(self):
        """Max x coordinate"""
        return self.xs.max()

    @property
    def y_max(self):
        """Max y coordinate"""
        return self.ys.max()

    @property
    def int_max(self):
        """Max coordinate along any axis as the closest integer"""
        return int(max(self.y_max, self.x_max))
