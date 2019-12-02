"""
Simple Quadtree structure based on Z-order curve.
"""
import numpy as np


def point_to_curve(y, x, p):
    """
    Transform 2D coordinates to distance along Z-order curve. This method works
    by examining the bits of y, x from the highest to lowest order. Determining
    which quadrant (in the Quadtree upon which this curve is being drawn) in
    which the point lands, and assigning this to a binary Z-order value.

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
        Distance along discontinuous Z-order curve.
    """
    max_value = 2*p-1
    if (x < 0 or x > max_value) or (y < 0 or y > max_value):
        raise ValueError(
            f'Must pick in valid range {0} <= x, y <= {max_value}: {x, y}'
        )

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


def curve_to_point(key, p):
    """
    Transform distance along Z-order curve back to 2D coordinates.

    Parameters:
    -----------
    key : int
        Distance along discontinuous Z-order curve, indicates a nodal
        position.
    p : int
        Precision of data, here x,y \in [0, 2*p-1]. Quadtree has a maximum of
        (2*p)**2 leaf nodes.

    Returns:
    --------
    coordinate : (int, int)
        2D spatial coordinates of Z-order key.
    """
    max_value = (2*p)**2

    if key < 0 or key > max_value:
        raise ValueError(
            f'Must pick in range {0} < value < {max_value}'
        )

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
    Find the parent quadrant of a Z-order key.

    Parameters:
    -----------
    key : int
        Distance along discontinuous Z-order curve, indicates a nodal
        position.

    Returns:
    --------
    key : int
        Parent quadrant Z-order key.
    """
    return key >> 2


def find_children(key):
    """
    Find the children of a given quadrant.

    Parameters:
    -----------
    key : int
        Distance along discontinuous Z-order curve, indicates a nodal
        position.

    Returns:
    --------
    keys : [int]
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

        # Assign sources/targets to leaf nodes
        self._assign_points_to_leaf_nodes(self.sources)
        self._assign_points_to_leaf_nodes(self.targets)

    @property
    def n_levels(self):
        """Maximum number of tree levels."""
        return int(np.log(2*self.precision)/np.log(2))

    @property
    def n_nodes(self):
        """Number of nodes in the tree in total."""
        return sum([4**level for level in range(self.n_levels+1)])

    @property
    def leaf_nodes(self):
        """Leaf nodes."""
        return np.array(
            [
                curve_to_point(i, self.precision)
                for i in range((2*self.precision)**2)
            ]
        )

    @property
    def leaf_node_keys(self):
        """Distances along Z-order curve corresponding to each leaf-node."""
        return np.array(
            [
                point_to_curve(*node, self.precision)
                for node in self.leaf_nodes
            ]
        )

    def _assign_points_to_leaf_nodes(self, points):
        """
        Sift through all points and calculate which leaf node they lie in,
        assign to points in place.

        Parameters:
        -----------
        points : Points

        Returns:
        --------
        None
        """
        for idx, point in enumerate(points):
            y, x = point[0], point[1]
            trunc_y, trunc_x = int(np.floor(y)), int(np.floor(x))
            key = point_to_curve(trunc_y, trunc_x, self.precision)
            points[idx: idx+1, 2] = key


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

    def __repr__(self):
        return str(self.points)

    def __setitem__(self, k, v):
        self.points[k] = v

    def __getitem__(self, k):
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
