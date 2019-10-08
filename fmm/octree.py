"""Implementation of an octree in Python."""

from . import hilbert as _hilbert
from . import utils as _utils

import numpy as _np
import numba as _numba


class Octree(object):
    """Data structure for handling Octrees."""

    def __init__(self, sources, targets, maximum_level, center, radius):
        """
        Initialize an Octree.

        Parameters
        ----------
        maximum_level : integer
            The maximum level of the Octree.
        points : np.ndarray
            An (N, 3) float64 array of N vertices
        """

        self._center = center
        self._radius = radius
        self._maximum_level = maximum_level
        self._sources = sources
        self._targets = targets

        self._source_leaf_nodes, self._sources_by_leafs, self._source_index_ptr = self._assign_points_to_leaf_nodes(
            self._sources
        )

        self._target_leaf_nodes, self._targets_by_leafs, self._target_index_ptr = self._assign_points_to_leaf_nodes(
            self._targets
        )
        self._non_empty_source_nodes = self._compute_set_of_all_non_empty_nodes(
            self._source_leaf_nodes
        )
        self._non_empty_target_nodes = self._compute_set_of_all_non_empty_nodes(
            self._target_leaf_nodes
        )

        self._target_neighbors = _numba_compute_neighbors(
            self._non_empty_target_nodes, self._non_empty_source_nodes
        )

    @property
    def radius(self):
        """Return radius of the Octree."""
        return self._radius

    @property
    def center(self):
        """Return center of the Octree."""
        return self._center

    @property
    def maximum_level(self):
        """Return the maximum level."""
        return self._maximum_level

    @property
    def sources(self):
        """Return the sources."""
        return self._sources

    @property
    def targets(self):
        """Return the targets."""
        return self._targets

    @property
    def target_neighbors(self):
        """Return the neighbors of the targets that contain source points."""
        return self._target_neighbors

    @property
    def source_leaf_nodes(self):
        """Return source leaf nodes."""
        return self._source_leaf_nodes

    @property
    def target_leaf_nodes(self):
        """Return target leaf nodes."""
        return self._target_leaf_nodes

    @property
    def sources_by_leafs(self):
        """Return sources by leafs."""
        return self._sources_by_leafs

    @property
    def targets_by_leafs(self):
        """Return targets by leafs."""
        return self._targets_by_leafs

    @property
    def source_index_ptr(self):
        """Return index ptr for sources by leafs."""
        return self._source_index_ptr

    @property
    def target_index_ptr(self):
        """Return index ptr for sources by leafs."""
        return self._target_index_ptr

    @property
    def target_neighbors(self):
        """Return array of target node neighbors."""
        return self._target_neighbors

    @property
    def non_empty_source_nodes(self):
        """Return non-empty source nodes."""
        return self._non_empty_source_nodes

    @property
    def non_empty_target_nodes(self):
        """Return non-empty target nodes."""
        return self._non_empty_target_nodes

    def parent(self, node_index):
        """Return the parent index of a node."""
        return _hilbert.get_parent(node_index)

    def children(self, node_index):
        """Return an iterator over the child indices."""

        first = node_index << 3
        last = 7 + (node_index << 3)
        return list(range(first, 1 + last))

    def nodes_per_side(self, level):
        """Return number of nodes along each dimension."""
        return 1 << level

    def nodes_per_level(self, level):
        """Return the number of nodes in a given level."""
        return 1 << 3 * level

    def _assign_points_to_leaf_nodes(self, points):
        """Assign points to leaf nodes."""

        assigned_nodes = _numba_assign_points_to_nodes(
            points, self.maximum_level, self.center, self.radius
        )

        point_indices_by_node = _np.argsort(assigned_nodes)

        index_ptr = []
        nodes = []

        count = 0
        previous_node = -1
        for node in assigned_nodes[point_indices_by_node]:
            if node != previous_node:
                index_ptr.append(count)
                nodes.append(node)
                previous_node = node
            count += 1
        index_ptr.append(count)

        return nodes, point_indices_by_node, index_ptr

    def _compute_set_of_all_non_empty_nodes(self, leaf_nodes):
        """Compute all non-empty nodes across the tree."""

        all_nodes = set(leaf_nodes)

        for node in leaf_nodes:
            parent = node
            while parent != 0:
                parent = self.parent(parent)
                all_nodes.add(parent)
        all_nodes.add(0)

        return _np.array(list(all_nodes), dtype=_np.int64)


@_numba.njit(cache=True)
def _numba_assign_points_to_nodes(points, level, x0, r0):
    """Assign points to leaf nodes."""

    npoints = len(points)
    assigned_nodes = _np.empty(npoints, dtype=_np.int64)
    for index in range(npoints):
        assigned_nodes[index] = _hilbert.get_key_from_point(
            points[index], level, x0, r0
        )
    return assigned_nodes


@_numba.njit(cache=True)
def _in_range(n1, n2, n3, bound):
    """Check if 0 <= n1, n2, n3 < bound."""
    return n1 >= 0 and n1 < bound and n2 >= 0 and n2 < bound and n3 >= 0 and n3 < bound


@_numba.njit(cache=True)
def _numba_compute_neighbors(nodes, comparison_nodes):
    """
    Compute all non-empty neighbors for the given nodes.

    The emptyness is determined through comparison with
    the second set comp_set. In this way target neighbors
    can be computed that contain source points.

    """

    comp_set = set(comparison_nodes)
    nnodes = len(nodes)

    neighbors = _np.empty((nnodes, 27), dtype=_np.int64)

    offset = _np.zeros(4, dtype=_np.int64)
    for index, node in enumerate(nodes):
        vec = _hilbert.get_4d_index_from_key(node)
        nodes_per_side = 1 << vec[3]
        count = -1
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    count += 1
                    offset[:3] = i, j, k
                    neighbor_vec = vec + offset
                    if _in_range(
                        neighbor_vec[0],
                        neighbor_vec[1],
                        neighbor_vec[2],
                        nodes_per_side,
                    ):
                        neighbor_key = _hilbert.get_key(neighbor_vec)
                        if neighbor_key in comp_set:
                            neighbors[index, count] = neighbor_key
                        else:
                            neighbors[index, count] = -1
                    else:
                        neighbors[index, count] = -1
    return neighbors

