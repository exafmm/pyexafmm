"""Implementation of an octree in Python."""
import numba
import numpy as np

import fmm.hilbert as hilbert


class Octree:
    """Data structure for handling Octrees."""

    def __init__(self, sources, targets, maximum_level, source_densities):
        """
        Initialize an Octree.

        Parameters
        ----------
        maximum_level : integer
            The maximum level of the Octree.
        points : np.ndarray
            An (N, 3) float64 array of N vertices
        """

        # Maximum  level
        self._maximum_level = maximum_level

        # The actual sources
        self._sources = sources

        # The actual targets
        self._targets = targets

        # Densities at sources
        self._source_densities = source_densities

        # Center of the box and radius of the box
        max_bound, min_bound = compute_bounds(sources, targets)
        self._center = compute_center(max_bound, min_bound)
        self._radius = compute_radius(max_bound, min_bound, self.center)

        # Maps source node to index
        self._source_node_to_index = None

        # Maps target node to index
        self._target_node_to_index = None

        # Non empty source nodes
        self._non_empty_source_nodes = None

        # Non empty target nodes
        self._non_empty_target_nodes = None

        # Indices of source leaf nodes
        self._source_leaf_nodes = None

        # Associated sources
        self._sources_by_leafs = None

        # Index ptr
        self._source_index_ptr = None

        # Indices of target leaf nodes
        self._target_leaf_nodes = None

        # Targets by leafs
        self._targets_by_leafs = None

        # Targets index pointer
        self._target_index_ptr = None

        # Target neighbors
        self._target_neighbors = None

        # Interaction list
        self._interaction_list = None

        (
            self._source_leaf_nodes,
            self._sources_by_leafs,
            self._source_index_ptr,
        ) = self._assign_points_to_leaf_nodes(sources)

        self._source_leaf_key_to_index = {key: index for index, key in enumerate(self._source_leaf_nodes)}

        (
            self._target_leaf_nodes,
            self._targets_by_leafs,
            self._target_index_ptr,
        ) = self._assign_points_to_leaf_nodes(targets)

        self._target_leaf_key_to_index = {key: index for index, key in enumerate(self._target_leaf_nodes)}

        (
            self._non_empty_source_nodes,
            self._source_node_to_index
        ) = self._enumerate_non_empty_nodes(self._source_leaf_nodes)

        (
            self._non_empty_target_nodes,
            self._target_node_to_index
        ) = self._enumerate_non_empty_nodes(self._target_leaf_nodes)

        self._source_nodes_by_level = _sort_nodes_by_level(self._non_empty_source_nodes)
        self._target_nodes_by_level = _sort_nodes_by_level(self._non_empty_target_nodes)

        self._target_neighbors = _numba_compute_neighbors(
            self._non_empty_target_nodes, self._source_node_to_index
        )

        self._interaction_list = _numba_compute_interaction_list(
            self._non_empty_target_nodes,
            self._target_neighbors,
            self._source_node_to_index,
            self._target_node_to_index,
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
    def source_densities(self):
        return self._source_densities

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
    def source_leaf_key_to_index(self):
        """Return leaf index from key."""
        return self._source_leaf_key_to_index

    @property
    def target_leaf_key_to_index(self):
        """Return target index from key."""
        return self._target_leaf_key_to_index

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
    def non_empty_source_nodes(self):
        """Return non-empty source nodes."""
        return self._non_empty_source_nodes

    @property
    def non_empty_target_nodes(self):
        """Return non-empty target nodes."""
        return self._non_empty_target_nodes

    @property
    def non_empty_source_nodes_by_level(self):
        """Non-empty source nodes by level."""
        return self._source_nodes_by_level

    @property
    def non_empty_target_nodes_by_level(self):
        """Non-empty target nodes by level."""
        return self._target_nodes_by_level

    @property
    def interaction_list(self):
        """Return interaction list."""
        return self._interaction_list

    @property
    def source_node_to_index(self):
        """Return index of a given source node key."""
        return self._source_node_to_index

    @property
    def target_node_to_index(self):
        """Return index of a given target node key."""
        return self._target_node_to_index

    def nodes_per_side(self, level):
        """Return number of nodes along each dimension."""
        return 1 << level

    def nodes_per_level(self, level):
        """Return the number of nodes in a given level."""
        return 1 << 3 * level

    def _assign_points_to_leaf_nodes(self, points):
        """
        Assign points to leaf nodes.

        Parameters:
        -----------
        points : np.array(shape=(npoints, 3))
            Points in 3D Cartesian coordinates.

        Returns:
        --------
        nodes : list[int]
            Non-empty leaf node keys, sorted by numeric value.
        point_indices_by_node : list[int]
            Indices that link (unsorted) array of points to (sorted) list of
            nodes returned by this method.
        index_ptr : list[int]
            Pointers (indices) that link the (sorted) array of points to each
            (unique) non-empty node returned by `nodes`.
        """

        # Find Hilbert keys for each point
        assigned_nodes = hilbert.get_keys_from_points(
            points, self.maximum_level, self.center, self.radius
        )

        # Need to sort point indices as we use a sorted list of nodes to check
        # whether points belong to same node.
        point_indices_by_node = np.argsort(assigned_nodes)

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

    def _enumerate_non_empty_nodes(self, leaf_nodes):
        """
        Enumerate all non-empty nodes across the tree.

        Parameters:
        -----------
        leaf_nodes : list[int]
            List of leaf node keys.

        Returns:
        --------
        list_of_nodes: list

        node_map: np.array(shape=(n_nodes))

        """

        nleafs = len(leaf_nodes)

        node_map = -1*np.ones(
            hilbert.get_number_of_all_nodes(self.maximum_level), dtype=np.int64
        )

        # Enumerate non empty leaf nodes with a value from range(nleafs)
        node_map[leaf_nodes] = range(nleafs)

        count = nleafs
        for node in leaf_nodes:
            parent = node
            while parent != 0:
                parent = hilbert.get_parent(parent)
                if node_map[parent] == -1:
                    node_map[parent] = count
                    count += 1


        # Returns indices of node_map not equal to -1
        list_of_nodes = np.flatnonzero(node_map != -1)

        # node_map[list_of_nodes] = non-empty node values (count)
        # indices sorted by counts
        indices = np.argsort(node_map[list_of_nodes])

        # print("indices", indices)

        # print("list_of_nodes[indices]", list_of_nodes[indices], len(list_of_nodes))
        return list_of_nodes[indices], node_map


# def _numba_assign_points_to_nodes(points, level, x0, r0):
    # """Assign points to leaf nodes."""

    # npoints = len(points)
    # assigned_nodes = np.empty(npoints, dtype=np.int64)
    # for index in range(npoints):
        # assigned_nodes[index] = hilbert.get_key_from_point(
            # points[index], level, x0, r0
        # )
    # return assigned_nodes


@numba.njit(cache=True)
def _in_range(n1, n2, n3, bound):
    """Check if 0 <= n1, n2, n3 < bound."""
    return (0 <= n1 < bound) and (0 <= n2 < bound) and (0 <= n3 < bound)


@numba.njit(cache=True)
def _numba_compute_neighbors(target_nodes, source_node_map):
    """
    Compute all non-empty neighbors for the given nodes.

    The emptyness is determined through comparison with
    the second set comp_set. In this way target neighbors
    can be computed that contain source points.

    """

    nnodes = len(target_nodes)

    neighbors = np.empty((nnodes, 27), dtype=np.int64)

    offset = np.zeros(4, dtype=np.int64)
    for index, node in enumerate(target_nodes):
        vec = hilbert.get_4d_index_from_key(node)
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
                        neighbor_key = hilbert.get_key_from_4d_index(neighbor_vec)
                        if source_node_map[neighbor_key] != -1:
                            neighbors[index, count] = neighbor_key
                        else:
                            neighbors[index, count] = -1
                    else:
                        neighbors[index, count] = -1
    return neighbors


@numba.njit(cache=True)
def _numba_compute_interaction_list(
        target_nodes, target_source_neighbors, source_node_to_index, target_node_to_index
):
    """Compute the interaction list."""

    def find_neighbors(array, node_index, neighbor_child):
        """Find node in neighbors array."""
        for index in range(27):
            if array[node_index, index] == neighbor_child:
                return True
        return False

    nnodes = len(target_nodes)

    interaction_list = -np.ones((nnodes, 27, 8), dtype=np.int64)

    for node_index, node in enumerate(target_nodes):
        level = hilbert.get_level(node)
        if level < 2:
            continue
        parent = hilbert.get_parent(node)
        for neighbor_index, neighbor in enumerate(
                target_source_neighbors[target_node_to_index[parent]]
        ):
            if neighbor == -1:
                continue
            for child_index, neighbor_child in enumerate(
                    hilbert.get_children(neighbor)
            ):
                if source_node_to_index[neighbor_child] != -1 and not find_neighbors(
                        target_source_neighbors, node_index, neighbor_child
                ):
                    interaction_list[node_index, neighbor_index, child_index] = neighbor_child
    return interaction_list


def _sort_nodes_by_level(keys):
    """Return dict with nodes sorted by level."""
    sorted_keys, indexptr = _numba_sort_nodes_by_level(keys)

    levels_dict = {}
    number_of_levels = len(indexptr) - 1
    for level in range(number_of_levels):
        levels_dict[level] = sorted_keys[indexptr[level] : indexptr[1 + level]]

    return levels_dict


@numba.njit(cache=True)
def _numba_sort_nodes_by_level(keys):
    """
    Sort nodes by level implementation.
    """

    sorted_keys = np.sort(keys)

    next_level = 1
    next_offset = hilbert.get_level_offset(next_level)
    indexptr = [0]
    for index, key in enumerate(sorted_keys):
        if key >= next_offset:
            indexptr.append(index)
            next_level += 1
            next_offset = hilbert.get_level_offset(next_level)
    indexptr.append(len(keys))
    return sorted_keys, indexptr


def compute_bounds(sources, targets):
    """
    Compute bounds of computational domain of an Octree containing given
        sources and targets.

    Parameters:
    -----------
    sources : np.array(shape=(3, nsources), dtype=np.float64)
    targets : np.array(shape=(3, ntargets), dtype=np.float64)

    Returns:
    --------
    (np.array(shape=(3,), dtype=np.float64), np.array(shape=(3,), dtype=np.float64))
        Tuple containing the maximal/minimal coordinate in the sources/targets
        provided.
    """
    min_bound = np.min(
        np.vstack([np.min(sources, axis=0), np.min(targets, axis=0)]), axis=0
    )

    max_bound = np.max(
        np.vstack([np.max(sources, axis=0), np.max(targets, axis=0)]), axis=0
    )

    return max_bound, min_bound


def compute_center(max_bound, min_bound):
    """
    Compute center of Octree's root node.

    Parameters:
    -----------
    max_bound : np.array(shape=(3,) dtype=np.float64)
        Maximal point in Octree's root node.
    min_bound : np.array(shape=(3,) dtype=np.float64)
        Minimal point in Octree's root node.

    Returns:
    --------
    np.array(shape=(3,), dtype=np.float64)
        Cartesian coordinates of center.
    """

    center = (min_bound + max_bound) / 2

    return center


def compute_radius(center, max_bound, min_bound):
    """
    Compute half side length of Octree's root node.

    Parameters:
    ----------
    center : np.array(shape=(3,) dtype=np.float64)
    max_bound : np.array(shape=(3,) dtype=np.float64)
        Maximal point in Octree's root node.
    min_bound : np.array(shape=(3,) dtype=np.float64)
        Minimal point in Octree's root node.:

    Returns:
    --------
    np.float64
    """
    factor = 1 + 1e-5
    radius = np.max([np.max(center - min_bound), np.max(max_bound - center)]) * factor

    return radius
