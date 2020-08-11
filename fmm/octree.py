"""
Implementation of an Octree, and associated helper methods
"""
import numba
import numpy as np

import fmm.hilbert as hilbert


class Octree:
    """Data structure for handling Octrees."""

    def __init__(self, sources, targets, source_densities, maximum_level):
        """
        Initialize an Octree.

        Parameters:
        -----------
        sources : np.array(shape=(nsources, 3), dtype=np.float64)
        targets : np.array(shape=(nsources, 3), dtype=np.float64)
        source_densities : np.array(shape=(nsources,), dtype=np.float64)
        maximum_level : int
            The maximum level of the Octree.
        """

        self.sources = sources
        self.targets = targets
        self.source_densities = source_densities
        self.maximum_level = maximum_level

        # Center of the box and radius of the box
        max_bound, min_bound = compute_bounds(sources, targets)
        self.center = compute_center(max_bound, min_bound)
        self.radius = compute_radius(max_bound, min_bound, self.center)

        (
            self.source_leaf_nodes,
            self.sources_by_leafs,
            self.source_index_ptr,
        ) = assign_points_to_leaf_nodes(
            maximum_level=maximum_level,
            x0=self.center,
            r0=self.radius,
            points=sources
            )

        self.source_leaf_key_to_index = {
            key: index
            for index, key
            in enumerate(self.source_leaf_nodes)
            }

        (
            self.target_leaf_nodes,
            self.targets_by_leafs,
            self.target_index_ptr,
        ) = assign_points_to_leaf_nodes(
            maximum_level=maximum_level,
            x0=self.center,
            r0=self.radius,
            points=targets
            )

        self.target_leaf_key_to_index = {
            key: index
            for index, key
            in enumerate(self.target_leaf_nodes)
            }

        (
            self.non_empty_source_nodes,
            self.source_node_to_index
        ) = enumerate_non_empty_nodes(
            maximum_level=maximum_level,
            leaves=self.source_leaf_nodes
            )

        (
            self.non_empty_target_nodes,
            self.target_node_to_index
        ) = enumerate_non_empty_nodes(
            maximum_level=maximum_level,
            leaves=self.target_leaf_nodes
            )

        self.non_empty_source_nodes_by_level = sort_keys_by_level(self.non_empty_source_nodes)
        self.non_empty_target_nodes_by_level = sort_keys_by_level(self.non_empty_target_nodes)

        self.target_neighbors = numba_compute_neighbors(
            targets=self.non_empty_target_nodes,
            source_node_to_index=self.source_node_to_index
        )

        self.interaction_list = numba_compute_interaction_list(
            targets=self.non_empty_target_nodes,
            target_neighbors=self.target_neighbors,
            source_node_to_index=self.source_node_to_index,
            target_node_to_index=self.target_node_to_index,
        )


def nodes_per_side(level):
    """Return number of nodes along each dimension."""
    return 1 << level


def nodes_per_level(level):
    """Return the number of nodes in a given level."""
    return 1 << 3 * level


def assign_points_to_leaf_nodes(maximum_level, x0, r0, points):
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
    index_pointer : list[int]
        Pointers (indices) that link the (sorted) array of points to each
        (unique) non-empty node returned by `nodes`.
    """

    # Find Hilbert keys for each point
    assigned_nodes = hilbert.get_keys_from_points(
        points, maximum_level, x0, r0
    )

    # Need to sort point indices as we use a sorted list of nodes to check
    # whether points belong to same node.
    point_indices_by_node = list(np.argsort(assigned_nodes))
    index_pointer = []
    nodes = []

    count = 0
    previous_node = -1
    for node in assigned_nodes[point_indices_by_node]:
        if node != previous_node:
            index_pointer.append(count)
            nodes.append(node)
            previous_node = node
        count += 1
    index_pointer.append(count)

    return nodes, point_indices_by_node, index_pointer


def enumerate_non_empty_nodes(maximum_level, leaves):
    """
    Enumerate all non-empty nodes across the Octree, across all levels.

    Parameters:
    -----------
    maximum_level : int
        Depth of the Octree.
    leaves : np.array(shape=(nleaf,), dtype=npint64)
        Non empty leaf nodes referenced by their Hilbert keys.

    Returns:
    --------
    non_empty_nodes: list[int]
        The non-empty nodes in the Octree, at all levels.
    node_to_index: np.array(shape=(n_nodes))
        Maps from a Hilbert key to the index in the list of `non_empty_nodes`
        returned by this method.
    """

    node_to_index = -1*np.ones(
        shape=(hilbert.get_number_of_all_nodes(maximum_level),),
        dtype=np.int64
    )

    # node_to_index maps between hilbert keys and index values
    # The index values
    nleaves = len(leaves)
    node_to_index[leaves] = range(nleaves)

    count = nleaves

    # Traverse from leaves to root enumerating each node
    for leaf in leaves:
        parent = leaf
        while parent != 0:
            parent = hilbert.get_parent(parent)
            if node_to_index[parent] == -1:
                node_to_index[parent] = count
                count += 1

    # Returns indices of node_to_index not equal to -1
    non_empty_nodes = np.flatnonzero(node_to_index != -1)

    # node_to_index[list_of_nodes] = non-empty node values (count)
    # indices sorted by counts
    indices = np.argsort(node_to_index[non_empty_nodes])

    return non_empty_nodes[indices], node_to_index


@numba.njit(cache=True)
def _in_range(n1, n2, n3, upper_bound, lower_bound):
    """
    Check if lower_bound <= n1, n2, n3 < upper_bound.

    Parameters:
    -----------
    n1 : float
    n2 : float
    n3 : float
    bound : float

    Returns:
    --------
    bool
        True if contained in interval, false otherwise.
    """
    return (
        (lower_bound <= n1 < upper_bound)
        and (lower_bound <= n2 < upper_bound)
        and (lower_bound <= n3 < upper_bound)
        )


@numba.njit(cache=True)
def numba_compute_neighbors(targets, source_node_to_index):
    """
    Compute all, non-empty, source neighbors for the given target nodes.

    Parameters:
    -----------
    targets : np.array(shape=(ntargets,), dtype=np.int64)
        Target nodes, referenced by their Hilbert key.
    source_node_to_index : np.array(shape=(n,sources), dtype=np.int64)
        Indexed by Hilbert key, values are indices for the non-empty source
        nodes found by the `enumerate_non_empty_nodes` method.

    Returns:
    --------
    np.array((ntargets, 27), dtype=np.int64)
        The non-empty nearest neighbors.
    """

    ntargets = len(targets)

    neighbors = np.empty((ntargets, 27), dtype=np.int64)
    offset = np.zeros(4, dtype=np.int64)

    for target_index, target in enumerate(targets):

        # Calculate 4D index, in order to find bound/key
        index_4d = hilbert.get_4d_index_from_key(target)

        # Max value of the index defined by the level
        max_4d_index = 1 << index_4d[3]

        count = -1
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):

                    count += 1
                    offset[:3] = i, j, k
                    neighbor_vec = index_4d + offset

                    if _in_range(
                            n1=neighbor_vec[0],
                            n2=neighbor_vec[1],
                            n3=neighbor_vec[2],
                            lower_bound=0,
                            upper_bound=max_4d_index,
                    ):

                        neighbor_key = hilbert.get_key_from_4d_index(neighbor_vec)

                        # Check if the source node is empty, if it isn't add
                        # it to the niehgbors of the target currently being
                        # considered. Otherwise, pass.

                        if source_node_to_index[neighbor_key] != -1:
                            neighbors[target_index, count] = neighbor_key

                        else:
                            neighbors[target_index, count] = -1

                    else:
                        neighbors[target_index, count] = -1

    return neighbors


@numba.njit(cache=True)
def _is_neighbor(target_neighbors, target_index, key):
    """
    Check if a given 'key' is in the target_neighbors array of a given node
        referenced by 'target_node_index'

    Parameters:
    -----------
    target_neighbors : np.array((ntargets, 27), dtype=np.int64)
        Calculated via `numba_compute_neighbors` method.
    target_index : np.int64
        Refers to index from `enumerate_non_empty_nodes` method.
    key : np.int64
        Hilbert key of node being checked for membership of the target's
        interaction list.

    Returns:
    --------
    bool
    """
    for i in range(27):
        if target_neighbors[target_index, i] == key:
            return True
    return False


@numba.njit(cache=True)
def numba_compute_interaction_list(
        targets,
        target_neighbors,
        source_node_to_index,
        target_node_to_index
):
    """
    Compute the interaction list for all given target nodes.

    Parameters:
    -----------
    targets : np.array(shape=(ntargets,), dtype=np.int64)
        Target nodes, referenced by Hilbert key, for which interaction lists are
        being computed.
    target_neighbors : np.array(shape=(ntargets, 27), dtype=np.int64)
        Contains information on non-empty source neighbor nodes of each target,
        computed via `numba_compute_neighbors`.
    source_node_to_index : np.array(shape=(nsources,), dtype=np.int64)
    target_node_to_index : np.array(shape=(ntargets,), dtype=np.int64)

    Returns:
    --------
    np.array(shape=(ntargets, 27, 8), dtype=np.int64)
        Interaction list, each target in n targets has an associated (27, 8)
        matrix associated with it.
    """

    ntargets = len(targets)

    interaction_list = -1 * np.ones((ntargets, 27, 8), dtype=np.int64)

    for target_index, target in enumerate(targets):
        target_level = hilbert.get_level(target)

        if target_level >= 2:

            # Find parent
            parent = hilbert.get_parent(target)

            # Find parent neighbors
            parent_index = target_node_to_index[parent]
            parent_neighbors = target_neighbors[parent_index]

            for parent_neighbor_index, parent_neighbor in enumerate(parent_neighbors):

                if parent_neighbor != -1:

                    parent_neighbor_children = hilbert.get_children(parent_neighbor)

                    for neigbhor_child_index, neighbor_child in enumerate(parent_neighbor_children):

                        is_neighbor = _is_neighbor(
                            target_neighbors=target_neighbors,
                            target_index=target_index,
                            key=neighbor_child
                            )


                        if source_node_to_index[neighbor_child] != -1 and ~is_neighbor:
                            interaction_list[target_index, parent_neighbor_index, neigbhor_child_index] = neighbor_child

    return interaction_list


def sort_keys_by_level(keys):
    """
    Return dict with nodes sorted by level.

    Parameters:
    -----------
    keys : np.array(shape=(nkeys,) dtype=np.int64)

    Returns:
    --------
    dict[int, np.array(dtype=np.int64)]
    """
    sorted_keys, index_pointer = _numba_sort_keys_by_level(keys)

    level_to_keys = {}
    max_level = len(index_pointer) - 1
    for level in range(max_level):
        level_to_keys[level] = sorted_keys[index_pointer[level] : index_pointer[1 + level]]

    return level_to_keys


@numba.njit(cache=True)
def _numba_sort_keys_by_level(keys):
    """
    Sort nodes, indexed by Hilbert keys, by level.

    Parameters:
    -----------
    keys : np.array(shape=(nkeys,) dtype=np.int64)

    Returns:
    --------
    (np.array(shape=(nkey,), dtype=np.int64), list[int])
        Return the sorted keys and the index pointer as a tuple.
    """

    # Store indices for start/end of keys at a given level in the index pointer
    initial_index = 0
    index_pointer = [initial_index]

    sorted_keys = np.sort(keys)

    next_level = 1
    next_offset = hilbert.get_level_offset(next_level)

    for index, key in enumerate(sorted_keys):
        if key >= next_offset:

            index_pointer.append(index)

            next_level += 1
            next_offset = hilbert.get_level_offset(next_level)

    final_index = len(keys)

    index_pointer.append(final_index)
    return sorted_keys, index_pointer


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
