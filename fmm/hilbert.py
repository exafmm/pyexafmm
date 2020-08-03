"""
Helper methods to manipulate Hilbert keys.
"""
import numba
import numpy as np


@numba.njit(cache=True)
def get_level(key):
    """
    Get octree level from Hilbert key.

    Parameters:
    -----------
    key : int

    Returns:
    --------
    int
    """
    level = -1
    offset = 0
    while key >= offset:
        level += 1
        offset += 1 << 3 * level
    return level


@numba.njit(cache=True)
def get_level_offset(level):
    """
    The `offset` of a level is determined as the starting starting point of the
    Hilbert keys for a given level, so that they don't collide with keys from
    previous levels.

    Parameters:
    -----------
    level : int

    Returns:
    --------
    int
    """

    return ((1 << 3 * level) - 1) // 7


@numba.njit(cache=True)
def remove_level_offset(key):
    """
    Return Hilbert key without the level offset.

    Parameters:
    -----------
    key : int

    Returns:
    --------
    int
    """
    level = get_level(key)
    return key - get_level_offset(level)


@numba.njit(cache=True)
def get_octant(key):
    """
    Return the octant of a key. The octant is determined by the three lowest
        order bits after the level offset has been removed.

    Parameters:
    -----------
    key : int

    Returns:
    --------
    int
    """
    return remove_level_offset(key) & 7


@numba.njit(cache=True)
def get_key_from_4d_index(index):
    """
    Compute Hilbert key from a 4D index, The 4D index is composed as
        [xidx, yidx, zidx, level], corresponding to the physical index of a node
        in a partitioned box. This method works by calculating the octant
        coordinates at level 1 [x, y, z] , where x,y,z ∈ {0, 1}, and appending
        the resulting bit value `xyz` to the key. It continues to do this until
        it reaches the maximum level of the octree. Finally it adds a level
        offset to ensure that the keys at each level are unique.

    Parameters:
    -----------
    index : np.array(shape=(4,), type=np.int64)

    Returns:
    --------
    int
    """
    max_level = index[-1]
    key = 0
    for level in range(max_level):
        key |= (index[2] & (1 << level)) << 2 * level
        key |= (index[1] & (1 << level)) << 2 * level + 1
        key |= (index[0] & (1 << level)) << 2 * level + 2

    key += get_level_offset(max_level)

    return key


@numba.njit(cache=True)
def get_4d_index_from_key(key):
    """
    Compute the 4D index from a Hilbert key. The 4D index is composed as
        [xidx, yidx, zidx, level], corresponding to the physical index of a node
        in a partitioned box.

    Parameters:
    -----------
    key : int
        Hilbert key

    Returns:
    --------
    index : np.array(shape=(4,), type=np.int64)
    """
    max_level = get_level(key)
    key = key - get_level_offset(max_level)
    index = np.zeros(4, np.int64)
    index[3] = max_level
    for level in range(max_level):
        index[2] |= (key & (1 << 3 * level)) >> 2 * level
        index[1] |= (key & (1 << 3 * level + 1)) >> (2 * level + 1)
        index[0] |= (key & (1 << 3 * level + 2)) >> (2 * level + 2)
    return index


@numba.njit(cache=True)
def get_4d_index_from_point(point, level, x0, r0):
    """
    Get 4D index from point in 3 dimensions contained in the computational
        domain defined by an Octree with a root node center at x0 and a root
        node radius of r0. This method is only valid for points known to be in
        the Octree's computational domain.

    Parameters:
    -----------
    point : np.array(shape=(3,), dtype=np.float64)
    level : np.int64
    x0 : np.array(shape=(3,))
        The center of the Octree's root node.
    r0: np.float64
        The half side length of the Octree's root node

    Returns:
    --------
    np.array(shape=(4,), dtype=np.int64)
    """
    index = np.empty(4, dtype=np.int64)
    index[3] = level
    xmin = x0 - r0

    side_length = 2 * r0 / (1 << level)
    index[:3] = np.floor((point - xmin) / side_length).astype(np.int64)

    return index


@numba.njit(cache=True)
def get_key_from_point(point, level, x0, r0):
    """
    Get Hilbert key from Cartesian coordinates of a point in the computational
        domain of a given Octree.

    Parameters:
    -----------
    point : np.array(shape=(3,), dtype=np.float64)
    level : np.int64
        The level at which the key is being calculated
    x0 : np.array(shape=(3,))
        The center of the Octree's root node.
    r0: np.float64
        The half side length of the Octree's root node

    Returns:
    --------
    np.int64
    """
    vec = get_4d_index_from_point(point, level, x0, r0)
    return get_key_from_4d_index(vec)


@numba.njit(cache=True)
def get_keys_from_points(points, level, x0, r0):
    """
    Get Hilbert keys from array of points in computational domain of a given
        Octree.

    Parameters:
    -----------
    points : np.array(shape=(3,n), dtype=np.float64)
        An array of `n` points.
    level : np.int64
        The level at which the key is being calculated
    x0 : np.array(shape=(3,))
        The center of the Octree's root node.
    r0: np.float64
        The half side length of the Octree's root node

    Returns:
    --------
    np.array(n, dtype=np.int64)
    """
    npoints = len(points)
    keys = np.empty(npoints, dtype=np.int64)
    indices = np.empty((npoints, 4), dtype=np.int64)
    indices[:, -1] = level
    xmin = x0 - r0
    side_length = 2 * r0 / (1 << level)
    indices[:, :3] = np.floor((points - xmin) / side_length).astype(np.int64)
    for i in range(npoints):
        keys[i] = get_key_from_4d_index(indices[i, :])
    return keys


@numba.njit(cache=True)
def get_center_from_4d_index(index, x0, r0):
    """
    Get center of given Octree node described by a 4d index.

    Parameters:
    -----------
    index : np.array(shape=(4,), dtype=np.int64)
        4D index.
    x0 : np.array(shape=(3,))
        Center of root node of Octree.
    r0 : np.float64
        Half width length of root node.

    Returns:
    --------
    np.array(shape=(3,))

    """
    xmin = x0 - r0
    level = index[-1]
    side_length = 2 * r0 / (1 << level)
    return (index[:3] + .5) * side_length + xmin


@numba.njit(cache=True)
def get_center_from_key(key, x0, r0):
    """
    Get (Cartesian) center of node from its Hilbert key.

    Parameters:
    -----------
    key : np.int64
    x0 : np.array(shape=(3,))
    r0 : np.float64

    Returns:
    --------
    np.array(shape=(3,))
    """
    index = get_4d_index_from_key(key)
    return get_center_from_4d_index(index, x0, r0)


@numba.njit(cache=True)
def get_parent(key):
    """
    Return the parent key of a given Hilbert key.

    Parameters:
    -----------
    key : np.int64

    Returns:
    --------
    np.int64
    """
    level = get_level(key)
    return (remove_level_offset(key) >> 3) + get_level_offset(level - 1)


@numba.njit(cache=True)
def get_children(key):
    """
    Return the child keys of a given Hilbert key.

    Parameters:
    -----------
    key : np.int64

    Returns:
    --------
    np.array(shape=(nchildren,))
    """
    level = get_level(key)
    return (remove_level_offset(key) << 3) + get_level_offset(level + 1) + np.arange(8)


@numba.njit(cache=True)
def get_number_of_all_nodes(level):
    """
    Return number of all nodes up to a given level.

    Paraneters:
    -----------
    level : np.int64

    Returns:
    --------
    np.int64
    """
    return get_level_offset(level + 1)


@numba.njit(cache=True)
def compute_neighbors(key):
    """
    Compute all near neighbors of a given a node indexed by a given Hilbert key.

    Parameters:
    -----------
    key : int
        Hilbert key.

    Returns:
    --------
    np.array(shape=(nneighbors,), dtype=np.int64)
        Array of neighbors.
    """
    vec = get_4d_index_from_key(key)

    max_coord = 1 << get_level(key)

    count = -1
    offset = np.zeros(shape=(4,), dtype=np.int64)

    neighbors = -1 * np.ones(shape=(27,), dtype=np.int64)

    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                count += 1
                offset[:3] = i, j, k

                neighbor_vec = vec + offset

                # Ignore displacements that give index values outside of domain
                # at this level
                if np.any(neighbor_vec < 0) or np.any(neighbor_vec >= max_coord):
                    pass

                # Otherwise, compute the key
                else:
                    neighbor_key = get_key_from_4d_index(neighbor_vec)
                    neighbors[count] = neighbor_key

    # Remove remaining -1s
    neighbors = neighbors[neighbors != -1]

    # Remove the key itself from the list of neighbors
    neighbors = np.unique(neighbors)
    neighbors = neighbors[neighbors != key]

    return neighbors


def compute_interaction_list(key):
    """
    Compute dense interaction list of a given key.

    Parameters:
    -----------
    key : int
        Hilbert key.

    Returns:
    --------
    list[int]
        Interaction list.
    """

    if key < 9:
        return []

    parent_key = get_parent(key)

    parent_neighbors = compute_neighbors(parent_key)
    child_neighbors = compute_neighbors(key)

    interaction_list = []

    for parent_neighbor in parent_neighbors:
        children = get_children(parent_neighbor)
        for child in children:
            if child not in child_neighbors and child != key:
                interaction_list.append(child)

    return list(set(interaction_list))
