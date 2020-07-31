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

    if level < 0:
        return -1
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
    Compute Hilbert key from a 4d index, The 4D index is composed as
        [xidx, yidx, zidx, level], corresponding to the physical index of a node
        in a partitioned box.

    """
    octree_depth = index[-1]
    key = 0
    for level in range(octree_depth):
        key |= (index[2] & (1 << level)) << 2 * level
        key |= (index[1] & (1 << level)) << 2 * level + 1
        key |= (index[0] & (1 << level)) << 2 * level + 2

    key += get_level_offset(octree_depth)

    return key


@numba.njit(cache=True)
def get_4d_index_from_key(key):
    """
    The 4D index is composed as [xidx, yidx, zidx, level], corresponding to
        physical index of node in partitioned box.

    Parameters:
    -----------
    key : int
        Hilbert key

    Returns:
    --------
    vec : np.array(shape=(4), type=np.int64)
    """
    level = get_level(key)
    key = key - get_level_offset(level)
    vec = np.zeros(4, np.int64)
    vec[3] = level
    for level_index in range(level):
        vec[2] |= (key & (1 << 3 * level_index)) >> 2 * level_index
        vec[1] |= (key & (1 << 3 * level_index + 1)) >> (2 * level_index + 1)
        vec[0] |= (key & (1 << 3 * level_index + 2)) >> (2 * level_index + 2)
    return vec


@numba.njit(cache=True)
def get_4d_index_from_point(point, level, x0, r0):
    """Get 4d index from point in 3 dimensions."""
    vec = np.empty(4, dtype=np.int64)
    vec[3] = level
    xmin = x0 - r0
    dx = 2 * r0 / (1 << level)
    vec[:3] = np.floor((point - xmin) / dx).astype(np.int64)
    return vec


@numba.njit(cache=True)
def get_key_from_point(point, level, x0, r0):
    """Get key from 3d point."""
    vec = get_4d_index_from_point(point, level, x0, r0)
    return get_key_from_4d_index(vec)


@numba.njit(cache=True)
def get_keys_from_points_array(points, level, x0, r0):
    """Get keys from array of points."""
    npoints = len(points)
    keys = np.empty(npoints, dtype=np.int64)
    vecs = np.empty((npoints, 4), dtype=np.int64)
    vecs[:, -1] = level
    xmin = x0 - r0
    dx = 2 * r0 / (1 << level)
    vecs[:, :3] = np.floor((points - xmin) / dx).astype(np.int64)
    for index in range(npoints):
        keys[index] = get_key_from_4d_index(vecs[index, :])
    return keys


@numba.njit(cache=True)
def get_center_from_4d_index(vec, x0, r0):
    """Get center of box from 4d index."""
    xmin = x0 - r0
    dx = 2 * r0 / (1 << vec[-1])
    return (vec[:3] + .5) * dx + xmin


@numba.njit(cache=True)
def get_center_from_key(key, x0, r0):
    """Get center of box from a key"""
    vec = get_4d_index_from_key(key)
    return get_center_from_4d_index(vec, x0, r0)


@numba.njit(cache=True)
def get_parent(key):
    """Return the parent key."""
    level = get_level(key)
    return (remove_level_offset(key) >> 3) + get_level_offset(level - 1)


@numba.njit(cache=True)
def get_children(key):
    """Return the parent key."""
    level = get_level(key)
    return (remove_level_offset(key) << 3) + get_level_offset(level + 1) + np.arange(8)


@numba.njit(cache=True)
def get_number_of_all_nodes(level):
    """Return number of all nodes up to a given level."""
    return get_level_offset(level + 1)


def compute_neighbors(key):
    """
    Compute *all* near neighbors of a given key.

    Parameters:
    -----------
    key : int
        Hilbert key.

    Returns:
    --------
    list[int]
        List of neighbors.
    """
    vec = get_4d_index_from_key(key)

    max_coord = 2**get_level(key)

    count = -1
    offset = np.zeros(4, dtype=np.int64)

    neighbors = []

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
                    neighbors.append(neighbor_key)

    # Remove the key itself from the list of neighbors
    neighbors = set(neighbors)
    neighbors.remove(key)

    return list(neighbors)


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
