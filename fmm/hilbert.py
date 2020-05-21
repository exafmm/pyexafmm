"""Implementation of Hilbert keys."""
import numpy as _np
import numba as _numba


@_numba.njit(cache=True)
def get_level(key):
    """
    Get level from Hilbert key.

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

@_numba.njit(cache=True)
def get_levels_for_array(indices):
    """Get levels for array of indices ."""
    result = _np.empty(len(indices), dtype=_np.uint32)

    for index, key in enumerate(indices):
        result[index] = get_level(key)


@_numba.njit(cache=True)
def level_offset(level):
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


@_numba.njit(cache=True)
def remove_offset(key):
    """Return key without the offset."""
    level = get_level(key)
    return key - level_offset(level)


@_numba.njit(cache=True)
def get_octant(key):
    """Return the octant of a key."""
    return remove_offset(key) & 7


@_numba.njit(cache=True)
def get_key(vec):
    """Compute key from a 4d index."""
    level = vec[-1]
    key = 0
    for level_index in range(level):
        key |= (vec[2] & (1 << level_index)) << 2 * level_index
        key |= (vec[1] & (1 << level_index)) << 2 * level_index + 1
        key |= (vec[0] & (1 << level_index)) << 2 * level_index + 2

    key += level_offset(level)

    return key


@_numba.njit(cache=True)
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
    key = key - level_offset(level)
    vec = _np.zeros(4, _np.int64)
    vec[3] = level
    for level_index in range(level):
        vec[2] |= (key & (1 << 3 * level_index)) >> 2 * level_index
        vec[1] |= (key & (1 << 3 * level_index + 1)) >> (2 * level_index + 1)
        vec[0] |= (key & (1 << 3 * level_index + 2)) >> (2 * level_index + 2)
    return vec


@_numba.njit(cache=True)
def get_4d_index_from_point(point, level, x0, r0):
    """Get 4d index from point in 3 dimensions."""
    vec = _np.empty(4, dtype=_np.int64)
    vec[3] = level
    xmin = x0 - r0
    dx = 2 * r0 / (1 << level)
    vec[:3] = _np.floor((point - xmin) / dx).astype(_np.int64)
    return vec

@_numba.njit(cache=True)
def get_key_from_point(point, level, x0, r0):
    """Get key from 3d point."""
    vec = get_4d_index_from_point(point, level, x0, r0)
    return get_key(vec)

@_numba.njit(cache=True)
def get_keys_from_points_array(points, level, x0, r0):
    """Get keys from array of points."""
    npoints = len(points)
    keys = _np.empty(npoints, dtype=_np.int64)
    vecs = _np.empty((npoints, 4), dtype=_np.int64)
    vecs[:, -1] = level
    xmin = x0 - r0
    dx = 2 * r0 / (1 << level)
    vecs[:, :3] = _np.floor((points - xmin) / dx).astype(_np.int64)
    for index in range(npoints):
        keys[index] = get_key(vecs[index, :])
    return keys

@_numba.njit(cache=True)
def get_center_from_4d_index(vec, x0, r0):
    """Get center of box from 4d index."""
    xmin = x0 - r0
    dx = 2 * r0 / (1 << vec[-1])
    return (vec[:3] + .5) * dx + xmin

@_numba.njit(cache=True)
def get_center_from_key(key, x0, r0):
    """Get center of box from a key"""
    vec = get_4d_index_from_key(key)
    return get_center_from_4d_index(vec, x0, r0)

@_numba.njit(cache=True)
def get_parent(key):
    """Return the parent key."""
    level = get_level(key)
    return (remove_offset(key) >> 3) + level_offset(level - 1)

@_numba.njit(cache=True)
def get_children(key):
    """Return the parent key."""
    level = get_level(key)
    return (remove_offset(key) << 3) + level_offset(level + 1) + _np.arange(8)

@_numba.njit(cache=True)
def get_number_of_all_nodes(level):
    """Return number of all nodes up to a given level."""
    return level_offset(level + 1)
