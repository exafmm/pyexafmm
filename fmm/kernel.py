"""
Kernels accelerated with CUDA and Numba.
"""
import math

import numba
import numpy as np

M_INV_4PI = 1.0 / (4*np.pi)

TOL = 1e-6

###############################################################################
#                                   Laplace                                   #
###############################################################################


@numba.njit(
    [numba.float32(numba.int32)],
    cache=True
)
def laplace_scale(level):
    """
    Level scale for the Laplace kernel.

    Parameters:
    -----------
    level : np.int32

    Returns:
    --------
    np.float32
    """
    return numba.float32(1/(2**level))


@numba.njit(cache=True)
def laplace_cpu(x, y):
    """
    Numba Laplace CPU kernel.

    Parameters:
    -----------
    x : np.array(shape=(3), dtype=np.float32)
        Source coordinate.
    y : np.array(shape=(3), dtype=np.float32)
        Target coordinate.

    Returns:
    --------
    np.float32
    """
    diff = (x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2
    tmp = np.reciprocal(np.sqrt(diff))*M_INV_4PI
    res = tmp if tmp < np.inf else 0.
    return res


@numba.njit(cache=True, parallel=False, fastmath=True, error_model="numpy")
def laplace_p2p_serial(sources, targets, source_densities):
    """
    Numba P2P operator for Laplace kernel.

    Parameters:
    -----------
    sources : np.array(shape=(n, 3), dtype=np.float32)
        The n source locations on a surface.
    targets : np.array(shape=(m, 3), dtype=np.float32)
        The m target locations on a surface.
    source_densities : np.array(shape=(m,), dtype=np.float32)
        Charge densities at source coordinates.

    Returns:
    --------
    np.array(shape=(ntargets), dtype=np.float32)
        Target potential densities.
    """
    ntargets = len(targets)
    nsources = len(sources)

    target_densities = np.zeros(shape=(ntargets), dtype=np.float32)

    for i in range(ntargets):
        target = targets[i]
        potential = 0
        for j in range(nsources):
            source = sources[j]
            source_density = source_densities[j]
            potential += laplace_cpu(target, source)*source_density

        target_densities[i] = potential

    return target_densities


@numba.njit(cache=True, parallel=True, fastmath=True, error_model="numpy")
def laplace_p2p_parallel(
        sources,
        targets,
        source_densities,
        source_index_pointer,
        target_index_pointer,
    ):
    """
    Parallelised P2P function, which relies on source/target data being described
        by index pointers such that they are arranged by interaction. i.e. targets
        and all sources required for P2P evaluation have the same pointer. This
        is needed to maximise cache re-use, and reduce accesses to memory during
        the parallel evaluation.

    Parameters:
    -----------
    sources: np.array((nsources, 3), dtype=np.float32)
    targets : np.array((ntargets, 3), dtype=np.float32)
    source_densities : np.array((nsources, 3), dtype=np.float32)
    source_index_pointer: np.array(nnodes+1, dtype=np.int32)
        Created using the backend.prepare_u_list_data function.
    target_index_pointer: np.array(nnodes+1, dtype=np.int32)
        Created using the backend.prepare_u_list_data function.
    """

    non_empty_targets = targets[target_index_pointer[0]:target_index_pointer[-1]]
    ntargets = len(non_empty_targets)
    target_densities = np.zeros(shape=(ntargets), dtype=np.float32)

    nleaves = len(target_index_pointer)-1

    for i in numba.prange(nleaves):
        local_targets = targets[target_index_pointer[i]:target_index_pointer[i+1]]
        local_sources = sources[source_index_pointer[i]:source_index_pointer[i+1]]
        local_source_densities = source_densities[source_index_pointer[i]:source_index_pointer[i+1]]

        local_target_densities = laplace_p2p_serial(local_sources, local_targets, local_source_densities)
        target_densities[target_index_pointer[i]:target_index_pointer[i+1]] += local_target_densities

    return target_densities


@numba.njit(cache=True, parallel=False, fastmath=True, error_model="numpy")
def laplace_gram_matrix_serial(sources, targets):
    """
    Dense Numba P2P operator for Laplace kernel.

    Parameters:
    -----------
    sources : np.array(shape=(n, 3), dtype=np.float32)
        The n source locations on a surface.
    targets : np.array(shape=(m, 3), dtype=np.float32)
        The m target locations on a surface.

    Returns:
    --------
    np.array(shape=(m, n), dtype=np.float32)
        The Gram matrix.
    """
    n_sources = len(sources)
    n_targets = len(targets)

    matrix = np.zeros(shape=(n_targets, n_sources), dtype=np.float32)

    for row_idx in range(n_targets):
        target = targets[row_idx]
        for col_idx in range(n_sources):
            source = sources[col_idx]
            matrix[row_idx][col_idx] += laplace_cpu(target, source)

    return matrix


@numba.njit(cache=True, parallel=True, fastmath=True, error_model="numpy")
def laplace_gram_matrix_parallel(sources, targets):
    """
    Dense Numba P2P operator for Laplace kernel.

    Parameters:
    -----------
    sources : np.array(shape=(n, 3), dtype=np.float32)
        The n source locations on a surface.
    targets : np.array(shape=(m, 3), dtype=np.float32)
        The m target locations on a surface.

    Returns:
    --------
    np.array(shape=(m, n), dtype=np.float32)
        The Gram matrix.
    """
    n_sources = len(sources)
    n_targets = len(targets)

    matrix = np.zeros(shape=(n_targets, n_sources), dtype=np.float32)

    for row_idx in numba.prange(n_targets):
        target = targets[row_idx]
        for col_idx in range(n_sources):
            source = sources[col_idx]
            matrix[row_idx][col_idx] += laplace_cpu(target, source)

    return matrix


KERNELS = {
    'laplace': {
        'eval': laplace_cpu,
        'scale': laplace_scale,
        'dense_gram': laplace_gram_matrix_serial,
        'dense_gram_parallel': laplace_gram_matrix_parallel,
        'p2p': laplace_p2p_serial,
        'p2p_parallel': laplace_p2p_parallel
    },
}
