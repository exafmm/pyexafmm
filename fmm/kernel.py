"""
Kernels accelerated with CUDA and Numba.
"""
import numba
import numba.types as types
import numpy as np

M_INV_4PI = 1.0 / (4*np.pi)
ZERO = 0.

###############################################################################
#                                   Laplace                                   #
###############################################################################


@numba.njit(cache=True)
def laplace_scale(level):
    """
    Laplace kernel scale.

    Parameters:
    -----------
    level : int

    Returns:
    --------
    float
    """
    return 1./(1 << level)


@numba.njit(cache=True)
def laplace_cpu(x, y, m_inv_4pi, zero):
    """
    Laplace kernel.

    Parameters:
    -----------
    x : np.array(shape=(3), dtype=float)
    y : np.array(shape=(3), dtype=float)
    m_inv_4pi : float

    Returns:
    --------
    float
    """
    diff = (x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2
    tmp = np.reciprocal(np.sqrt(diff))*m_inv_4pi
    res = tmp if tmp < np.inf else zero
    return res


@numba.njit(cache=True, parallel=False, fastmath=True, error_model="numpy")
def laplace_p2p_serial(sources, targets, source_densities):
    """
    Numba P2P operator for Laplace kernel.

    Parameters:
    -----------
    sources : np.array(shape=(n, 3), dtype=float)
        The n source locations on a surface.
    targets : np.array(shape=(m, 3), dtype=float)
        The m target locations on a surface.
    source_densities : np.array(shape=(m,), dtype=float)
        Charge densities at source coordinates.

    Returns:
    --------
    np.array(shape=(m), dtype=float)
        Potentials.
    """
    m = len(targets)
    n = len(sources)
    dtype = targets.dtype
    m_inv_4pi = dtype.type(M_INV_4PI)
    zero = dtype.type(ZERO)

    potentials = np.zeros(shape=m, dtype=dtype)

    for i in range(m):
        target = targets[i]
        potential = 0
        for j in range(n):
            source = sources[j]
            source_density = source_densities[j]
            potential += laplace_cpu(target, source, m_inv_4pi, zero)*source_density

        potentials[i] = potential

    return potentials


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
        the parallel evaluation. This function calculates potentials and gradients.

    Parameters:
    -----------
    sources: np.array((n, 3), dtype=float)
    targets : np.array((m, 3), dtype=float)
    source_densities : np.array((n, 3), dtype=float)
    source_index_pointer: np.array(nnodes+1, dtype=int)
        Created using the backend.prepare_u_list_data function.
    target_index_pointer: np.array(nnodes+1, dtype=int)
        Created using the backend.prepare_u_list_data function.

    Returns:
    --------
    np.array(shape=(m, 4), dtype=float)
        Potentials and potentials gradients.
    """

    non_empty_targets = targets[target_index_pointer[0]:target_index_pointer[-1]]
    m = len(non_empty_targets)
    dtype = targets.dtype
    potentials = np.zeros(shape=(m, 4), dtype=dtype)

    nleaves = len(target_index_pointer)-1

    for i in numba.prange(nleaves):
        local_targets = targets[target_index_pointer[i]:target_index_pointer[i+1]]
        local_sources = sources[source_index_pointer[i]:source_index_pointer[i+1]]
        local_source_densities = source_densities[source_index_pointer[i]:source_index_pointer[i+1]]

        local_target_densities = laplace_p2p_serial(local_sources, local_targets, local_source_densities)
        local_gradients = laplace_gradient(local_sources, local_targets, local_source_densities)
        potentials[target_index_pointer[i]:target_index_pointer[i+1], 0] += local_target_densities
        potentials[target_index_pointer[i]:target_index_pointer[i+1], 1:] += local_gradients

    return potentials


@numba.njit(cache=True, parallel=False, fastmath=True, error_model="numpy")
def laplace_gram_matrix_serial(sources, targets):
    """
    Laplace Gram matrix, computed in serial.

    Parameters:
    -----------
    sources : np.array(shape=(n, 3), dtype=float)
    targets : np.array(shape=(m, 3), dtype=float)

    Returns:
    --------
    np.array(shape=(m, n), dtype=float)
    """
    n = len(sources)
    m = len(targets)
    dtype = sources.dtype
    m_inv_4pi = dtype.type(M_INV_4PI)
    zero = dtype.type(ZERO)
    gram_matrix = np.zeros(shape=(m, n), dtype=dtype)

    for row_idx in range(m):
        target = targets[row_idx]
        for col_idx in range(n):
            source = sources[col_idx]
            gram_matrix[row_idx][col_idx] += laplace_cpu(target, source, m_inv_4pi, zero)

    return gram_matrix


@numba.njit(cache=True, parallel=True, fastmath=True, error_model="numpy")
def laplace_gram_matrix_parallel(sources, targets):
    """
    Laplace Gram matrix, computed in parallel.

    Parameters:
    -----------
    sources : np.array(shape=(n, 3), dtype=float)
    targets : np.array(shape=(m, 3), dtype=float)

    Returns:
    --------
    np.array(shape=(m, n), dtype=float)
    """
    n = len(sources)
    m = len(targets)
    dtype = sources.dtype
    m_inv_4pi = dtype.type(M_INV_4PI)
    zero = dtype.type(ZERO)
    gram_matrix = np.zeros(shape=(m, n), dtype=dtype)

    for row_idx in numba.prange(m):
        target = targets[row_idx]
        for col_idx in range(n):
            source = sources[col_idx]
            gram_matrix[row_idx][col_idx] += laplace_cpu(target, source, m_inv_4pi, zero)

    return gram_matrix


@numba.njit(cache=True)
def laplace_grad_cpu(x, y, c, m_inv_4pi, zero):
    """
    Laplce gradient.

    Parameters:
    -----------
    x : np.array(shape=(3), dtype=float)
    y : np.array(shape=(3), dtype=float)
    c : int
        Component to consider, c ∈ {0, 1, 2}.
    Returns:
    --------
    float
    """

    num = x[c] - y[c]
    diff = (x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2
    invdiff = np.reciprocal(np.sqrt(diff))
    tmp = invdiff
    tmp *= invdiff
    tmp *= invdiff
    tmp *= num
    tmp *= m_inv_4pi
    res = tmp if tmp < np.inf else zero
    return res


@numba.njit(cache=True)
def laplace_gradient(sources, targets, source_densities):
    """
    Numba P2P operator for gradient of Laplace kernel.

    Parameters:
    -----------
    sources : np.array(shape=(n, 3), dtype=float)
        The n source locations on a surface.
    targets : np.array(shape=(m, 3), dtype=float)
        The m target locations on a surface.
    source_densities : np.array(shape=(m,), dtype=float)
        Charge densities at source coordinates.

    Returns:
    --------
    np.array(shape=(ntargets, 3), dtype=float)
        Target potential gradients.
    """
    n = len(sources)
    m = len(targets)
    dtype = sources.dtype
    m_inv_4pi = dtype.type(M_INV_4PI)
    zero = dtype.type(ZERO)
    gradients = np.zeros((m, 3), np.float32)

    for i in range(m):
        target = targets[i]
        for j in range(n):
            source = sources[j]
            gradients[i][0] -= source_densities[j]*laplace_grad_cpu(target, source, 0, m_inv_4pi, zero)
            gradients[i][1] -= source_densities[j]*laplace_grad_cpu(target, source, 1, m_inv_4pi, zero)
            gradients[i][2] -= source_densities[j]*laplace_grad_cpu(target, source, 2, m_inv_4pi, zero)
    return gradients


KERNELS = {
    'laplace': {
        'eval': laplace_cpu,
        'scale': laplace_scale,
        'dense_gram': laplace_gram_matrix_serial,
        'dense_gram_parallel': laplace_gram_matrix_parallel,
        'p2p': laplace_p2p_serial,
        'p2p_parallel': laplace_p2p_parallel,
        'gradient': laplace_gradient,
    },
}
