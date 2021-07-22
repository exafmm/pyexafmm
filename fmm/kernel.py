"""
Kernels accelerated with CUDA and Numba.
"""
import numba
import numpy as np

M_INV_4PI = 1.0 / (4*np.pi)

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


@numba.njit(cache=True, error_model="numpy")
def laplace_cpu(x, y):
    """
    Laplace kernel.

    Parameters:
    -----------
    x : np.array(shape=(1, 3), dtype=float)
    y : np.array(shape=(1, 3), dtype=float)

    Returns:
    --------
    float
    """
    diff = (x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2
    tmp = np.reciprocal(np.sqrt(diff))*M_INV_4PI
    res = tmp if tmp < np.inf else 0.
    return res


@numba.njit(cache=True, parallel=False, fastmath=True, error_model="numpy")
def laplace_p2p_serial(sources, targets, source_densities):
    """
    Laplace P2P operator.

    Parameters:
    -----------
    sources : np.array(shape=(N, 3), dtype=float)
        N sources.
    targets : np.array(shape=(M, 3), dtype=float)
        M targets.
    source_densities : np.array(shape=(N), dtype=float)
        N source densities.

    Returns:
    --------
    np.array(shape=(M), dtype=float)
        Potentials.
    """
    ntargets = len(targets)
    nsources = len(sources)

    potentials = np.zeros(shape=(ntargets))

    for i in range(ntargets):
        target = targets[i]
        potential = 0
        for j in range(nsources):
            source = sources[j]
            source_density = source_densities[j]
            potential += laplace_cpu(target, source)*source_density

        potentials[i] = potential

    return potentials


@numba.njit(cache=True, parallel=True, error_model="numpy")
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
    sources : np.array(shape=(N, 3), dtype=float)
        N sources.
    targets : np.array(shape=(M, 3), dtype=float)
        M targets.
    source_densities : np.array(shape=(N), dtype=float)
        N source densities.
    source_index_pointer: np.array(nnodes+1, dtype=int)
        Created using the backend.prepare_u_list_data function.
    target_index_pointer: np.array(nnodes+1, dtype=int)
        Created using the backend.prepare_u_list_data function.
    """

    non_empty_targets = targets[target_index_pointer[0]:target_index_pointer[-1]]
    ntargets = len(non_empty_targets)
    potentials = np.zeros(shape=(ntargets, 4))

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
    Explicit Gram matrix, computed serially.

    Parameters:
    -----------
    sources : np.array(shape=(N, 3), dtype=float)
        N sources.
    targets : np.array(shape=(M, 3), dtype=float)
        M targets.

    Returns:
    --------
    np.array(shape=(M, N), dtype=float)
    """
    n_sources = len(sources)
    n_targets = len(targets)

    gram_matrix = np.zeros(shape=(n_targets, n_sources))

    for row_idx in range(n_targets):
        target = targets[row_idx]
        for col_idx in range(n_sources):
            source = sources[col_idx]
            gram_matrix[row_idx][col_idx] += laplace_cpu(target, source)

    return gram_matrix


@numba.njit(cache=True, parallel=True, fastmath=True, error_model="numpy")
def laplace_gram_matrix_parallel(sources, targets):
    """
    Explicit Gram matrix, computed in parallel.

    Parameters:
    -----------
    sources : np.array(shape=(N, 3), dtype=float)
        N sources.
    targets : np.array(shape=(M, 3), dtype=float)
        M targets.

    Returns:
    --------
    np.array(shape=(M, N), dtype=float)
    """
    n_sources = len(sources)
    n_targets = len(targets)

    gram_matrix = np.zeros(shape=(n_targets, n_sources))

    for row_idx in numba.prange(n_targets):
        target = targets[row_idx]
        for col_idx in range(n_sources):
            source = sources[col_idx]
            gram_matrix[row_idx][col_idx] += laplace_cpu(target, source)

    return gram_matrix


@numba.njit(cache=True)
def laplace_grad_cpu(x, y, c):
    """
    Numba Laplace Gradient CPU kernel.

    Parameters:
    -----------
    x : np.array(shape=(1, 3), dtype=float)
    y : np.array(shape=(1, 3), dtype=float)
    c : int
        Component to consider, c ∈ {0, 1, 2}.

    Returns:
    --------
    float
        c component of gradient.
    """

    num = x[c] - y[c]
    diff = (x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2
    invdiff = np.reciprocal(np.sqrt(diff))
    tmp = invdiff
    tmp *= invdiff
    tmp *= invdiff
    tmp *= num
    tmp *= M_INV_4PI
    res = tmp if tmp < np.inf else 0.
    return res


@numba.njit(cache=True)
def laplace_gradient(sources, targets, source_densities):
    """
    Numba P2P operator for gradient of Laplace kernel.

    Parameters:
    -----------
    sources : np.array(shape=(N, 3), dtype=float)
        N sources.
    targets : np.array(shape=(M, 3), dtype=float)
        M targets.
    source_densities : np.array(shape=(N), dtype=float)
        N source densities.

    Returns:
    --------
    np.array(shape=(M, 3), dtype=float)
    """
    nsources = len(sources)
    ntargets = len(targets)

    gradients = np.zeros((ntargets, 3), np.float32)

    for i in range(ntargets):
        target = targets[i]
        for j in range(nsources):
            source = sources[j]
            gradients[i][0] -= source_densities[j]*laplace_grad_cpu(target, source, 0)
            gradients[i][1] -= source_densities[j]*laplace_grad_cpu(target, source, 1)
            gradients[i][2] -= source_densities[j]*laplace_grad_cpu(target, source, 2)
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