"""
Kernels accelerated with CUDA and Numba.
"""
import math

import numba
from numba import cuda
import numpy as np

# GPU Kernel parameters
BLOCK_WIDTH = 32
BLOCK_HEIGHT = 1024
M_INV_4PI = 1.0 / (4*np.pi)

TOL = 1e-6

###############################################################################
#                                   Laplace                                   #
###############################################################################


@numba.njit(cache=True)
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
    return np.float32(1/(2**level))


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


@cuda.jit(device=True)
def laplace_cuda(ax, ay, az, bx, by, bz):
    """
    Numba-Cuda Laplace device kernel.

    Parameters:
    -----------
    ax : np.float32
        'x' coordinate of a point.
    ay : np.float32
        'y' coordinate of a point.
    az : np.float32
        'z' coordinate of a point.
    bx : np.float32
        'x' coordinate of a point.
    by : np.float32
        'y' coordinate of a point.
    bz : np.float32
        'z' coordinate of a point.

    Returns:
    --------
    nb.cuda.float32
    """
    rx = ax-bx
    ry = ay-by
    rz = az-bz

    dist = rx**2+ry**2+rz**2
    dist_sqr = math.sqrt(dist)
    inv_dist_sqr =  1./(4*math.pi*dist_sqr)

    if math.isinf(inv_dist_sqr):
        return 0.

    return numba.float32(inv_dist_sqr)


@cuda.jit
def laplace_implicit_gram_matrix(
        sources,
        targets,
        rhs,
        result,
        height,
        width,
        idx
    ):
    """
    Implicitly apply the Gram matrix to a vector, computing Green's function on
        the fly on the device, and multiplying by random matrices to approximate
        the basis. Multiple RHS are computed in a loop, specified by the index.

    Parameters:
    -----------
    sources : np.array(shape=(nsources, 3))
        nsources rows of gram matrix
    targets : np.array(shape=(ntargets, 3))
        ntargets columns of gram matrix
    rhs : np.array(shape=(ntargets, nrhs))
        Multiple right hand sides, indexed by 'idx'
    result : np.array(shape=(height, nrhs))
        Dimensions of result matrix defined by matrix height
        and number of right hand sides
    height : int
        'height' of implicit gram matrix. Defined by m, where
        m > n, and m is either ntargets or nsources.
    width : int
        'width' of implicit gram matrix.
    idx: int
        RHS index.
    """

    blockWidth = cuda.shared.array(1, numba.int32)
    blockxInd = cuda.shared.array(1, numba.int32)
    blockyInd = cuda.shared.array(1, numba.int32)

    # Calculate once, and share amongs thread-block
    if cuda.threadIdx.x == 0:
        if (cuda.blockIdx.x + 1)*BLOCK_WIDTH <= width:
            blockWidth[0] = numba.int32(BLOCK_WIDTH)
        else:
            blockWidth[0] = numba.int32(width % BLOCK_WIDTH)

        blockxInd[0] = numba.int32(cuda.blockIdx.x*BLOCK_WIDTH)
        blockyInd[0] = numba.int32(cuda.blockIdx.y*BLOCK_HEIGHT)

    cuda.syncthreads()

    # Extract tile of RHS, of size up to BLOCK_WIDTH
    tmp = cuda.shared.array(BLOCK_WIDTH, numba.float32)

    if cuda.threadIdx.x < blockWidth[0]:
        tmp[cuda.threadIdx.x] = rhs[idx][blockxInd[0]+cuda.threadIdx.x]

    cuda.syncthreads()

    # Accumulate matvec of tile into variable
    row_sum = 0

    threadyInd = blockyInd[0]+cuda.threadIdx.x

    if threadyInd < height:
        for i in range(blockWidth[0]):
            col_idx = blockxInd[0] + i
            row_idx = threadyInd
            source = sources[row_idx]
            target  = targets[col_idx]

            sx = source[0]
            sy = source[1]
            sz = source[2]

            tx = target[0]
            ty = target[1]
            tz = target[2]

            row_sum += laplace_cuda(sx, sy, sz, tx, ty, tz)*tmp[i]

    cuda.atomic.add(result, (threadyInd, idx), row_sum)


@cuda.jit
def laplace_implicit_gram_matrix_blocked(
        sources,
        targets,
        rhs,
        result,
        height,
        width,
        sub_height,
        sub_width,
        idx
    ):
    """
    Implicitly apply the Gram matrix to a vector, computing Green's function on
        the fly on the device, and multiplying by random matrices to approximate
        the basis. Multiple RHS are computed in a loop, specified by the index.
        This kernel assumes that the sources/targets specified by the user
        specify multiple Gram matrices, i.e. the global Gram matrix is 'blocked',

        e.g. gram = ((g_1) | (g_2) | ... | (g_{n-1}) | g_n)

        therefore, one additionally needs to specify the dimensions of the
        'sub' gram matrices in order to pick out the correct matrix elements
        for application to a given RHS.

    Parameters:
    -----------
    sources : np.array(shape=(nsources, 3), dtype=np.float32)
        nsources rows of gram matrix
    targets : np.array(shape=(ntargets, 3), dtype=np.float32)
        ntargets columns of gram matrix
    rhs : np.array(shape=(ntargets, nrhs), dtype=np.float32)
        Multiple right hand sides, indexed by 'idx'
    result : np.array(shape=(height, nrhs), dtype=np.float32)
        Dimensions of result matrix defined by matrix height
        and number of right hand sides
    height : np.int32
        'height' of global Gram matrix. Defined by m, where
        m > n, and m is either ntargets or nsources.
    width : np.int32
        'width' of Global gram matrix.
    sub_height : np.int32
        height of sub-Gram matrix. Defined by m, where
        m > n, and m is either ntargets or nsources.
    sub_width : np.int32
        'width' of sub-Gram matrix.
    idx: np.int32
        RHS index.
    """

    blockWidth = cuda.shared.array(1, numba.int32)
    blockxInd = cuda.shared.array(1, numba.int32)
    blockyInd = cuda.shared.array(1, numba.int32)

    # Calculate once, and share amongs thread-block
    if cuda.threadIdx.x == 0:
        if (cuda.blockIdx.x + 1)*BLOCK_WIDTH <= width:
            blockWidth[0] = numba.int32(BLOCK_WIDTH)
        else:
            blockWidth[0] = numba.int32(width % BLOCK_WIDTH)

        blockxInd[0] = numba.int32(cuda.blockIdx.x*BLOCK_WIDTH)
        blockyInd[0] = numba.int32(cuda.blockIdx.y*BLOCK_HEIGHT)

    cuda.syncthreads()

    # Extract tile of RHS, of size up to BLOCK_WIDTH
    tmp = cuda.shared.array(BLOCK_WIDTH, numba.float32)

    if cuda.threadIdx.x < blockWidth[0]:
        tmp[cuda.threadIdx.x] = rhs[idx][blockxInd[0]+cuda.threadIdx.x]

    cuda.syncthreads()

    # Accumulate matvec of tile into variable
    row_sum = 0

    threadyInd = blockyInd[0]+cuda.threadIdx.x
    threadxInd = blockxInd[0]+cuda.threadIdx.x

    # Find the index of the sub-Gram matrix, dependent on orientation.
    if sub_width == width:
        submatrixInd = numba.int32(math.floor(threadyInd/sub_height))*sub_width

    if sub_height == height:
        submatrixInd = numba.int32(math.floor(threadxInd/sub_width))*sub_height

    if threadyInd < height:
        for i in range(blockWidth[0]):
            col_idx = (blockxInd[0] + i)
            row_idx = threadyInd

            # Pick out matrix element for sub-Gram matrix
            if sub_width == width:
                source = sources[col_idx+submatrixInd]
                target  = targets[row_idx]

            elif sub_height == height:
                source = sources[col_idx]
                target = targets[row_idx+submatrixInd]

            sx = source[0]
            sy = source[1]
            sz = source[2]

            tx = target[0]
            ty = target[1]
            tz = target[2]

            row_sum += laplace_cuda(sx, sy, sz, tx, ty, tz)*tmp[i]

    cuda.atomic.add(result, (threadyInd, idx), row_sum)


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
        'cuda': laplace_cuda,
        'dense_gram': laplace_gram_matrix_serial,
        'dense_gram_parallel': laplace_gram_matrix_parallel,
        'implicit_gram': laplace_implicit_gram_matrix,
        'implicit_gram_blocked': laplace_implicit_gram_matrix_blocked,
        'p2p': laplace_p2p_serial,
        'p2p_parallel': laplace_p2p_parallel
    },
}
