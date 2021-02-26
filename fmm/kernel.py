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

TOL = 1e-6

###############################################################################
#                                   Laplace                                   #
###############################################################################


def laplace_scale(level):
    return 1/(2**level)


@numba.njit(cache=True)
def laplace_cpu(x, y):
    diff = x-y
    diff2 = diff*diff

    if np.all(diff2 < TOL):
        return np.float64(0)

    diff2 = np.sqrt(np.sum(diff2))

    return np.reciprocal(4*np.pi*diff2)


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
    """
    rx = ax-bx
    ry = ay-by
    rz = az-bz

    dist = rx**2+ry**2+rz**2
    dist_sqr = math.sqrt(dist)
    inv_dist_sqr =  1./(4*math.pi*dist_sqr)

    if math.isinf(inv_dist_sqr):
        return 0.

    return inv_dist_sqr


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


@numba.jit
def laplace_p2p(sources, targets, source_densities):

    ntargets = len(targets)
    nsources = len(sources)

    target_densities = np.zeros(shape=(ntargets))

    for i in range(ntargets):
        target = targets[i]
        potential = 0
        for j in range(nsources):
            source = sources[j]
            source_density = source_densities[j]
            potential += laplace_cpu(target, source)*source_density

        target_densities[i] = potential

    return target_densities


@numba.njit(cache=True)
def laplace_gram_matrix(sources, targets):
    """
    Compute Gram matrix of Laplace kernel.

    Parameters:
    -----------
    sources : np.array(shape=(n, 3))
        The n source locations on a surface.
    targets : np.array(shape=(m, 3))
        The m target locations on a surface.

    Returns:
    --------
    np.array(shape=(n, m))
        The Gram matrix.
    """
    n_sources = len(sources)
    n_targets = len(targets)

    matrix = np.zeros(shape=(n_targets, n_sources))

    for row_idx in range(n_targets):
        target = targets[row_idx]
        for col_idx in range(n_sources):
            source = sources[col_idx]
            matrix[row_idx][col_idx] = laplace_cpu(target, source)

    return matrix


KERNELS = {
    'laplace': {
        'eval': laplace_cpu,
        'scale': laplace_scale,
        'cuda': laplace_cuda,
        'dense_gram': laplace_gram_matrix,
        'implicit_gram': laplace_implicit_gram_matrix,
        'p2p': laplace_p2p
    },
}
