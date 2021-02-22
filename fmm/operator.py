"""
Operator helper methods.
"""
import os
import pathlib

from numba import cuda
import numba
import numpy as np

import fmm.kernel as kernel


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent

# GPU Kernel parameters
BLOCK_WIDTH = 32
BLOCK_HEIGHT = 1024


@numba.njit(cache=True)
def compute_surface(order):
    """
    Compute surface to a specified order.

    Parameters:
    -----------
    order : int
        Order

    """
    n_coeffs = 6*(order-1)**2 + 2
    surf = np.zeros(shape=(n_coeffs, 3))

    surf[0] = np.array([-1, -1, -1])
    count = 1

    # Hold x fixed
    for i in range(order-1):
        for j in range(order-1):
            surf[count][0] = -1
            surf[count][1] = (2*(i+1)-(order-1))/(order-1)
            surf[count][2] = (2*j-(order-1))/(order-1)
            count += 1

    # Hold y fixed
    for i in range(order-1):
        for j in range(order-1):
            surf[count][0] = (2*j-(order-1))/(order-1)
            surf[count][1] = -1
            surf[count][2] = (2*(i+1)-(order-1))/(order-1)
            count += 1

    # Hold z fixed
    for i in range(order-1):
        for j in range(order-1):
            surf[count][0] = (2*(i+1)-(order-1))/(order-1)
            surf[count][1] = (2*j-(order-1))/(order-1)
            surf[count][2] = -1
            count += 1

    # Reflect about origin, for remaining faces
    for i in range(n_coeffs//2):
        surf[count+i] = -surf[i]

    return surf


@numba.njit(cache=True)
def scale_surface(surface, radius, level, center, alpha):
    """
    Shift and scale a given surface to a new center, and radius relative to the
        original surface.

    Parameters:
    -----------
    surface : np.array(shape=(n, 3))
        Original node surface, being shifted/scaled.
    radius : float
        Half side length of the Octree's root node that this surface lives in.
    level : int
        Octree level of the shifted node.
    center : coordinate
        Coordinates of the centre of the shifted node.
    alpha : float
        Ratio between side length of shifted/scaled node and original node.

    Returns:
    --------
    np.array(shape=(n_coeffs, 3))
        Vector of coordinates of surface points. `n_coeffs` is the number of
        points that discretise the surface of a node.
    """

    n_coeffs = len(surface)

    # Translate box to specified centre, and scale
    scaled_radius = (0.5)**level * radius
    dilated_radius = alpha*scaled_radius

    scaled_surface = np.copy(surface)

    for i in range(n_coeffs):
        scaled_surface[i] = surface[i]*dilated_radius + center

    return scaled_surface


# @numba.njit(cache=True)
def gram_matrix(kernel_function, sources, targets):
    """
    Compute Gram matrix of given kernel function. Elements are the pairwise
        interactions of sources/targets under the action of the kernel function.

    Parameters:
    -----------
    kernel_function : function
        Kernel function
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
            matrix[row_idx][col_idx] = kernel_function(target, source)

    return matrix


@cuda.jit
def implicit_gram_matrix(
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

            row_sum += kernel.laplace_cuda(sx, sy, sz, tx, ty, tz)*tmp[i]

    cuda.atomic.add(result, (threadyInd, idx), row_sum)


def compute_pseudo_inverse(matrix, cond=None):
    """
    Compute pseudo-inverse using SVD of a given matrix. Based on the backward-
        stable pseudo-inverse introduced by Malhotra et al. 2018.

    Parameters:
    ----------
    matrix: np.array(shape=any)
    cond : float
        Optional regularisation parameter

    Returns:
    --------
    (np.array, np.array, np.array, np.array)
        Tuple, where first two elements multiply together to form the inverse
        of the matrix, and the second two elements multiply together to form the
        inverse of the matrix's transpose.
    """
    # Compute SVD
    u, s, v_t = np.linalg.svd(matrix)

    # Compute inverse of diagonal matrix with regularisation, hand tuned.
    if cond is None:
        cond = max(s)*0.00725

    a = cond*np.ones(len(s)) + s*s

    tol = np.finfo(float).eps*4*max(a)

    for i, val in enumerate(a):
        if  abs(val) < tol:
            a[i] = 0
        else:
            a[i] = 1/val

    s = np.matmul(np.diag(a), np.diag(s))

    # Components of the inverse of the matrix
    av = np.matmul(v_t.T, s)
    au = u.T

    return av, au


def compute_check_to_equivalent_inverse(
        kernel_function,
        check_surface,
        equivalent_surface,
        cond=None
        ):
    """
    Compute the inverse of the upward check-to-equivalent gram matrix, and the
        same for it's transpose - which amounts to the inverse of the downward
        check-to-equivalent gram matrix.

    Parameters:
    -----------
    kernel_function : function
    check_surface : np.array(shape=(n, 3))
    equivalent_surface : np.array(shape=(n, 3))
    cond : float [optional]
        Regularisation parameter

    Returns:
    --------
    tuple
        Tuple of upward check-to-equivalent inverse stored as two compoennts,
        and downard check-to-equivalent inverse stored as two components.
    """
    # Compute Gram Matrix of upward check to upward equivalent surfaces
    c2e = gram_matrix(
        kernel_function=kernel_function,
        sources=equivalent_surface,
        targets=check_surface
    )

    # Compute Inverse of Gram Matrix
    if cond is None:
        c2e_inverse_v, c2e_inverse_u = compute_pseudo_inverse(c2e)
    else:
        c2e_inverse_v, c2e_inverse_u = compute_pseudo_inverse(c2e, cond)

    return c2e_inverse_v, c2e_inverse_u


# def p2p(kernel_function, targets, sources, source_densities):
#     """
#     Directly calculate potential at m targets from n sources.

#     Parameters:
#     -----------
#     kernel_function : function
#     targets : np.array(shape=(m, 3))
#     sources : np.array(shape=(n, 3))
#     source_densities : np.array(shape=(n))

#     Returns:
#     --------
#     Potential
#         Potential denities at all target points from from all sources.
#     """

#     # Potential at target locations
#     target_densities = np.zeros(shape=(len(targets)))

#     for i, target in enumerate(targets):
#         potential = 0
#         for j, source in enumerate(sources):
#             source_density = source_densities[j]
#             potential += kernel_function(target, source)*source_density

#         target_densities[i] = potential

#     return Potential(targets, target_densities)
