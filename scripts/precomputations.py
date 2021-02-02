"""
Run all operator precomputations, and compressions.
"""
from numba import cuda
import numpy as np

import adaptoctree.morton as morton

import fmm.operator as operator


TPB = 32

@cuda.jit
def matmul(A, B, C):
    """
    Fast matrix-matrix multiply using shared memory.
    """
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=np.float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=np.float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.
    for i in range(bpg):
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        cuda.syncthreads()

        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        cuda.syncthreads()

    C[x, y] = tmp


def compute_m2l_matrix(
        target, kernel, surface, alpha_inner, dc2e_v, dc2e_u, v_list, depth
    ):
    """
    Compute dense M2L matrix for a given target node.

    Parameters:
    -----------
    target : np.int64
        Target Hilbert Key.
    kernel : function
    surface : np.array(shape=(n, 3), dtype=np.float64)
    alpha_inner : np.float64
        Ratio between side length of shifted/scaled node and original node.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Center of Octree's root node.
    r0 : np.float64
        Half side length of Octree's root node.
    dc2e_v : np.array(shape=(n, n))
        First component of the inverse of the downward-check-to-equivalent
        Gram matrix at level 0.
    dc2e_v : np.array(shape=(n, n))
        Second component of the inverse of the downward-check-to-equivalent
        Gram matrix at level 0.

    Returns:
    --------
    np.array(shape=(n, n))
        'Short and Fat' M2L matrix
    """
    # Get level
    level = morton.find_level(target)

    # Container for results
    se2tc_list = []

    # Compute target check surface
    target_center = morton.find_relative_center_from_key(
        key=target,
        depth=depth
    )

    target_check_surface = operator.scale_surface(
        surface=surface,
        radius=r0,
        level=level,
        center=target_center,
        alpha=alpha_inner
    )

    # Compute m2l matrices
    for source in v_list:

        source_center = morton.find_relative_center_from_key(
            key=source,
            depth=depth
        )

        source_equivalent_surface = operator.scale_surface(
            surface=surface,
            radius=r0,
            level=level,
            center=source_center,
            alpha=alpha_inner
        )

        se2tc = operator.gram_matrix(
            kernel_function=kernel,
            sources=source_equivalent_surface,
            targets=target_check_surface
        )

        scale = kernel.scale(level)

        se2tc_list.append(se2tc)

    se2tc = np.bmat(se2tc_list)
    # To be transferred to GPU
    tmp = np.matmul(dc2e_u, se2tc)
    m2l_matrix = np.matmul(scale*dc2e_v, tmp)

    return m2l_matrix