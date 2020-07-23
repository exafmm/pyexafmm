"""
Operator helper methods.
"""
import numpy as np

from fmm.density import Potential
import fmm.rotation_matrices as rotations


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

    matrix = np.zeros(shape=(len(targets), len(sources)))

    for row_idx, target in enumerate(targets):
        for col_idx, source in enumerate(sources):
            matrix[row_idx][col_idx] = kernel_function(target, source)

    return matrix


def compute_pseudo_inverse(matrix, alpha=None):
    """
    Compute pseudo-inverse using SVD of a given matrix. Based on the backward-
        stable pseudo-inverse introduced by Malhotra et al. 2018.

    Parameters:
    ----------
    matrix: np.array(shape=any)
    alpha : float
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

    if alpha is None:
        alpha = max(s)*0.00725
    else:
        alpha = 0

    a = alpha*np.ones(len(s)) + s*s

    tol = np.finfo(float).eps*4*max(a)

    for i, val in enumerate(a):
        if  abs(val) < tol:
            a[i] = 0
        else:
            a[i] = 1/val

    s = np.matmul(np.diag(a), np.diag(s))

    # Compnents of the inverse of the matrix
    av = np.matmul(v_t.T, s)
    au = u.T

    # Components of the inverse of the matrix transpose
    bv = np.matmul(u, s)
    bu = v_t

    return (av, au, bv, bu)


def compute_check_to_equivalent_inverse(
        kernel_function, upward_check_surface, upward_equivalent_surface
        ):
    """
    Compute the inverse of the upward check-to-equivalent gram matrix, and the
        same for it's transpose - which amounts to the inverse of the downward
        check-to-equivalent gram matrix.

    Parameters:
    -----------
    kernel_function : function
    upward_check_surface : np.array(shape=(n, 3))
    upward_equivalent_surface : np.array(shape=(n, 3))

    Returns:
    --------
    tuple
        Tuple of upward check-to-equivalent inverse stored as two compoennts,
        and downard check-to-equivalent inverse stored as two components.
    """
    # Compute Gram Matrix of upward check to upward equivalent surfaces
    c2e = gram_matrix(
        kernel_function=kernel_function,
        sources=upward_equivalent_surface,
        targets=upward_check_surface
    )

    # Compute SVD of Gram Matrix
    uc2e_v, uc2e_u, dc2e_v, dc2e_u = compute_pseudo_inverse(c2e)

    return (uc2e_v, uc2e_u, dc2e_v, dc2e_u)


def p2p(kernel_function, targets, sources, source_densities):
    """
    Directly calculate potential at m targets from n sources.

    Parameters:
    -----------
    kernel_function : function
    targets : np.array(shape=(m, 3))
    sources : np.array(shape=(n, 3))
    source_densities : np.array(shape=(n))

    Returns:
    --------
    Potential
        Potential denities at all target points from from all sources.
    """

    # Potential at target locations
    target_densities = np.zeros(shape=(len(targets)))

    for i, target in enumerate(targets):
        potential = 0
        for j, source in enumerate(sources):
            source_density = source_densities[j]
            potential += kernel_function(target, source)*source_density

        target_densities[i] = potential

    return Potential(targets, target_densities)


def compute_equivalent_orientations(vector):
    """
    Compute equivalent rotations/reflections of a given vector. This is a helper
        method to look up the correct precomputed M2L operator.

    Parameters:
    -----------
    vector : np.array(shape=(1, 3))
        A 3D cartesian vector
    """

    orientations = np.zeros(shape=(48, 3))

    tmp = vector
    idx = 0
    for _ in range(4):
        tmp = np.matmul(rotations.X_ROT_90, tmp)
        orientations[idx] = tmp
        idx += 1

    for _ in range(4):
        tmp = np.matmul(rotations.Y_ROT_90, tmp)
        orientations[idx] = tmp
        idx += 1

    for _ in range(4):
        tmp = np.matmul(rotations.Z_ROT_90, tmp)
        orientations[idx] = tmp
        idx += 1

    for i in range(12):
        orientations[idx] = np.matmul(rotations.X_REF, orientations[i])
        orientations[idx+1] = np.matmul(rotations.Y_REF, orientations[i])
        orientations[idx+2] = np.matmul(rotations.Z_REF, orientations[i])
        idx += 3

    orientations = np.unique(orientations, axis=0)

    return orientations


def compute_m2l_operator_index(
    sources_relative_to_targets, source_4d_index, target_4d_index
    ):
    """
    Return the index of the m2l operator for a given source and target box.

    Parameters:
    -----------
    sources_relative_to_targets: np.array(shape=(n, 5), dtype=np.float64)
        Where `n` is the number of sources. Of the form
        [[xidx, yidx, zidx, relative_distance, key]...]
    source_4d_index : np.array(shape=(1, 4), dtype=np.int64)
        Of the form [xidx, yidx, zidx, level]
    target_4d_index : np.array(shape=(1, 4), dtype=np.int64)

    Returns:
    --------
    int
        Operator index.
    """
    relative_4d_index = source_4d_index - target_4d_index

    equivalent_orientations = \
        compute_equivalent_orientations(relative_4d_index[:3])

    # Search for the first equivalent orientation already computed
    for orientation in equivalent_orientations:
        operator_index = np.where(
            np.all(sources_relative_to_targets[:, :3] == orientation, axis=1)
        )

        if operator_index[0].size != 0:
            break

    return operator_index[0][0]
