"""Cached Operator Class
"""
import numpy as np


def compute_surface(order):
    """
    Compute surface to a specified order
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
    Compute vectors to correspond to quadrature points on surface of a specified
        cube.

    Parameters:
    -----------
    surface : int
        Order of the expansion.
    radius : float
        Half side length of the octree's root node.
    level : int
        (Octree) level of cube.
    center : coordinate
        Coordinates of the centre of the cube.
    alpha : float
        Ratio between side length of surface cube and original cube.

    Returns:
    --------
    np.array(shape=(n_coeffs, 3))
        Vector of coordinates of surface points. `n_coeffs` is the number of
        points that discretise the surface of a cube.
    """

    n_coeffs = len(surface)

    # Translate box to specified centre, and scale
    scaled_radius = (0.5)**level * radius
    dilated_radius = alpha*scaled_radius

    for i in range(n_coeffs):
        surface[i] = surface[i]*dilated_radius + center

    return surface


def gram_matrix(kernel, sources, targets):
    """
    Compute Gram matrix of given kernel function. Elements are the pairwise
        interactions of sources/targets under the action of the kernel function.

    Parameters:
    -----------
    kernel : function
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

    matrix = np.zeros(shape=(len(sources), len(targets)))

    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            matrix[i][j] = kernel(source, target)

    return matrix


def compute_check_to_equivalent(
    kernel_function, upward_check_surface, upward_equivalent_surface):

    # Compute Gram Matrix of upward check to upward equivalent surfaces
    upward_check_to_equivalent = gram_matrix(
        kernel_function, upward_check_surface, upward_equivalent_surface)

    # Compute SVD of Gram Matrix
    u, s, v_t = np.linalg.svd(upward_check_to_equivalent)

    # Compute Pseudo-Inverse of Gram matrix
    tol = 1e-1
    for i, val in enumerate(s):
        if  abs(val) < tol:
            s[i] = 0
        else:
            s[i] = 1/val

    s = np.diag(s)

    uc2e_v = np.matmul(v_t.T, s)
    uc2e_u = u.T

    dc2e_v = np.matmul(u, s)
    dc2e_u = v_t

    return (uc2e_v, uc2e_u, dc2e_v, dc2e_u)


