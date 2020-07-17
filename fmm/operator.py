"""
Operator helper methods.
"""
import numpy as np

from fmm.density import Charge, Potential

import utils.data as data


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
    Compute vectors to correspond to quadrature points on surface of a specified
        node.

    Parameters:
    -----------
    surface : int
        Order of the expansion.
    radius : float
        Half side length of the octree's root node.
    level : int
        (Octree) level of node.
    center : coordinate
        Coordinates of the centre of the node.
    alpha : float
        Ratio between side length of surface node and original node.

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


def compute_check_to_equivalent_inverse(
    kernel_function, upward_check_surface, upward_equivalent_surface):

    # Compute Gram Matrix of upward check to upward equivalent surfaces
    upward_check_to_equivalent = gram_matrix(
        kernel_function, upward_check_surface, upward_equivalent_surface)

    # Compute SVD of Gram Matrix
    u, s, v_t = np.linalg.svd(upward_check_to_equivalent)

    # Compute Pseudo-Inverse of Gram matrix
    tol = 1e-5
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


def p2m(
        operator_dirpath,
        kernel_function,
        radius,
        level,
        center,
        leaf_sources,
        leaf_source_densities
    ):
    """
    Compute multipole expansion from sources at the leaf level supported at
        discrete points on the upward equivalent surface.

    Parameters:
    -----------
    operator_dirpath : str
        Where precomputed operators are stored.
    kernel_function : function
    center : np.array(shape=(3))
        The center of expansion.
    radius : float
        Half-side length of root node.
    level : int
        The maximium level of the octree.
    leaf_sources : np.array(shape=(n, 3))
        Sources in a given leaf node, at which multipole expansion is being
        computed.
    leaf_source_densities : np.array(shape=(n, 1))
        Source densities corresponding to leaf points.

    Returns:
    --------
    Charge
        Charge densities calculated at the discrete points on the equivalent
        surface.
    """

    surface = data.load_hdf5_to_array('surface', 'surface', operator_dirpath)

    upward_check_surface = scale_surface(
        surface=surface,
        radius=radius,
        level=level,
        center=center,
        alpha=2.95
    )

    upward_equivalent_surface = scale_surface(
        surface=surface,
        radius=radius,
        level=level,
        center=center,
        alpha=1.05
    )

    # Set unit densities at leaves for now
    leaf_source_densities = np.ones(shape=(len(leaf_sources)))

    # Lookup pseudo-inverse of kernel matrix
    uc2e_u = data.load_hdf5_to_array('uc2e_u', 'uc2e_u', operator_dirpath)
    uc2e_v = data.load_hdf5_to_array('uc2e_v', 'uc2e_v', operator_dirpath)

    scale = (1/2)**(level)

    # Compute check potential directly using leaves
    check_potential = p2p(
        kernel_function=kernel_function,
        targets=upward_check_surface,
        sources=leaf_sources,
        source_densities=leaf_source_densities
        ).density

    # Compute upward equivalent density
    tmp = np.matmul(scale*uc2e_u, check_potential)
    upward_equivalent_density = np.matmul(uc2e_v, tmp)

    return Charge(upward_equivalent_surface, upward_equivalent_density)


def compute_m2l_operator_index(sources_relative_to_targets, source_4d_index, target_4d_index):
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

    operator_index = np.where(
        np.all(sources_relative_to_targets[:, :3] == relative_4d_index[:3], axis=1)
    )

    return operator_index
