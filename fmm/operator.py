"""
Operator helper methods.
"""
import os
import pathlib

import numpy as np

from fmm.density import Potential
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent


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
        kernel_function,
        upward_check_surface,
        upward_equivalent_surface,
        alpha=None
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
    alpha : float [optional]
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
        sources=upward_equivalent_surface,
        targets=upward_check_surface
    )

    e2c = gram_matrix(
        kernel_function=kernel_function,
        sources=upward_check_surface,
        targets=upward_equivalent_surface
    )

    # Compute SVD of Gram Matrix

    if alpha is None:
        uc2e_v, uc2e_u, _, _ = compute_pseudo_inverse(c2e)
        dc2e_v, dc2e_u, _, _ = compute_pseudo_inverse(e2c)
    else:
        uc2e_v, uc2e_u, _, _ = compute_pseudo_inverse(c2e, alpha)
        dc2e_v, dc2e_u, _, _ = compute_pseudo_inverse(e2c, alpha)

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


class M2LOperators:
    """
    Class to bundle precomputed M2L operators with their respective lookup table
        to translate from Hilbert key to index within the precomputed datastructure
        containing all M2L operators.
    """
    def __init__(self, config_filename=None):
        """
        Parameters:
        -----------
        config_filename : None/str
            Defaults to project config: config.json.
        """
        if config_filename is not None:
            config_filepath = PARENT / config_filename
        else:
            config_filepath = PARENT / "config.json"

        self.config = data.load_json(config_filepath)
        self.m2l_dirpath = PARENT/ self.config["operator_dirname"]

        # Load operators and key2index lookup tables
        operator_files = self.m2l_dirpath.glob('m2l*')
        index_to_key_files = self.m2l_dirpath.glob('index*')

        self.operators = {
            level: None for level in range(2, self.config['octree_max_level']+1)
        }

        self.index_to_key = {
            level: None for level in range(2, self.config['octree_max_level']+1)
        }

        for filename in operator_files:
            level = self.get_level(str(filename))
            self.operators[level] = data.load_pickle(
                f'm2l_level_{level}', self.m2l_dirpath
            )

        for filename in index_to_key_files:
            level = self.get_level(str(filename))
            self.index_to_key[level] = data.load_pickle(
                f'index_to_key_level_{level}', self.m2l_dirpath
            )

    @staticmethod
    def get_level(filename):
        """Get level from the m2l operator's filename"""
        level = int(filename.split('.')[0][-1])
        return level
