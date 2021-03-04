"""
Operator helper methods.
"""
import os
import pathlib

import numba
import numpy as np


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent


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
def scale_surface(surf, radius, level, center, alpha):
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

    n_coeffs = len(surf)

    # Translate box to specified centre, and scale
    scaled_radius = (0.5)**level * radius
    dilated_radius = alpha*scaled_radius

    scaled_surf = np.copy(surf)

    for i in range(n_coeffs):
        scaled_surf[i] = surf[i]*dilated_radius + center

    return scaled_surf
