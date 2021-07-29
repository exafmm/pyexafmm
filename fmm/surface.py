"""
Surface creation utilities
"""
import numba
import numpy as np


def compute_surface(order, dtype):
    """
    Compute surface to a specified order.

    Parameters:
    -----------
    order : int
        Order, n_coefficients = 6(order-1)^2 + 2
    dtype : type
        np.float32, np.float64
    """
    n_coeffs = 6*(order-1)**2 + 2
    surf = np.zeros(shape=(n_coeffs, 3), dtype=dtype)

    surf[0] = np.array([-1, -1, -1], dtype=dtype)
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
    surf : np.array(shape=(n, 3), dtype=float)
        Discretized surface.
    radius : float
        Half side length of the octree root node.
    level : int
        Tree level of the scaled node.
    center : np.array(shape=(1, 3), dtype=float)
       Centre of the shifted node.
    alpha : float
        Relative size of surface.

    Returns:
    --------
    np.array(shape=(n_coeffs, 3), dtype=float)
        Scaled and shifted surface.
    """
    n_coeffs = len(surf)
    dtype = surf.dtype.type
    # Translate box to specified centre, and scale
    scaled_radius = (0.5)**level*radius
    dilated_radius = alpha*scaled_radius

    # Cast center and radius
    dilated_radius = dtype(dilated_radius)
    center = center.astype(dtype)

    scaled_surf = np.zeros_like(surf)

    for i in range(n_coeffs):
        scaled_surf[i] = surf[i]*dilated_radius + center

    return scaled_surf
