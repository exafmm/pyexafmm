"""
Tests for the surface module.
"""
import os
import pathlib

import numpy as np
import pytest

import fmm.surface as surface
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent.parent


def surf32():
    return surface.compute_surface(5, np.float32)


def surf64():
    return surface.compute_surface(5, np.float64)


@pytest.mark.parametrize(
    "order, dtype",
    [
        (2, np.float32),
        (2, np.float64),
    ]
)
def test_compute_surface(order, dtype):
    """Test surface computation"""
    surf =  surface.compute_surface(order, dtype)

    # Test that surface centered at origin
    assert np.array_equal(surf.mean(axis=0), np.array([0, 0, 0]))

    # Test surface has expected dimension
    n_coeffs = 6*(order-1)**2 + 2
    assert surf.shape == (n_coeffs, 3)

    # Test that surface is of correct type
    assert isinstance(surf[0, 0], dtype)


@pytest.mark.parametrize(
    "surf, radius, level, center, alpha, dtype",
    [
        (surf32(), 1., 0, np.array([0.5, 0.5, 0.5]), 2, np.float32),
        (surf64(), 1., 1, np.array([0.5, 0.5, 0.5]), 2, np.float64),
    ]
)
def test_scale_surface(surf, radius, level, center, alpha, dtype):
    """Test shifting/scaling surface"""

    scaled_surf = surface.scale_surface(
        surf=surf,
        radius=radius,
        level=level,
        center=center,
        alpha=alpha
    )

    # Test that the center has been shifted as expected
    assert np.array_equal(scaled_surf.mean(axis=0), center)

    # Test that the scaling of the radius is as expected
    for i in range(3):

        expected_diameter = 2*alpha*radius*(0.5)**level
        assert(
            (max(scaled_surf[:, i]) - min(scaled_surf[:, i])) == expected_diameter
            )

    # Test that the original surface remains unchanged
    assert ~np.array_equal(surf, scaled_surf)
    assert np.array_equal(surf,  surface.compute_surface(5, dtype))

    # Test the data type
    assert isinstance(scaled_surf[0, 0], dtype)
