"""
Tests for the operator module.
"""
import os
import pathlib

import numpy as np
import pytest

import fmm.operator as operator
from fmm.kernel import KERNELS
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent.parent

ORDER = 3
NTARGETS = 4
NSOURCES = 3

CONFIG_FILEPATH = ROOT / 'test_config.json'
CONFIG = data.load_json(CONFIG_FILEPATH)


@pytest.fixture
def surface():
    """Order 2 surface"""
    return operator.compute_surface(order=ORDER)


@pytest.fixture
def random():
    """Random float generator"""
    np.random.seed(0)
    return np.random.rand


@pytest.fixture
def sources(random):
    """Some random sources"""
    return random(NSOURCES, 3)


@pytest.fixture
def targets(random):
    """Some random targets"""
    return random(NTARGETS, 3)


@pytest.fixture
def upward_check_surface(surface):
    return operator.scale_surface(
        surface, 1, 0, np.array([0, 0, 0]), 2.95
    )


@pytest.fixture
def upward_equivalent_surface(surface):
    return operator.scale_surface(
        surface, 1, 0, np.array([0, 0, 0]), 1.05
    )


@pytest.fixture
def gram_matrix(upward_check_surface, upward_equivalent_surface):
    return operator.gram_matrix(
        kernel_function=KERNELS['laplace'](),
        sources=upward_equivalent_surface,
        targets=upward_check_surface
    )


@pytest.mark.parametrize(
    "order",
    [
        (2),
    ]
)
def test_compute_surface(order):
    """Test surface computation"""
    surface = operator.compute_surface(order)

    # Test that surface centered at origin
    assert np.array_equal(surface.mean(axis=0), np.array([0, 0, 0]))

    # Test surface has expected dimension
    n_coeffs = 6*(order-1)**2 + 2
    assert surface.shape == (n_coeffs, 3)


@pytest.mark.parametrize(
    "radius, level, center, alpha",
    [
        (1, 0, np.array([0.5, 0.5, 0.5]), 2),
        (1, 1, np.array([0.5, 0.5, 0.5]), 2)
    ]
)
def test_scale_surface(surface, radius, level, center, alpha):
    """Test shifting/scaling surface"""

    scaled_surface = operator.scale_surface(
        surface=surface,
        radius=radius,
        level=level,
        center=center,
        alpha=alpha
    )

    # Test that the center has been shifted as expected
    assert np.array_equal(scaled_surface.mean(axis=0), center)

    # Test that the scaling of the radius is as expected
    for i in range(3):

        expected_diameter = 2*alpha*radius*(0.5)**level
        assert(
            (max(scaled_surface[:, i]) - min(scaled_surface[:, i]))
            == expected_diameter
            )

    # Test that the original surface remains unchanged
    assert ~np.array_equal(surface, scaled_surface)
    assert np.array_equal(surface, operator.compute_surface(ORDER))
