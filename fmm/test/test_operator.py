import numpy as np
import pytest

import fmm.operator as operator
from fmm.kernel import KERNELS


@pytest.fixture
def surface():
    """Order 2 surface"""
    return operator.compute_surface(order=2)


@pytest.fixture
def random():
    """Random float generator"""
    np.random.seed(0)
    return np.random.rand


@pytest.fixture
def sources(random):
    """Some random sources"""
    return random(3, 3)


@pytest.fixture
def targets(random):
    """Some random targets"""
    return random(4, 3)


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
    assert np.array_equal(surface, operator.compute_surface(2))


@pytest.mark.parametrize(
    "kernel_function",
    [
        (KERNELS['identity']()),
        (KERNELS['laplace']())
    ]
)
def test_gram_matrix(
        kernel_function,
        sources,
        targets,
        upward_equivalent_surface,
        upward_check_surface
    ):

    # Test for gram matrix between source and target points
    K = operator.gram_matrix(
        kernel_function=kernel_function,
        sources=sources,
        targets=targets
        )

    # Check that the gram matrix is the expected dimension
    assert K.shape == (len(targets), len(sources))

    # Check that entries are as expected (0th target, 1st source)
    assert K[0][1] == kernel_function(targets[0], sources[1])

    # Test for gram matrix between upward equivalent and check surfaces
    K = operator.gram_matrix(
        kernel_function=kernel_function,
        sources=upward_equivalent_surface,
        targets=upward_check_surface
    )

    # Check for symmetry
    assert np.all(K.T == K)


def test_compute_pseudo_inverse():

    # mock (diagonal!) gram matrix
    mock_gram_matrix = np.diag(np.ones(2)*10)

    av, au, bv, bu = operator.compute_pseudo_inverse(mock_gram_matrix)

    K_inv = np.matmul(av, au)
    K_inv_expected = np.linalg.inv(np.diag(np.ones(2)*10))

    assert np.all(np.isclose(K_inv, K_inv_expected, rtol=0.001))


def test_p2p():
    pass


def test_compute_equivalent_orientations():
    pass


def test_compute_m2l_operator_index():
    pass
