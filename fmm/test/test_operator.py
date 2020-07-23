import numpy as np
import pytest

import fmm.operator as operator
from fmm.kernel import KERNELS

ORDER = 3
NTARGETS = 4
NSOURCES = 3

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

    KT = operator.gram_matrix(
        kernel_function=kernel_function,
        sources=upward_check_surface,
        targets=upward_equivalent_surface
    )

    # Check for symmetry when sources and targets surfaces swapped
    print(K.T == KT)


@pytest.mark.parametrize(
    "K, alpha",
    [
        # Diagonal matrix, no regularisation
        (np.diag(np.ones(2)*10), 0),
        # Random full rank matrix, no regularisation
        (random()(3, 3), 0),
        # Realistic gram matrix, with regularisation
        (
            operator.gram_matrix(
                kernel_function=KERNELS['laplace'](),
                sources=upward_equivalent_surface(surface()),
                targets=upward_check_surface(surface())
                ),
            0
        )
    ]
)
def test_compute_pseudo_inverse(K, alpha):

    # Compute pseudo inverse of matrix K
    av, au, bv, bu = operator.compute_pseudo_inverse(K, alpha)
    K_inv = np.matmul(av, au)

    result = np.matmul(K, K_inv)
    expected = np.diag(np.ones(len(K)))

    print(result[0])
    assert np.all(np.isclose(result, expected, rtol=0.001))


def test_compute_pseudo_inverse_transpose():

    K = operator.gram_matrix(
            kernel_function=KERNELS['laplace'](),
            sources=upward_equivalent_surface(surface()),
            targets=upward_check_surface(surface())
        )

    KT = operator.gram_matrix(
            kernel_function=KERNELS['laplace'](),
            targets=upward_equivalent_surface(surface()),
            sources=upward_check_surface(surface())
        )

    _, _, bv, bu = operator.compute_pseudo_inverse(K)

    av, au, _, _ = operator.compute_pseudo_inverse(KT)

    expected = np.matmul(bv, bu)
    result = np.matmul(av, au)

    assert np.all(np.isclose(result, expected, rtol=0.01))



@pytest.mark.parametrize(
    "kernel_function",
    [
        KERNELS['laplace']()
    ]
)
def test_p2p(kernel_function, sources, targets):

    result = operator.p2p(
        kernel_function=kernel_function,
        targets=targets,
        sources=sources,
        source_densities=np.ones(len(sources))
    )

    potential_density = result.density

    # Check a given target
    target_idx = 0
    expected = 0
    for i, source in enumerate(sources):
        expected += kernel_function(source, targets[target_idx])

    assert expected == potential_density[target_idx]
