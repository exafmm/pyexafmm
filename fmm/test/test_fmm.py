"""
Test the FMM
"""
from functools import partial

import numpy as np
import pytest

from fmm.fmm import Fmm, laplace, surface, p2p, p2m, m2m, m2l, l2l
from fmm.octree import Octree
import fmm.hilbert as hilbert

@pytest.fixture
def n_points():
    return 10


@pytest.fixture
def order():
    return 3


@pytest.fixture
def n_level_octree(n_points):
    rand = np.random.RandomState(0)
    sources = rand.rand(n_points, 3)
    targets = rand.rand(n_points, 3)

    return partial(Octree, targets, sources)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        # 1. Points closed together, potential set to large constant
        (0, 0, 1e10),
        # 2. Points well separated
        (0, 1, 1/(4*np.pi))
    ]
)
def test_laplace(x, y, expected):
    assert laplace(x, y) == expected


@pytest.mark.parametrize(
    "order, radius, level, center, alpha, ncoeffs",
    [
        (2, 1, 0, np.array([0, 0, 0]), 1, 8)
    ]
)
def test_surface(order, radius, level, center, alpha, ncoeffs):
    """
    Strategy: test surface is centered at specified location, with specified
    radius.
    """
    ndim = 3
    surf = surface(order, radius, level, center, alpha)

    # Test correct number of quadrature points
    assert surf.shape == (ncoeffs, ndim)

    # Test radius
    for i in range(ndim):
        assert max(surf[:, i]) - min(surf[:, i]) == 2*radius

    # Test center
    for i in range(ndim):
        assert np.mean(surf[:, i]) == center[i]


@pytest.mark.parametrize(
    "targets, sources, source_densities, expected",
    [
        # 1. Single source, single target, unit density
        (
            np.array([[0, 0, 0]]),
            np.array([[0, 0, 1]]),
            np.array([1]),
            np.array([1/(4*np.pi)])
        )
    ]
)
def test_p2p(targets, sources, source_densities, expected):
    """
    Test with single-layer Laplace kernel.
    Strategy: test whether equivalent density and direct evaluation are
    approximately equivalent in the far field.
    """

    result = p2p(laplace, targets, sources, source_densities).density

    assert result == expected


def test_p2m(n_level_octree, order):
    """Test with single-layer Laplace kernel"""

    octree = n_level_octree(maximum_level=1)
    sources = octree.sources
    source_densities = np.ones(shape=(len(sources)))

    result = p2m(
        kernel_function=laplace,
        order=order,
        center=octree.center,
        radius=octree.radius,
        maximum_level=1,
        leaf_sources=octree.sources
        )

    distant_point = np.array([[10, 0, 0]])

    direct = p2p(laplace, distant_point, sources, source_densities)
    equivalent = p2p(laplace, distant_point, result.surface, result.density)

    for i in range(len(equivalent.density)):
        a = equivalent.density[i]; b = direct.density[i]
        assert np.isclose(a, b, rtol=1e-1)

    # Test that the surfaces are not equal, as expected
    assert ~np.array_equal(equivalent.surface, direct.surface)


def test_m2m(n_level_octree, order):
    """
    Test with single-layer Laplace kernel.
    Strategy: Test whether child equivalent density and parent equivalent
    density are approximately equivalent to each other in the far field.
    """

    npoints = 6*(order-1)**2 + 2
    octree = n_level_octree(maximum_level=1)
    parent_key = 0
    child_key = 1

    x0 = octree.center; r0 = octree.radius
    parent_center = hilbert.get_center_from_key(parent_key, x0, r0)
    child_center = hilbert.get_center_from_key(child_key, x0, r0)

    parent_level = hilbert.get_level(parent_key)
    child_level = hilbert.get_level(child_key)

    child_equivalent_density = np.ones(shape=(npoints))
    child_equivalent_surface = surface(
        order=order,
        radius=r0,
        level=child_level,
        center=child_center,
        alpha=1.05
    )

    result = m2m(
        kernel_function=laplace,
        order=order,
        parent_center=parent_center,
        child_center=child_center,
        radius=octree.radius,
        child_level=child_level,
        parent_level=parent_level,
        child_equivalent_density=child_equivalent_density
    )

    parent_equivalent_surface, parent_equivalent_density = result.surface, result.density

    distant_point = np.array([[1000, 0, 0]])

    child_result = p2p(
        laplace, distant_point, child_equivalent_surface, child_equivalent_density
    )

    parent_result = p2p(
        laplace, distant_point, parent_equivalent_surface, parent_equivalent_density
    )

    assert np.isclose(child_result.density[0], parent_result.density[0], atol=1e-3)

    # Test that the surfaces are not equal, as expected
    assert ~np.array_equal(parent_result.surface, child_result.surface)


def test_m2l(order):
    """
    Test with single-layer Laplace kernel
    Strategy: Create two boxes in the far field of each other, translate the
    multipole expansion from one box (the source) to the other (the target).
    Measure for the equivalence of their potentials in the far field of both.
    """

    # Number of points discretising surface of box of a given order
    npoints = 6*(order-1)**2 + 2

    # Radius of root node
    radius=1

    # (Octree) Level of boxes
    tgt_level = src_level = 5

    # Ensure that the centre of both boxes are far enough from each other to be
    # well separated at the specified level
    src_center = np.array([0, 0, 0])
    tgt_center = np.array([1, 0, 0])

    # Set unit densities as multipole expansion terms for source box
    src_equivalent_density = np.ones(shape=(npoints))

    src_equivalent_surface = surface(
        order=order,
        radius=radius,
        level=src_level,
        center=src_center,
        alpha=1.05
    )

    result = m2l(
        kernel_function=laplace,
        order=order,
        radius=radius,
        source_center=src_center,
        source_level=src_level,
        target_center=tgt_center,
        target_level=tgt_level,
        source_equivalent_density=src_equivalent_density
    )

    tgt_equivalent_surface, tgt_equivalent_density = result.surface, result.density

    local_point = np.array([[1, 0, 0]])

    tgt_result = p2p(
        kernel_function=laplace,
        targets=local_point,
        sources=tgt_equivalent_surface,
        source_densities=tgt_equivalent_density
    )

    src_result = p2p(
        kernel_function=laplace,
        targets=local_point,
        sources=src_equivalent_surface,
        source_densities=src_equivalent_density
    )

    assert np.isclose(tgt_result.density, src_result.density, rtol=1e-1)

    # Check that surfaces are not equivalent, as expected
    assert ~np.array_equal(tgt_result.surface, src_result.surface)


def test_l2l(n_level_octree, order):
    """
    Test with single-layer Laplace kernel
    Strategy: Test whether child equivalent density is matched to it's parents
    in the far field of both.
    """
    order=2
    npoints = 6*(order-1)**2 + 2

    octree = n_level_octree(maximum_level=1)
    parent_key = 0
    child_key = 1

    x0 = octree.center; r0 = octree.radius

    parent_center = hilbert.get_center_from_key(parent_key, x0, r0)
    child_center = hilbert.get_center_from_key(child_key, x0, r0)

    parent_level = hilbert.get_level(parent_key)
    child_level = hilbert.get_level(child_key)

    parent_equivalent_density = np.ones(shape=(npoints))
    parent_equivalent_surface = surface(
        order=order,
        radius=r0,
        level=parent_level,
        center=parent_center,
        alpha=2.95
    )

    result = l2l(
        kernel_function=laplace,
        order=order,
        radius=r0,
        parent_center=parent_center,
        parent_level=parent_level,
        child_center=child_center,
        child_level=child_level,
        parent_equivalent_density=parent_equivalent_density
    )

    child_equivalent_surface, child_equivalent_density = result.surface, result.density

    local_point = np.array([list(child_center)])

    child_result = p2p(
        laplace, local_point, child_equivalent_surface, child_equivalent_density
    )

    parent_result = p2p(
        laplace, local_point, parent_equivalent_surface, parent_equivalent_density
    )

    # Check equivalence of densities in far field
    assert np.isclose(child_result.density[0], parent_result.density[0], rtol=1e-1)

    # Test that the surfaces are not equal, as expected
    assert ~np.array_equal(parent_result.surface, child_result.surface)


# def test_upward_pass(order, n_level_octree):
#     octree = n_level_octree(3)
#     fmm = Fmm(octree, order, laplace)

#     fmm.upward_pass()
#     x0 = octree.center; r0 = octree.radius

#     equivalent_surface = surface(
#         order=order,
#         radius=r0,
#         level=0,
#         center=x0,
#         alpha=1.05
#     )

#     multipole_expansion = fmm._source_data[0]

#     distant_point = np.array([[1e4, 0, 0]])

#     multipole_result = p2p(
#         kernel_function=laplace,
#         targets=distant_point,
#         sources=equivalent_surface,
#         source_densities=multipole_expansion.expansion
#     )

#     direct_result = p2p(
#         kernel_function=laplace,
#         targets=distant_point,
#         sources=octree.sources,
#         source_densities=np.ones(len(octree.sources))
#     )


#     assert np.isclose(direct_result.density, multipole_result.density, rtol=1e-1)


def test_fmm(order, n_level_octree):
    octree = n_level_octree(1)
    fmm = Fmm(octree, order, laplace)

    fmm.upward_pass()
    fmm.downward_pass()

    print(fmm._result_data)

    unit_sources = np.ones(len(octree.sources))
    direct = p2p(
        laplace, octree.targets, octree.sources, unit_sources
    )
    print(f"direct: {direct.density}")
    assert False