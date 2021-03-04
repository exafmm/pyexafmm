"""
Test for kernel module.
"""
import numpy as np
import pytest

import fmm.kernel as kernel


@pytest.mark.parametrize(
    "level, expected",
    [
        (0, 1),
        (1, 0.5),
        (2, 0.25)
    ]
)
def test_laplace_scale(level, expected):
    assert kernel.laplace_scale(level) == expected


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (np.array([1, 0, 0]), np.array([0, 0, 0]), 1./(4*np.pi))
    ]
)
def test_laplace_cpu(x, y, expected):
    assert kernel.laplace_cpu(x, y) == expected


@pytest.mark.parametrize(
    "sources, targets, source_densities",
    [
        (
            np.random.rand(10, 3),
            np.random.rand(10, 3),
            np.ones(10),

        )
    ]
)
def test_laplace_p2p(sources, targets, source_densities):

    result = kernel.laplace_p2p(
        sources,
        targets,
        source_densities
    )

    # Check each target
    for i in range(len(targets)):
        target = targets[i]
        target_potential = 0
        for j in range(len(sources)):
            source = sources[j]
            source_density = source_densities[j]
            target_potential += kernel.laplace_cpu(target, source)*source_density

        assert result[i] == target_potential


@pytest.mark.parametrize(
    "sources, targets",
    [
        (
            np.random.rand(12, 3),
            np.random.rand(34, 3),
        )
    ]
)
def test_laplace_gram_matrix(sources, targets):

    K = kernel.laplace_gram_matrix(sources, targets)

    # Check that it's the correct shape
    assert len(K[0]) == 12
    assert len(K) == 34
