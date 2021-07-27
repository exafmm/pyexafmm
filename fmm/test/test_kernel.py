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
        (
            np.array([1, 0, 0], np.float32),
            np.array([0, 0, 0], np.float32),
            np.float32(1./(4*np.pi))
        )
    ]
)
def test_laplace_kernel_cpu(x, y, expected):
    dtype = x.dtype
    m_inv_4pi = dtype.type(kernel.M_INV_4PI)
    zero = dtype.type(0)
    result = kernel.laplace_kernel_cpu(x, y, m_inv_4pi, zero)
    assert np.isclose(result, expected)


@pytest.mark.parametrize(
    "sources, targets, source_densities",
    [
        (
            np.random.rand(10, 3).astype(np.float32),
            np.random.rand(10, 3).astype(np.float32),
            np.random.rand(10).astype(np.float32),
        ),
        (
            np.random.rand(10, 3).astype(np.float64),
            np.random.rand(10, 3).astype(np.float64),
            np.random.rand(10).astype(np.float64),
        ),
    ]
)
def test_laplace_p2p_serial(sources, targets, source_densities):

    result = kernel.laplace_p2p_serial(
        sources,
        targets,
        source_densities
    )

    dtype = sources.dtype
    m_inv_4pi = dtype.type(kernel.M_INV_4PI)
    zero = dtype.type(0)

    # Check each target
    for i in range(len(targets)):
        target = targets[i]
        target_potential = dtype.type(0)
        for j in range(len(sources)):
            source = sources[j]
            source_density = source_densities[j]
            target_potential += kernel.laplace_kernel_cpu(target, source, m_inv_4pi, zero)*source_density

        assert np.isclose(result[i], target_potential)

    # Check type is correctly casted
    assert isinstance(result[0], dtype.type)


@pytest.mark.parametrize(
    "sources, targets",
    [
        (
            np.random.rand(12, 3).astype(np.float32),
            np.random.rand(34, 3).astype(np.float32),
        ),
        (
            np.random.rand(12, 3).astype(np.float64),
            np.random.rand(34, 3).astype(np.float64),
        ),
    ]
)
def test_laplace_gram_matrix_serial(sources, targets):

    gram_matrix = kernel.laplace_gram_matrix_serial(sources, targets)

    # Check that it's the correct shape
    assert len(gram_matrix[0]) == 12
    assert len(gram_matrix) == 34

    # Check type is correctly casted
    dtype = sources.dtype
    assert isinstance(gram_matrix[0, 0], dtype.type)


@pytest.mark.parametrize(
    "x, y, c",
    [
        (
            np.array([1, 0, 0], np.float32),
            np.array([0, 0, 0], np.float32),
            0,
        )
    ]
)
def test_laplace_grad_cpu(x, y, c):
    dtype = x.dtype
    m_inv_4pi = dtype.type(kernel.M_INV_4PI)
    zero = dtype.type(0)
    result = kernel.laplace_grad_cpu(x, y, c, m_inv_4pi, zero)

    assert np.isclose(result, m_inv_4pi)


@pytest.mark.parametrize(
    "sources, targets, source_densities",
    [
        (
            np.random.rand(10, 3).astype(np.float32),
            np.random.rand(10, 3).astype(np.float32),
            np.random.rand(10).astype(np.float32),
        ),
        (
            np.random.rand(10, 3).astype(np.float64),
            np.random.rand(10, 3).astype(np.float64),
            np.random.rand(10).astype(np.float64),
        ),
    ]
)
def test_laplace_gradient(sources, targets, source_densities):

    result = kernel.laplace_gradient(
        sources,
        targets,
        source_densities
    )

    dtype = sources.dtype
    m_inv_4pi = dtype.type(kernel.M_INV_4PI)
    zero = dtype.type(0)

    target_gradients = np.zeros((10, 3), dtype=dtype.type)

    # Check each target
    for i in range(len(targets)):
        target = targets[i]
        for j in range(len(sources)):
            source = sources[j]
            source_density = source_densities[j]
            target_gradients[i][0] -= kernel.laplace_grad_cpu(target, source, 0, m_inv_4pi, zero)*source_density
            target_gradients[i][1] -= kernel.laplace_grad_cpu(target, source, 1, m_inv_4pi, zero)*source_density
            target_gradients[i][2] -= kernel.laplace_grad_cpu(target, source, 2, m_inv_4pi, zero)*source_density

    for i in range(len(targets)):
        assert np.allclose(result[i], target_gradients[i])

    # Check type is correctly casted
    assert isinstance(result[0, 0], dtype.type)
