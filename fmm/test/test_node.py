"""Tests for Node object"""
import numpy as np
import pytest

import fmm.node as node


@pytest.mark.parametrize(
    "key, ncoefficients, indices",
    [
        (0, 8, None)
    ]
)
def test_node_instantiation(key, ncoefficients, indices):
    obj = node.Node(key, ncoefficients, indices)

    assert obj.expansion.shape == (8,)
    if indices is None:
        assert isinstance(obj.indices, set)
