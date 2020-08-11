"""
Test the FMM
"""
from unittest.mock import MagicMock

import collections
import os
import pathlib

import numpy as np
import pytest

from fmm.fmm import Fmm
import fmm.operator as operator
import fmm.hilbert as hilbert
import fmm.kernel as kernel
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILEPATH = HERE.parent.parent / "test_config.json"
CONFIG = data.load_json(CONFIG_FILEPATH)

ORDER = CONFIG['order']

OPERATOR_DIRPATH = HERE.parent.parent / CONFIG['operator_dirname']
SCRIPT_DIRPATH = HERE.parent.parent / 'scripts'

KERNEL_FUNCTION = kernel.KERNELS['laplace']()


@pytest.fixture
def l2l():
    return data.load_hdf5_to_array('l2l', 'l2l', OPERATOR_DIRPATH)


@pytest.fixture
def fmm():
    return Fmm(config_filename='test_config.json')


def test_upward_pass(fmm):
    """
    Compare direct computations against multipole expansions
    """
    fmm.upward_pass()

    surface = operator.compute_surface(order=CONFIG['order'])

    current_level = fmm.maximum_level

    sources = fmm.sources

    distant_point = np.array([[1e3, 0, 0]])

    # Post order traversal
    while current_level > -1:

        source_node_keys = fmm.octree.non_empty_source_nodes_by_level[current_level]

        # Sources in terms of hilbert keys at the current level.
        sources_by_key = hilbert.get_keys_from_points(
            points=sources,
            level=current_level,
            x0=fmm.octree.center,
            r0=fmm.octree.radius
        )

        print(f'level {current_level}')

        for source_node_key in source_node_keys:

            source_node_expansion = fmm.source_data[source_node_key].expansion

            source_node_center = hilbert.get_center_from_key(
                source_node_key, fmm.octree.center, fmm.octree.radius
            )

            source_node_surface = operator.scale_surface(
                surface=surface,
                radius=fmm.octree.radius,
                level=current_level,
                center=source_node_center,
                alpha=2.95
            )

            source_result = operator.p2p(
                kernel_function=fmm.kernel_function,
                targets=distant_point,
                sources=source_node_surface,
                source_densities=source_node_expansion
            )

            # Pick out sources within this target box.

            sources_in_box = []
            for source_idx, source in enumerate(sources_by_key):
                if source == source_node_key:
                    sources_in_box.append(fmm.sources[source_idx])

            sources_in_box = np.array(sources_in_box)

            direct_result = operator.p2p(
                kernel_function=fmm.kernel_function,
                targets=distant_point.reshape(1, 3),
                sources=sources_in_box,
                source_densities=np.ones(len(sources_in_box))
            )

            assert np.allclose(source_result.density, direct_result.density, atol=0, rtol=0.01)

        current_level -= 1
