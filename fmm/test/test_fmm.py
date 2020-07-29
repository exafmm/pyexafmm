"""
Test the FMM
"""
import collections
import os
import pathlib
import subprocess

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
    Starting at the leaf level, test whether the multipole expansion of the a
        set of nodes with a shared parent match those generated by the multipole
        of their parent.
    """
    fmm.upward_pass()

    surface = data.load_hdf5_to_array('surface', 'surface', OPERATOR_DIRPATH)

    distant_point = np.array([[1e3, 0, 0]])

    leaf_level = fmm.maximum_level
    current_level = leaf_level

    # Group nodes by parent
    nodes_by_parent = collections.defaultdict(list)

    while current_level > 0:

        for key, node in fmm.source_data.items():
            if hilbert.get_level(key) == current_level:
                nodes_by_parent[hilbert.get_parent(key)].append(node)

        current_level -= 1

    # For each node, compare multipole expansion result at distant target with
    # result from the parent node's expansion, starting at the leaf level.
    current_level = leaf_level

    while current_level > 0:

        parent_keys = [
            hilbert.get_parent(key) for key, node in fmm.source_data.items()
            if hilbert.get_level(key) == current_level
        ]

        # Filter for unique parents
        parent_keys = list(set(parent_keys))

        for parent_key in parent_keys:
            # Get parent multipole expansion
            parent_node = fmm.source_data[parent_key]

            parent_expansion = parent_node.expansion
            parent_center = hilbert.get_center_from_key(
                parent_key, fmm.octree.center, fmm.octree.radius
            )

            parent_equivalent_surface = operator.scale_surface(
                surface=surface,
                radius=fmm.octree.radius,
                level=current_level-1,
                center=parent_center,
                alpha=1.05
            )

            parent_potential = operator.p2p(
                kernel_function=KERNEL_FUNCTION,
                targets=distant_point,
                sources=parent_equivalent_surface,
                source_densities=parent_expansion
            ).density

            # Compare parent expansion to those of it's children
            child_potential = 0
            for child_key in hilbert.get_children(parent_key):
                if child_key in fmm.source_data.keys():

                    child_expansion = fmm.source_data[child_key].expansion
                    child_center = hilbert.get_center_from_key(
                        child_key, fmm.octree.center, fmm.octree.radius
                    )

                    child_equivalent_surface = operator.scale_surface(
                        surface,
                        fmm.octree.radius,
                        current_level,
                        child_center,
                        1.05
                    )

                    child_potential += operator.p2p(
                        kernel_function=KERNEL_FUNCTION,
                        targets=distant_point,
                        sources=child_equivalent_surface,
                        source_densities=child_expansion
                    ).density

            assert np.isclose(child_potential, parent_potential, rtol=0.1)

        current_level -= 1

    # Test source to mulitpole at leaf level
    leaf_nodes = {
        key: node for key, node in fmm.source_data.items()
        if hilbert.get_level(key) == leaf_level
    }

    for key, node in leaf_nodes.items():

        leaf_center = hilbert.get_center_from_key(
            key, fmm.octree.center, fmm.octree.radius
        )

        leaf_index = fmm.octree.source_node_to_index[key]

        leaf_source_indices = fmm.octree.sources_by_leafs[
                fmm.octree.source_index_ptr[leaf_index]:
                fmm.octree.source_index_ptr[leaf_index + 1]
            ]

        leaf_sources = fmm.octree.sources[leaf_source_indices]
        leaf_source_densities = fmm.octree.source_densities[leaf_source_indices]

        leaf_equivalent_surface = operator.scale_surface(
            surface=surface,
            radius=fmm.octree.radius,
            level=leaf_level,
            center=leaf_center,
            alpha=1.05
        )

        # Directly compute potential from sources
        source_result = 0
        for source in leaf_sources:
            source_result += operator.p2p(
                kernel_function=KERNEL_FUNCTION,
                targets=distant_point,
                sources=source.reshape(1, 3),
                source_densities=leaf_source_densities
            ).density

        # Compare with result from leaf multipole expansion
        leaf_result = operator.p2p(
            kernel_function=KERNEL_FUNCTION,
            targets=distant_point,
            sources=leaf_equivalent_surface,
            source_densities=node.expansion
        ).density

        assert np.isclose(leaf_result, source_result, rtol=0.1)


def test_downward_pass(fmm, l2l):
    fmm.upward_pass()
    fmm.downward_pass()

    surface = data.load_hdf5_to_array('surface', 'surface', OPERATOR_DIRPATH)

    # Group nodes by parent
    nodes_by_parent = collections.defaultdict(list)
    current_level = fmm.maximum_level

    while current_level > 0:

        for key, node in fmm.source_data.items():
            if hilbert.get_level(key) == current_level:
                nodes_by_parent[hilbert.get_parent(key)].append(node)

        current_level -= 1

    # Post-order tree traversal, checking local expansion of parent against that
    # of their children
    current_level = 2

    while current_level < fmm.maximum_level:

        target_keys = fmm.octree._target_nodes_by_level[current_level]

        for target_key in target_keys:

            target_expansion = fmm.target_data[target_key].expansion
            target_center = hilbert.get_center_from_key(
                target_key, fmm.octree.center, fmm.octree.radius
            )
            target_surface = operator.scale_surface(
                surface=surface,
                radius=fmm.octree.radius,
                level=current_level,
                center=target_center,
                alpha=2.95
            )

            children = hilbert.get_children(target_key)

            for child in children:

                if child in fmm.target_data.keys():

                    child_expansion = fmm.target_data[child].expansion
                    child_center = hilbert.get_center_from_key(
                        child, fmm.octree.center, fmm.octree.radius
                    )
                    child_surface = operator.scale_surface(
                        surface=surface,
                        radius=fmm.octree.radius,
                        level=current_level+1,
                        center=child_center,
                        alpha=2.95
                    )

                    # Evaluate expansions at child center
                    parent_result = operator.p2p(
                        kernel_function=fmm.kernel_function,
                        targets=child_center.reshape(1, 3),
                        sources=target_surface,
                        source_densities=target_expansion
                    )

                    child_result = operator.p2p(
                        kernel_function=fmm.kernel_function,
                        targets=child_center.reshape(1, 3),
                        sources=child_surface,
                        source_densities=child_expansion
                    )

                    assert np.isclose(parent_result.density, child_result.density, rtol=0.10)

        current_level += 1


def test_fmm(fmm):
    fmm.upward_pass()
    fmm.downward_pass()

    fmm_results = np.array([
        res.density[0] for res in fmm.result_data
    ])

    direct = operator.p2p(
        kernel_function=KERNEL_FUNCTION,
        targets=fmm.octree.targets,
        sources=fmm.octree.sources,
        source_densities=fmm.octree.source_densities
    ).density

    error = abs(fmm_results) - abs(direct)

    percentage_error = 100*error/direct

    print("average percentage error", sum(percentage_error)/len(error))
    print(fmm_results[:10])
    print(direct[:10])

    assert False