"""
Test the fmm module.

Configured with test data 'test.hdf5' and a test config, 'test_config.json',
bundled with the package.
"""
import numpy as np

import adaptoctree.morton as morton

from fmm.fmm import Fmm
import fmm.surface as surface
from fmm.kernel import KERNELS


def test_upward_pass():
    """
    Test that multipole expansion of root node is the same as a direct
        computation for a set of source points, at a target point located
        at a distance.
    """
    experiment = Fmm('test_config')
    experiment.upward_pass()


    for key in experiment.complete[experiment.complete_levels == experiment.depth]:

        center = morton.find_physical_center_from_key(key, experiment.x0, experiment.r0)
        radius = experiment.x0 / (1 << experiment.depth)

        upward_equivalent_surface = surface.scale_surface(
            surf=experiment.equivalent_surface,
            radius=experiment.r0,
            level=experiment.depth,
            center=center,
            alpha=experiment.alpha_inner
        )

        distant_point = center+radius*3

        kernel = experiment.config['kernel']
        p2p = KERNELS[kernel]['p2p']

        leaf_idx = experiment.key_to_leaf_index[key]
        global_idx = experiment.key_to_index[key]

        lidx = global_idx*experiment.nequivalent_points
        ridx = lidx+experiment.nequivalent_points

        source_coordinates = experiment.sources[experiment.source_index_pointer[leaf_idx]:experiment.source_index_pointer[leaf_idx+1]]
        source_densities = experiment.source_densities[experiment.source_index_pointer[leaf_idx]:experiment.source_index_pointer[leaf_idx+1]]

        direct = p2p(
            sources=source_coordinates,
            targets=distant_point,
            source_densities=source_densities
        )

        equivalent = p2p(
            sources=upward_equivalent_surface,
            targets=distant_point,
            source_densities=experiment.multipole_expansions[lidx:ridx]
        )

        assert np.allclose(direct, equivalent, atol=1e-2, rtol=0)


def test_m2l():
    """
    Test that the local expansions of a given target node correspond to the
        multipole expansions of source nodes in its V list at a local point
        within the target node.
    """
    experiment = Fmm('test_config')

    experiment.run()

    p2p = experiment.p2p_function

    level_2_idxs = experiment.complete_levels == 2

    for i, key in enumerate(experiment.complete[level_2_idxs]):
        idx = experiment.key_to_index[key]
        target_lidx = idx*experiment.nequivalent_points
        target_ridx = target_lidx+experiment.nequivalent_points
        v_list = experiment.v_lists[idx]
        v_list = v_list[v_list != -1]


        level = np.int32(morton.find_level(key))
        center = morton.find_physical_center_from_key(key, experiment.x0, experiment.r0).astype(np.float32)

        local_expansion = experiment.local_expansions[target_lidx:target_ridx]

        downward_check_surface = surface.scale_surface(
            surf=experiment.equivalent_surface,
            radius=experiment.r0,
            level=level,
            center=center,
            alpha=experiment.alpha_inner
        )

        downward_equivalent_surface = surface.scale_surface(
            surf=experiment.equivalent_surface,
            radius=experiment.r0,
            level=level,
            center=center,
            alpha=experiment.alpha_outer
        )

        equivalent = p2p(
            sources=downward_equivalent_surface,
            targets=downward_check_surface,
            source_densities=local_expansion
        )

        direct = np.zeros_like(equivalent)

        for source in v_list:

            source_level = np.int32(morton.find_level(source))
            source_center = morton.find_physical_center_from_key(source, experiment.x0, experiment.r0).astype(np.float32)

            upward_equivalent_surface = surface.scale_surface(
                surf=experiment.equivalent_surface,
                radius=experiment.r0,
                level=source_level,
                center=source_center,
                alpha=experiment.alpha_inner
            )

            source_idx = experiment.key_to_index[source]
            source_lidx = source_idx*experiment.nequivalent_points
            source_ridx = source_lidx+experiment.nequivalent_points

            tmp = p2p(
                sources=upward_equivalent_surface,
                targets=downward_check_surface,
                source_densities=experiment.multipole_expansions[source_lidx:source_ridx]
            )

            direct += tmp

        assert np.allclose(direct, equivalent, rtol=0.01, atol=0)


def test_fmm():
    """
    End To End Fmm Test.
    """
    experiment = Fmm('test_config')

    experiment.run()

    kernel = experiment.config['kernel']
    p2p = KERNELS[kernel]['p2p']

    direct = p2p(experiment.sources, experiment.targets, experiment.source_densities)
    equivalent = experiment.target_potentials

    np.allclose(direct, equivalent, rtol=1e-2, atol=0)