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
    e = Fmm('test_config')
    e.upward_pass()

    # Test leaf expansions
    for key in e.complete[e.complete_levels == e.depth]:

        center = morton.find_physical_center_from_key(key, e.x0, e.r0)
        radius = e.r0 / (1 << e.depth)

        upward_equivalent_surface = surface.scale_surface(
            surf=e.equivalent_surface,
            radius=e.r0,
            level=e.depth,
            center=center,
            alpha=e.alpha_inner
        )

        distant_point = center+(radius*3)

        kernel = e.config['kernel']
        p2p = KERNELS[kernel]['p2p']

        leaf_idx = e.key_to_leaf_index[key]
        global_idx = e.key_to_index[key]

        lidx = global_idx*e.nequivalent_points
        ridx = lidx+e.nequivalent_points

        source_coordinates = e.sources[e.source_index_pointer[leaf_idx]:e.source_index_pointer[leaf_idx+1]]
        source_densities = e.source_densities[e.source_index_pointer[leaf_idx]:e.source_index_pointer[leaf_idx+1]]

        direct = p2p(
            sources=source_coordinates,
            targets=distant_point,
            source_densities=source_densities
        )

        equivalent = p2p(
            sources=upward_equivalent_surface,
            targets=distant_point,
            source_densities=e.multipole_expansions[lidx:ridx]
        )

        assert np.allclose(direct, equivalent, atol=1e-2, rtol=0)

    # Test root expansion
    upward_equivalent_surface = surface.scale_surface(
        surf=e.equivalent_surface,
        radius=e.r0,
        level=0,
        center=e.x0,
        alpha=e.alpha_inner
    )

    root = 0
    global_idx = e.key_to_index[root]

    lidx = global_idx*e.nequivalent_points
    ridx = lidx+e.nequivalent_points

    distant_point = e.x0+(e.r0*3)

    direct = p2p(
        sources=e.sources,
        targets=distant_point,
        source_densities=e.source_densities,
    )

    equivalent = p2p(
        sources=upward_equivalent_surface,
        targets=distant_point,
        source_densities=e.multipole_expansions[lidx:ridx]
    )


def test_m2l():
    """
    Test that the local expansions of a given target node correspond to the
        multipole expansions of source nodes in its V list at a local point
        within the target node.
    """
    e = Fmm('test_config')

    e.run()

    p2p = e.p2p_function

    level_2_idxs = e.complete_levels == 2

    for i, key in enumerate(e.complete[level_2_idxs]):
        idx = e.key_to_index[key]
        target_lidx = idx*e.nequivalent_points
        target_ridx = target_lidx+e.nequivalent_points
        v_list = e.v_lists[idx]
        v_list = v_list[v_list != -1]


        level = np.int32(morton.find_level(key))
        center = morton.find_physical_center_from_key(key, e.x0, e.r0)

        local_expansion = e.local_expansions[target_lidx:target_ridx]

        downward_check_surface = surface.scale_surface(
            surf=e.equivalent_surface,
            radius=e.r0,
            level=level,
            center=center,
            alpha=e.alpha_inner
        )

        downward_equivalent_surface = surface.scale_surface(
            surf=e.equivalent_surface,
            radius=e.r0,
            level=level,
            center=center,
            alpha=e.alpha_outer
        )

        equivalent = p2p(
            sources=downward_equivalent_surface,
            targets=downward_check_surface,
            source_densities=local_expansion
        )

        direct = np.zeros_like(equivalent)

        for source in v_list:

            source_level = np.int32(morton.find_level(source))
            source_center = morton.find_physical_center_from_key(source, e.x0, e.r0)

            upward_equivalent_surface = surface.scale_surface(
                surf=e.equivalent_surface,
                radius=e.r0,
                level=source_level,
                center=source_center,
                alpha=e.alpha_inner
            )

            source_idx = e.key_to_index[source]
            source_lidx = source_idx*e.nequivalent_points
            source_ridx = source_lidx+e.nequivalent_points

            tmp = p2p(
                sources=upward_equivalent_surface,
                targets=downward_check_surface,
                source_densities=e.multipole_expansions[source_lidx:source_ridx]
            )

            direct += tmp

        assert np.allclose(direct, equivalent, rtol=0.01, atol=0)


def test_fmm():
    """
    End To End Fmm Test.
    """
    e = Fmm('test_config')

    e.run()

    kernel = e.config['kernel']

    # Test potential
    p2p = KERNELS[kernel]['p2p']

    direct = p2p(e.sources, e.targets, e.source_densities)
    equivalent = e.target_potentials
    accuracy = -np.log10(np.mean(abs(direct-equivalent[:,0])/direct))
    assert accuracy > 5

    # Test gradients
    grad = KERNELS[kernel]['gradient']
    direct = grad(e.sources, e.targets, e.source_densities)
    accuracy = -np.log10(abs(np.mean((direct[:,0]-equivalent[:, 1])/direct[:, 0])))
    assert accuracy > 3

    accuracy = -np.log10(abs(np.mean((direct[:,1]-equivalent[:, 2])/direct[:, 1])))
    assert accuracy > 3

    accuracy = -np.log10(abs(np.mean((direct[:,2]-equivalent[:, 3])/direct[:, 2])))
    assert accuracy > 3
