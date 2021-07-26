"""
Test the fmm module.

Configured with test data 'test.hdf5' and a test config, 'test_config.json',
bundled with the package.
"""
import numpy as np

import adaptoctree.morton as morton

from fmm.fmm import Fmm
from fmm.kernel import KERNELS


def test_upward_pass():
    """
    Test that multipole expansions are translated correctly at leaf nodes, and
    again at root node.
    """
    e = Fmm('test_config')
    e.upward_pass()
    kernel = e.config['kernel']
    p2p = KERNELS[kernel]['p2p']

    # Test leaf expansions
    for key in e.complete[e.complete_levels == e.depth]:

        center = morton.find_physical_center_from_key(key, e.x0, e.r0)
        radius = e.r0 / (1 << e.depth)

        upward_equivalent_surface = e.upward_equivalent_surfaces[e.depth].copy()
        upward_equivalent_surface += center

        distant_point = center+(radius*3)

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
    root = 0
    upward_equivalent_surface = e.upward_equivalent_surfaces[0].copy()
    distant_point = e.x0+(e.r0*3)

    global_idx = e.key_to_index[root]

    lidx = global_idx*e.nequivalent_points
    ridx = lidx+e.nequivalent_points

    direct = p2p(
        sources=e.sources,
        targets=distant_point,
        source_densities=e.source_densities
    )

    equivalent = p2p(
        sources=upward_equivalent_surface,
        targets=distant_point,
        source_densities=e.multipole_expansions[lidx:ridx]
    )

    assert np.allclose(direct, equivalent, atol=1e-2, rtol=0)


# def test_m2l():
#     """
#     Test that the local expansions of a given target node correspond to the
#         multipole expansions of source nodes in its V list at a local point
#         within the target node.
#     """
#     experiment = Fmm('test_config')

#     experiment.run()

#     p2p = experiment.p2p_function

#     level_2_idxs = experiment.complete_levels == 2

#     for i, key in enumerate(experiment.complete[level_2_idxs]):
#         idx = experiment.key_to_index[key]
#         target_lidx = idx*experiment.nequivalent_points
#         target_ridx = target_lidx+experiment.nequivalent_points
#         v_list = experiment.v_lists[idx]
#         v_list = v_list[v_list != -1]


#         level = np.int32(morton.find_level(key))
#         center = morton.find_physical_center_from_key(key, experiment.x0, experiment.r0).astype(np.float32)

#         local_expansion = experiment.local_expansions[target_lidx:target_ridx]

#         downward_check_surface = surface.scale_surface(
#             surf=experiment.equivalent_surface,
#             radius=experiment.r0,
#             level=level,
#             center=center,
#             alpha=experiment.alpha_inner
#         )

#         downward_equivalent_surface = surface.scale_surface(
#             surf=experiment.equivalent_surface,
#             radius=experiment.r0,
#             level=level,
#             center=center,
#             alpha=experiment.alpha_outer
#         )

#         equivalent = p2p(
#             sources=downward_equivalent_surface,
#             targets=downward_check_surface,
#             source_densities=local_expansion
#         )

#         direct = np.zeros_like(equivalent)

#         for source in v_list:

#             source_level = np.int32(morton.find_level(source))
#             source_center = morton.find_physical_center_from_key(source, experiment.x0, experiment.r0).astype(np.float32)

#             upward_equivalent_surface = surface.scale_surface(
#                 surf=experiment.equivalent_surface,
#                 radius=experiment.r0,
#                 level=source_level,
#                 center=source_center,
#                 alpha=experiment.alpha_inner
#             )

#             source_idx = experiment.key_to_index[source]
#             source_lidx = source_idx*experiment.nequivalent_points
#             source_ridx = source_lidx+experiment.nequivalent_points

#             tmp = p2p(
#                 sources=upward_equivalent_surface,
#                 targets=downward_check_surface,
#                 source_densities=experiment.multipole_expansions[source_lidx:source_ridx]
#             )

#             direct += tmp

#         assert np.allclose(direct, equivalent, rtol=0.01, atol=0)


# def test_fmm():
#     """
#     End To End Fmm Test.
#     """
#     experiment = Fmm('test_config')

#     experiment.run()

#     kernel = experiment.config['kernel']

#     # Test potential
#     p2p = KERNELS[kernel]['p2p']

#     direct = p2p(experiment.sources, experiment.targets, experiment.source_densities)
#     equivalent = experiment.target_potentials
#     accuracy = -np.log10(np.mean(abs(direct-equivalent[:,0])/direct))
#     assert accuracy > 5

#     # Test gradients
#     grad = KERNELS[kernel]['gradient']
#     direct = grad(experiment.sources, experiment.targets, experiment.source_densities)
#     accuracy = -np.log10(np.mean(abs(direct[:,0]-equivalent[:, 1])/direct[:, 0]))
#     assert accuracy > 3

#     accuracy = -np.log10(np.mean(abs(direct[:,1]-equivalent[:, 2])/direct[:, 1]))
#     assert accuracy > 3

#     accuracy = -np.log10(np.mean(abs(direct[:,2]-equivalent[:, 3])/direct[:, 2]))
#     assert accuracy > 3
