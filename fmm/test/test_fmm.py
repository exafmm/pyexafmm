"""
Test the FMM
"""
import numpy as np
import pytest

from fmm.fmm import Fmm
from fmm.octree import Octree
from fmm.operator import scale_surface, p2p
import fmm.hilbert as hilbert


def test_upward_pass():
    fmm = Fmm(config_filename='test_config.json')
    fmm.upward_pass()

    root_equivalent_surface = scale_surface(
        fmm.surface, fmm.octree.radius, 0, fmm.octree.center, 1.05
    )

    # print(fmm.source_data[0])

    distant_point = np.array([[1e3, 0, 0]])

    direct_fmm = p2p(
        fmm.kernel_function,
        distant_point,
        root_equivalent_surface,
        fmm.source_densities
    )

    direct_particles = p2p(
        fmm.kernel_function,
        distant_point,
        fmm.sources,
        fmm.source_densities
    )
    print(direct_fmm, direct_particles)
    assert False



# def test_downward_pass(n_level_octree, order):

#     level = 5
#     octree = n_level_octree(level)

#     fmm = Fmm(octree, order, laplace)

#     fmm.upward_pass()
#     fmm.downward_pass()

#     direct_p2p = p2p(
#         kernel_function=laplace,
#         targets=octree.targets,
#         sources=octree.sources,
#         source_densities=np.ones(len(octree.sources))
#     ).density

#     fmm_p2p = np.array([res.density[0] for res in fmm.result_data])

#     print('fmm', fmm_p2p)
#     print('direct', direct_p2p)
#     for i in range(len(direct_p2p)):
#         assert np.isclose(direct_p2p[i], fmm_p2p[i], rtol=0.1)