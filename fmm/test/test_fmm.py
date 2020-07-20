"""
Test the FMM
"""
import numpy as np
import pytest

from fmm.fmm import Fmm
from fmm.operator import scale_surface, p2p


def test_upward_pass():
    fmm = Fmm(config_filename='test_config.json')
    fmm.upward_pass()

    root_equivalent_surface = scale_surface(
        fmm.surface, fmm.octree.radius, 0, fmm.octree.center, 1.05
    )

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

    # assert np.isclose(direct_fmm.density, direct_particles.density, rtol=0.1)


def test_downward_pass():

    fmm = Fmm(config_filename='test_config.json')

    fmm.upward_pass()
    fmm.downward_pass()

    print(fmm.result_data)

    assert False