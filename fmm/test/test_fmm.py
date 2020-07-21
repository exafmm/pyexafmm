"""
Test the FMM
"""
import numpy as np
import pytest

from fmm.fmm import Fmm
from fmm.operator import scale_surface, p2p


def test_upward_pass():
    fmm = Fmm(config_filename='config.json')
    fmm.upward_pass()

    root_equivalent_surface = scale_surface(
        fmm.surface, fmm.octree.radius, 0, fmm.octree.center, 1.05
    )

    distant_point = np.array([[1e3, 0, 0]])

    direct_fmm = p2p(
        fmm.kernel_function,
        distant_point,
        root_equivalent_surface,
        fmm.source_data[0].expansion
    )

    direct_particles = p2p(
        fmm.kernel_function,
        distant_point,
        fmm.sources,
        fmm.source_densities
    )

    assert np.isclose(direct_fmm.density, direct_particles.density, rtol=0.01)


def test_downward_pass():

    fmm = Fmm(config_filename='config.json')

    fmm.upward_pass()
    fmm.downward_pass()

    fmm_results = np.array([res.density[0] for res in fmm.result_data])

    direct = p2p(
        fmm.kernel_function,
        fmm.targets,
        fmm.sources,
        fmm.source_densities
    )


