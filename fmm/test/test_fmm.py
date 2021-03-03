"""
Test the FMM
"""
import numpy as np

import adaptoctree.morton as morton

from fmm.fmm import Fmm
import fmm.operator as operator
from fmm.kernel import KERNELS


RTOL = 1e-1


def test_upward_pass():

    fmm = Fmm('test_config')
    fmm.upward_pass()

    upward_equivalent_surface = operator.scale_surface(
        surface=fmm.equivalent_surface,
        radius=fmm.r0,
        level=0,
        center=fmm.x0,
        alpha=fmm.config['alpha_outer']
    )

    distant_point = np.array([[1e4, 0, 0]])

    kernel = fmm.config['kernel']
    p2p = KERNELS[kernel]['p2p']

    direct = p2p(
        sources=fmm.sources,
        targets=distant_point,
        source_densities=fmm.source_densities
    )

    equivalent = p2p(
        sources=upward_equivalent_surface,
        targets=distant_point,
        source_densities=fmm.multipole_expansions[0]
    )

    assert np.allclose(direct, equivalent, rtol=RTOL)


def test_downward_pass():

    fmm = Fmm('test_config')

    fmm.upward_pass()
    fmm.downward_pass()

    kernel = fmm.config['kernel']
    p2p = KERNELS[kernel]['p2p']

    key = 1114114
    idx = np.where(fmm.complete == key)
    v_list = fmm.v_lists[idx]
    v_list = v_list[v_list != -1]

    target_idxs = key == fmm.targets_to_keys
    target_coordinates = fmm.targets[target_idxs]
    level = morton.find_level(key)
    center = morton.find_physical_center_from_key(key, fmm.x0, fmm.r0)

    local_expansion = fmm.local_expansions[key]

    downward_equivalent_surface = operator.scale_surface(
        surface=fmm.equivalent_surface,
        radius=fmm.r0,
        level=level,
        center=center,
        alpha=fmm.config['alpha_outer']
    )

    equivalent = p2p(
        sources=downward_equivalent_surface,
        targets=target_coordinates,
        source_densities=local_expansion
    )

    direct = np.zeros_like(equivalent)

    for source in v_list:

        source_level = morton.find_level(source)
        source_center = morton.find_physical_center_from_key(source, fmm.x0, fmm.r0)

        upward_equivalent_surface = operator.scale_surface(
            surface=fmm.equivalent_surface,
            radius=fmm.r0,
            level=source_level,
            center=source_center,
            alpha=fmm.config['alpha_inner']
        )

        tmp = p2p(
            sources=upward_equivalent_surface,
            targets=target_coordinates,
            source_densities=fmm.multipole_expansions[source]
        )

        direct += tmp

    assert np.allclose(direct, equivalent, rtol=RTOL)


def test_fmm():
    """
    End To End Fmm Test
    """
    fmm = Fmm('test_config')
    fmm.upward_pass()
    fmm.downward_pass()

    kernel = fmm.config['kernel']
    p2p = KERNELS[kernel]['p2p']

    direct = p2p(
        targets=fmm.targets,
        sources=fmm.sources,
        source_densities=fmm.source_densities
    )

    assert np.allclose(direct, fmm.target_potentials, rtol=RTOL)
