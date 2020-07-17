"""
Test the FMM
"""
from functools import partial

import numpy as np
import pytest

from fmm.fmm import Fmm, Laplace, surface, p2p, p2m, m2m, m2l, l2l
from fmm.octree import Octree
import fmm.hilbert as hilbert


@pytest.mark.parametrize(
    "level",
    [
        1, 2, 3
    ]
)
def test_upward_pass(level, order, n_level_octree):
    """Test whether Multipole expansion is translated to root node.
    """
    octree = n_level_octree(level)
    fmm = Fmm()

    fmm.upward_pass()
    x0 = octree.center
    r0 = octree.radius

    equivalent_surface = surface(
        order=order,
        radius=r0,
        level=0,
        center=x0,
        alpha=1.05
    )

    multipole_expansion = fmm.source_data[0]

    distant_point = np.array([[1e4, 0, 0]])

    multipole_result = p2p(
        kernel_function=laplace,
        targets=distant_point,
        sources=equivalent_surface,
        source_densities=multipole_expansion.expansion
    )

    direct_result = p2p(
        kernel_function=laplace,
        targets=distant_point,
        sources=octree.sources,
        source_densities=np.ones(len(octree.sources))
    )

    print(direct_result)

    assert np.isclose(direct_result.density, multipole_result.density, rtol=1.5e-1)


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