"""
Test the FMM
"""
import numpy as np
import pytest

from fmm.fmm import Fmm
import fmm.operator as operator
from fmm.kernel import KERNELS


RTOL = 1e-1


@pytest.fixture
def fmm():
    return Fmm(config_filename='test_config')


# def test_upward_pass(fmm):

#     fmm.upward_pass()

#     upward_equivalent_surface = operator.scale_surface(
#         surface=fmm.equivalent_surface,
#         radius=fmm.r0,
#         level=0,
#         center=fmm.x0,
#         alpha=fmm.config['alpha_outer']
#     )

#     distant_point = np.array([[1e4, 0, 0]])

#     kernel = fmm.config['kernel']
#     p2p = KERNELS[kernel]['p2p']

#     direct = p2p(
#         sources=fmm.sources,
#         targets=distant_point,
#         source_densities=fmm.source_densities
#     )

#     equivalent = p2p(
#         sources=upward_equivalent_surface,
#         targets=distant_point,
#         source_densities=fmm.multipole_expansions[0]
#     )

#     assert np.allclose(direct, equivalent, rtol=RTOL)


def test_downward_pass(fmm):
    fmm.upward_pass()
    print('here', 65537 in fmm.sources_to_keys)
    fmm.downward_pass()

    print(fmm.local_expansions[196610])
    assert False



# def test_fmm(fmm):
#     """
#     End To End Fmm Test
#     """
#     fmm.upward_pass()
#     fmm.downward_pass()

#     direct = operator.p2p(
#         kernel_function=fmm.kernel_function,
#         targets=fmm.targets,
#         sources=fmm.sources,
#         source_densities=fmm.source_densities
#     ).density

#     fmm_results = np.array([result.density for result in fmm.result_data]).flatten()

#     percentage_error = 100*(abs(fmm_results - direct))/direct

#     average_percentage_error = sum(percentage_error)/len(percentage_error)

#     # Tested with order 2, so not very accurate
#     assert abs(average_percentage_error) < 7.5
