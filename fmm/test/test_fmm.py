"""
Test the FMM
"""
import os
import pathlib

import numpy as np
import pytest

from fmm.fmm import Fmm
import fmm.operator as operator
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


def test_fmm(fmm):
    """
    End To End Fmm Test
    """
    fmm.upward_pass()
    fmm.downward_pass()

    direct = operator.p2p(
        kernel_function=fmm.kernel_function,
        targets=fmm.targets,
        sources=fmm.sources,
        source_densities=fmm.source_densities
    ).density

    fmm_results = np.array([result.density for result in fmm.result_data]).flatten()

    percentage_error = 100*(abs(fmm_results - direct))/direct

    average_percentage_error = sum(percentage_error)/len(percentage_error)

    assert average_percentage_error < 7.5
