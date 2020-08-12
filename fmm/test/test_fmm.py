"""
Test the FMM
"""
import collections
import os
import pathlib
import subprocess

import numpy as np
import pytest

from fmm.fmm import Fmm
import fmm.operator as operator
import fmm.hilbert as hilbert
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
