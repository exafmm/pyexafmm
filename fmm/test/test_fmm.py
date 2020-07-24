"""
Test the FMM
"""
import os
import pathlib
import subprocess

import numpy as np
import pytest

from fmm.fmm import Fmm
from fmm.operator import scale_surface, p2p
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILEPATH = HERE.parent.parent / "test_config.json"
CONFIG = data.load_json(CONFIG_FILEPATH)

ORDER = CONFIG['order']

OPERATOR_DIRPATH = HERE.parent.parent / f'precomputed_operators_order_{ORDER}'
SCRIPT_DIRPATH = HERE.parent.parent / 'scripts'


def setup_module(module):
    os.chdir(HERE.parent)
    subprocess.run(['python', SCRIPT_DIRPATH / 'precompute_operators.py', CONFIG_FILEPATH])
    os.chdir('test')


def teardown_module(module):
    os.chdir(HERE.parent.parent)
    subprocess.run(['rm', '-fr', f'precomputed_operators_order_{ORDER}'])


@pytest.fixture
def fmm():
    return Fmm(config_filename='test_config.json')


def test_upward_pass(fmm):
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
