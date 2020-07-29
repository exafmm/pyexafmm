
"""
Generate random test data
"""
import os
import pathlib
import shutil
import sys

import numpy as np

from fmm.kernel import Laplace
import fmm.operator as operator
import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent

def random_data(npoints):

    rand = np.random.rand
    sources = targets = rand(npoints, 3)
    source_densities = np.ones(npoints)

    return (targets, sources, source_densities)


def well_separated_data(npoints):

    source_center = np.array([-0.75, -0.75, -0.75])
    target_center = np.array([0.75, 0.75, 0.75])
    rand = np.random.rand(npoints, 3)*0.1

    sources = rand + source_center
    targets = rand + target_center
    source_densities = np.ones(npoints)

    return (targets, sources, source_densities)


DATA = {
    'random': random_data,
    'separated': well_separated_data
}


def main(**config):

    data_dirpath = PARENT / f"{config['data_dirname']}/"
    npoints = config['npoints']
    dtype = config['dtype']

    if os.path.isdir(data_dirpath):
        shutil.rmtree(data_dirpath)

    data_func = DATA[dtype]

    sources, targets, source_densities = data_func(npoints)

    data.save_array_to_hdf5(data_dirpath, 'random_sources', sources)
    data.save_array_to_hdf5(data_dirpath, 'random_targets', targets)
    data.save_array_to_hdf5(data_dirpath, 'source_densities', source_densities)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        raise ValueError(
            f'Must Specify Config Filepath and npoints!\
                e.g. `python generate_test_data.py /path/to/config.json 100`')
    else:
        config_filepath = sys.argv[1]
        npoints = sys.argv[2]
        dtype = sys.argv[3]
        config = data.load_json(config_filepath)
        config['npoints'] = int(npoints)
        config['dtype'] = dtype
        main(**config)

