
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


def main(**config):

    data_dirpath = PARENT / f"{config['data_dirname']}/"
    npoints = config['npoints']

    if os.path.isdir(data_dirpath):
        shutil.rmtree(data_dirpath)

    rand = np.random.rand
    sources = targets = rand(npoints, 3)
    source_densities = np.ones(npoints)

    p2p_results = operator.p2p(
        kernel_function=Laplace(),
        targets=targets,
        sources=sources,
        source_densities=source_densities
    ).density

    data.save_array_to_hdf5(data_dirpath, 'random_sources', sources)
    data.save_array_to_hdf5(data_dirpath, 'random_targets', targets)
    data.save_array_to_hdf5(data_dirpath, 'source_densities', source_densities)
    data.save_array_to_hdf5(data_dirpath, 'p2p_results', p2p_results)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        raise ValueError(
            f'Must Specify Config Filepath and npoints!\
                e.g. `python generate_test_data.py /path/to/config.json 100`')
    else:
        config_filepath = sys.argv[1]
        npoints = sys.argv[2]
        config = data.load_json(config_filepath)
        config['npoints'] = int(npoints)
        main(**config)

