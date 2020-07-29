
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
    """
    Generate `npoint` random points with coordinate values in the range [0, 1).
        Points are both targets and sources.

    Parameters:
    -----------
    npoints : int

    Returns :
        tuple(
            np.array(shape=(npoints, 3)),
            np.array(shape=(npoints, 3)),
            np.array(shape=npoints)
        )
    """
    rand = np.random.rand
    sources = targets = rand(npoints, 3)
    source_densities = np.ones(npoints)

    return (targets, sources, source_densities)


def well_separated_data(npoints):
    """
    Generate `npoints` targets and `npoints` sources, which are wells separated
        from on another such that there are only 2 nodes occupied nodes in the
        octree up to and incuding level 4.

    Parameters:
    -----------
    npoints : int

    Returns :
        tuple(
            np.array(shape=(npoints, 3)),
            np.array(shape=(npoints, 3)),
            np.array(shape=npoints)
        )
    """
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

    data.save_array_to_hdf5(data_dirpath, 'sources', sources)
    data.save_array_to_hdf5(data_dirpath, 'targets', targets)
    data.save_array_to_hdf5(data_dirpath, 'source_densities', source_densities)


if __name__ == "__main__":

    if len(sys.argv) < 4:
        raise ValueError(
            f'Must Specify Config Filepath number of points and data type!\
                e.g. `python generate_test_data.py /path/to/config.json 100 random`'
                )

    elif sys.argv[3] not in DATA.keys():
        raise ValueError(
            f'Data type `{sys.argv[3]}` not valid. Must be either`separated` or `random`'
             )

    else:
        config_filepath = sys.argv[1]
        npoints = sys.argv[2]
        dtype = sys.argv[3]
        config = data.load_json(config_filepath)
        config['npoints'] = int(npoints)
        config['dtype'] = dtype
        main(**config)

