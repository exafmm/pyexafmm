
"""
Generate random test data
"""
import os
import pathlib
import sys

import h5py
import numpy as np

from fmm.dtype import DTYPE

import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent

WORKING_DIR = pathlib.Path(os.getcwd())


def random_data(npoints, dtype):
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
    np.random.seed(0)
    sources = targets = np.random.rand(npoints, 3).astype(dtype)
    source_densities = np.random.rand(npoints).astype(dtype)

    return (targets, sources, source_densities)


def spherical_data(npoints, dtype):
    """
    Generate `npoints` targets and `npoints` sources, which are supported on the
        surface of a sphere, with a unit diameter.

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
    np.random.seed(0)
    phi = np.random.rand(npoints)*2*np.pi
    costheta = (np.random.rand(npoints)-0.5)*2

    theta = np.arccos(costheta)
    r = 0.5

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    sources = np.vstack((x, y, z)).astype(dtype)
    source_densities = np.random.rand(npoints).astype(dtype)

    return (sources.T, sources.T, source_densities)


DATA_FUNCTIONS = {
    'random': random_data,
    'sphere': spherical_data,
}


def main(**config):

    npoints = config['npoints']
    data_function = DATA_FUNCTIONS[config['data_type']]
    float_type = DTYPE[config['precision']]['float']

    sources, targets, source_densities = data_function(npoints, float_type)

    db = h5py.File(WORKING_DIR/f"{config['experiment']}.hdf5", 'a')

    if f'particle_data' in db.keys():
        del db[f'particle_data']['sources']
        del db[f'particle_data']['targets']
        del db[f'particle_data']['source_densities']

        db[f'particle_data']['sources'] = sources
        db[f'particle_data']['targets'] = targets
        db[f'particle_data']['source_densities'] = source_densities

    else:
        db.create_group(f'particle_data')

        db[f'particle_data']['sources'] = sources
        db[f'particle_data']['targets'] = targets
        db[f'particle_data']['source_densities'] = source_densities

    db.close()


if __name__ == "__main__":

    if sys.argv[2] not in DATA_FUNCTIONS.keys():
        raise ValueError(
            f'Data type `{sys.argv[2]}` not valid'
             )

    else:
        config_filepath = sys.argv[1]
        config = data.load_json(config_filepath)
        main(**config)
