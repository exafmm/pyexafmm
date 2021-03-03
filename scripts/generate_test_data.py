
"""
Generate random test data
"""
import os
import pathlib
import shutil
import sys

import numpy as np

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
    source_center = np.array([-1, -1, -1])
    target_center = np.array([1, 1, 1])
    rand = np.random.rand(npoints, 3)*0.1

    sources = rand + source_center
    targets = rand + target_center
    source_densities = np.ones(npoints)

    return (targets, sources, source_densities)


def spiral_data(npoints):

    theta = np.linspace(0, np.pi, npoints)
    phi = np.linspace(0, 2*np.pi, npoints)

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    sources = np.vstack([x, y, z]).T
    targets = sources
    source_densities = np.ones(npoints)

    return (targets, sources, source_densities)


DATA_FUNCTIONS = {
    'random': random_data,
    'separated': well_separated_data,
    'spiral': spiral_data
}


def main(**config):

    npoints = config['npoints']
    data_function = DATA_FUNCTIONS[config['data_type']]

    sources, targets, source_densities = data_function(npoints)

    db = data.load_hdf5(config['experiment'], PARENT, 'a')

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
