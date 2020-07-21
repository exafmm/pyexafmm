import os
import pathlib

import numpy as np

import utils.data as data


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent

DATA_DIRPATH = PARENT / "data"

def main():

    rand = np.random.rand

    npoints = 100
    sources = targets = rand(npoints, 3)
    source_densities = np.ones(npoints)

    data.save_array_to_hdf5(DATA_DIRPATH, 'random_sources', sources)
    data.save_array_to_hdf5(DATA_DIRPATH, 'random_targets', targets)
    data.save_array_to_hdf5(DATA_DIRPATH, 'source_densities', source_densities)


if __name__ == "__main__":
    main()
