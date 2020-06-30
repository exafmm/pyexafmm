import pathlib

import h5py
import json
import numpy as np

from fmm.fmm import Laplace
from fmm.operator import (
    compute_surface, gram_matrix, scale_surface, compute_check_to_equivalent
    )


def save_array_to_h5py(dirname, filename, array):
    """
    Save a Numpy Array to HDF5 format.

    Parameters:
    -----------
    filename : str
    array : np.ndarray
    """
    dirpath = pathlib.Path(dirname)
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = dirpath / f'{filename}.hdf5'

    with h5py.File(filepath, 'a') as f:
        f.create_dataset(f"{filename}", data=array)


def load_h5py(filename, directory):

    dirpath = pathlib.Path(directory)
    filepath = dirpath / f'{filename}.hdf5'

    return h5py.File(filepath, 'r')


def load_json(filename, directory):

    dirpath = pathlib.Path(directory)
    filepath = dirpath/ f'{filename}.json'

    with open(filepath, 'r') as f:
        obj = json.load(f)
    return obj


def file_in_directory(filename, directory):
    """
    Check if a file with a given name already exists in a given directory.

    filename : str
    directory: str
    """
    dirpath = pathlib.Path(directory).glob('*.hdf5')

    files = [f for f in dirpath if f.is_file()]

    for file_ in files:
        if filename in file_.name:
            return True
    return False


CONFIG_OBJECTS = {
    'centers': {
        'origin': np.array([[0, 0, 0]])
    },
    'kernel_functions': {
        'laplace': Laplace()
    }
}


def main(
    dirname,
    surface_filename,
    check_to_equivalent_filename,
    order,
    root_node_radius,
    root_node_level,
    root_node_center,
    alpha_inner,
    alpha_outer,
    kernel,
    ):

    # Check if surface already exists
    if file_in_directory(surface_filename, dirname):
        print(f"Already Computed Surface of Order {order}")
        print(f"Loading ...")
        surface = load_h5py(surface_filename, dirname)

    else:
        print(f"Computing Surface of Order {order}")
        surface = compute_surface(order)
        print("Saving Surface to HDF5")
        save_array_to_h5py(dirname, f'{surface_filename}', surface)

    # Use surfaces to compute check to equivalent gram matrix
    if file_in_directory(check_to_equivalent_filename, dirname):
        print(f"Already Compute Check To Equivalent Kernel of Order {order}")
        print("Loading...")
        c2e_matrix = load_h5py(check_to_equivalent_filename, dirname)

    else:
        center = CONFIG_OBJECTS['centers'][root_node_center]
        level = root_node_level
        radius = root_node_radius

        kernel_function = CONFIG_OBJECTS['kernel_functions'][kernel]

        # Compute upward check surface and upward equivalent surface
        # These are computed in a decomposed from the SVD of the Gram Matrix
        # of these two surfaces
        upward_equivalent_surface = scale_surface(
            surface, radius, level, center, alpha_inner
        )
        upward_check_surface = scale_surface(
            surface, radius, level, center, alpha_outer
        )

        uc2e_v, uc2e_u, dc2e_v, dc2e_u = compute_check_to_equivalent(
            kernel_function, upward_check_surface, upward_equivalent_surface
        )

        # Save matrices
        save_array_to_h5py(dirname, 'uc2e_v', uc2e_v)
        save_array_to_h5py(dirname, 'uc2e_u', uc2e_u)
        save_array_to_h5py(dirname, 'dc2e_v', dc2e_v)
        save_array_to_h5py(dirname, 'dc2e_u', dc2e_u)






if __name__ == "__main__":
    config = load_json('config', '../')
    main(**config)