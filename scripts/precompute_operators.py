import pathlib

import h5py
import json
import numpy as np

from fmm.fmm import Laplace
from fmm.operator import (
    compute_surface, gram_matrix, scale_surface, compute_check_to_equivalent
    )
import fmm.hilbert


def save_array_to_hdf5(dirname, filename, array):
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


def load_hdf5(filename, directory):

    dirpath = pathlib.Path(directory)
    filepath = dirpath / f'{filename}.hdf5'

    return h5py.File(filepath, 'r')


def load_hdf5_to_array(dataname, filename, directory):

    hdf5_file = load_hdf5(filename, directory)

    return hdf5_file[dataname][:]


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
        surface = load_hdf5_to_array(surface_filename, surface_filename, dirname)

    else:
        print(f"Computing Surface of Order {order}")
        surface = compute_surface(order)
        print("Saving Surface to HDF5")
        save_array_to_hdf5(dirname, f'{surface_filename}', surface)

    # Use surfaces to compute check to equivalent Gram matrix
    if file_in_directory('uc2e_u', dirname):
        print(f"Already Computed Check To Equivalent Kernel of Order {order}")
        print("Loading...")
        uc2e_u = load_hdf5_to_array('uc2e_u', 'uc2e_u', dirname)
        uc2e_v = load_hdf5_to_array('uc2e_v', 'uc2e_v', dirname)
        dc2e_u = load_hdf5_to_array('dc2e_u', 'dc2e_u', dirname)
        dc2e_v = load_hdf5_to_array('dc2e_v', 'dc2e_v', dirname)

    else:
        print(f"Computing SVD Decompositions of Check To Equivalent Gram Matrix of Order {order}")
        center = CONFIG_OBJECTS['centers'][root_node_center]
        level = root_node_level
        radius = root_node_radius

        kernel_function = CONFIG_OBJECTS['kernel_functions'][kernel]

        # Compute upward check surface and upward equivalent surface
        # These are computed in a decomposed from the SVD of the Gram matrix
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
        print("Saving SVD Decompositions")
        save_array_to_hdf5(dirname, 'uc2e_v', uc2e_v)
        save_array_to_hdf5(dirname, 'uc2e_u', uc2e_u)
        save_array_to_hdf5(dirname, 'dc2e_v', dc2e_v)
        save_array_to_hdf5(dirname, 'dc2e_u', dc2e_u)


    # Compute M2M operator
    if file_in_directory('m2m', dirname) and file_in_directory('l2l', dirname):
        print(f"Already Computed M2M & L2L Operators of Order {order}")

    else:
        parent_center = CONFIG_OBJECTS['centers'][root_node_center]
        kernel_function = CONFIG_OBJECTS['kernel_functions'][kernel]

        parent_radius = root_node_radius
        parent_level = 0
        child_level = 1

        child_centers = [
            fmm.hilbert.get_center_from_key(child, parent_center, parent_radius)
            for child in fmm.hilbert.get_children(0)
        ]

        parent_upward_check_surface = scale_surface(
                surface, parent_radius, parent_level, parent_center, alpha_outer
            )

        m2m = []
        l2l = []

        loading = '.'

        print(f"Computing M2M & L2L Operators of Order {order}")
        for child_center in child_centers:
            print(loading)

            child_upward_equivalent_surface = scale_surface(
                surface, parent_radius, child_level, child_center, alpha_inner
            )

            pc2ce = gram_matrix(
                kernel_function,
                parent_upward_check_surface,
                child_upward_equivalent_surface
            )

            # Compute M2M operator for this octant
            tmp = np.matmul(uc2e_u, pc2ce)
            m2m.append(np.matmul(uc2e_v, tmp))

            # Compute L2L operator for this octant
            pc2ce = pc2ce.T
            tmp = np.matmul(pc2ce, dc2e_v)
            l2l.append(np.matmul(tmp, dc2e_u))

            loading += '.'

        # Save m2m & l2l operators
        m2m = np.array(m2m)
        l2l = np.array(l2l)
        print("Saving M2M & L2L Operators")
        save_array_to_hdf5(dirname, 'm2m', m2m)
        save_array_to_hdf5(dirname, 'l2l', l2l)

    # Compute M2L operator

    print(m2m[0].shape)






if __name__ == "__main__":
    config = load_json('config', '../')
    main(**config)