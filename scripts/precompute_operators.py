import pathlib

import h5py
import numpy as np

from fmm.operator import compute_surface, scale_surface

PRECOMPUTED_DIR = '../precomputed/'


def save_array_to_h5py(filename, array):
    """
    Save a Numpy Array to HDF5 format.

    Parameters:
    -----------
    filename : str
    array : np.ndarray
    """

    p = pathlib.Path(PRECOMPUTED_DIR)
    p.mkdir(parents=True, exist_ok=True)
    filepath = p / f'{filename}.hdf5'

    with h5py.File(filepath, 'a') as f:
        dset = f.create_dataset(f"{filename}", data=array)


def saved_array(filename, directory):

    dirpath = pathlib.Path(directory).glob('*.hdf5')

    files = [file_ for file_ in dirpath if file_.is_file()]

    for file_ in files:
        if filename in file_.name:
            return True
    return False


def precompute_surface(filename, order):

    print("Computing Surface")
    surface = compute_surface(order)
    print("Saving Surface to HDF5")
    save_array_to_h5py(f'{filename}', surface)


def precompute_check_to_equivalent(filename, order, level):
    pass


def main(filename, order):

    # Check if surface already exists
    if saved_array(filename, PRECOMPUTED_DIR):
        print(f"Already Computed Surface of Order {order}")
    else:
        print(f"Computing Surface of Order {order}")
        precompute_surface(filename, order)


if __name__ == "__main__":

    kwargs = {
        'filename': 'surface_order_5',
        'order': 5
    }

    main(**kwargs)