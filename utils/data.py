"""
Data manipulation and i/o utils.
"""
import json
import pathlib
import pickle

import h5py


def save_array_to_hdf5(directory, filename, array):
    """
    Save a Numpy Array to HDF5 format.

    Parameters:
    -----------
    dirname : str
    filename : str
    array : np.ndarray

    Returns:
    --------
    None
    """
    dirpath = pathlib.Path(directory)
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = dirpath / f'{filename}.hdf5'

    with h5py.File(filepath, 'a') as f:
        f.create_dataset(f"{filename}", data=array)


def save_pickle(obj, filename, directory):

    dirpath = pathlib.Path(directory)
    filepath = dirpath / f'{filename}.pkl'

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename, directory):

    dirpath = pathlib.Path(directory)
    filepath = dirpath / f'{filename}.pkl'

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    return obj


def load_hdf5(filename, directory):
    """
    Load HDF5 file from disk.

    Parameters:
    -----------
    filename : str
    directory : str

    Returns:
    --------
    h5py.File
    """
    dirpath = pathlib.Path(directory)
    filepath = dirpath / f'{filename}.hdf5'

    return h5py.File(filepath, 'r')


def load_hdf5_to_array(dataname, filename, directory):
    """
    Load HDF5 file from disk into an Numpy array object.

    Parameters:
    ----------
    dataname : str
        HDF5 object data name
    filename : str
    directory : str

    Returns:
    --------
    np.ndarray
    """

    hdf5_file = load_hdf5(filename, directory)

    return hdf5_file[dataname][:]


def load_json(filepath):
    """
    Load json into dictionary.

    Parameters:
    -----------
    filename : str
    directory : str

    Returns:
    --------
    dict
    """

    with open(filepath, 'r') as f:
        obj = json.load(f)
    return obj


def file_in_directory(filename, directory, ext='hdf5'):
    """
    Check if a file with a given name already exists in a given directory.

    Parameters:
    -----------
    filename : str
    directory: str

    Returns:
    --------
    bool
    """
    dirpath = pathlib.Path(directory).glob(f'*.{ext}')

    files = [f for f in dirpath if f.is_file()]

    for file_ in files:
        if filename in file_.name:
            return True
    return False


def directory_exists(dirpath):

    dirpath = pathlib.Path(dirpath)

    if dirpath.is_dir():
        return True
    return False