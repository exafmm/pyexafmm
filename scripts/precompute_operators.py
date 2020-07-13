import pathlib

import h5py
import json
import numpy as np

from fmm.fmm import Laplace
from fmm.operator import (
    compute_surface, gram_matrix, scale_surface, compute_check_to_equivalent
    )
import fmm.hilbert


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


def load_json(filename, directory):
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

    dirpath = pathlib.Path(directory)
    filepath = dirpath/ f'{filename}.json'

    with open(filepath, 'r') as f:
        obj = json.load(f)
    return obj


def file_in_directory(filename, directory):
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


def compute_neighbors(key):
    vec = fmm.hilbert.get_4d_index_from_key(key)
    count = -1
    offset = np.zeros(4, dtype=np.int64)

    neighbors = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                count += 1
                offset[:3] = i, j, k
                neighbor_vec = vec + offset
                neighbor_key = fmm.hilbert.get_key(neighbor_vec)
                neighbors.append(neighbor_key)

    return neighbors


def compute_interaction_list(key):
    parent_key = fmm.hilbert.get_parent(key)

    parent_neighbors = compute_neighbors(parent_key)
    child_neighbors = compute_neighbors(key)

    interaction_list = []
    for parent_neighbor in parent_neighbors:
        children = fmm.hilbert.get_children(parent_neighbor)
        for child in children:
            if child not in child_neighbors:
                interaction_list.append(child)

    return interaction_list


def main(
    operator_dirname,
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
    if file_in_directory(surface_filename, operator_dirname):
        print(f"Already Computed Surface of Order {order}")
        print(f"Loading ...")
        surface = load_hdf5_to_array(surface_filename, surface_filename, operator_dirname)

    else:
        print(f"Computing Surface of Order {order}")
        surface = compute_surface(order)
        print("Saving Surface to HDF5")
        save_array_to_hdf5(operator_dirname, f'{surface_filename}', surface)

    # Use surfaces to compute check to equivalent Gram matrix
    if file_in_directory('uc2e_u', operator_dirname):
        print(f"Already Computed Check To Equivalent Kernel of Order {order}")
        print("Loading...")
        uc2e_u = load_hdf5_to_array('uc2e_u', 'uc2e_u', operator_dirname)
        uc2e_v = load_hdf5_to_array('uc2e_v', 'uc2e_v', operator_dirname)
        dc2e_u = load_hdf5_to_array('dc2e_u', 'dc2e_u', operator_dirname)
        dc2e_v = load_hdf5_to_array('dc2e_v', 'dc2e_v', operator_dirname)

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
        save_array_to_hdf5(operator_dirname, 'uc2e_v', uc2e_v)
        save_array_to_hdf5(operator_dirname, 'uc2e_u', uc2e_u)
        save_array_to_hdf5(operator_dirname, 'dc2e_v', dc2e_v)
        save_array_to_hdf5(operator_dirname, 'dc2e_u', dc2e_u)

    # Compute M2M operator
    if file_in_directory('m2m', operator_dirname) and file_in_directory('l2l', operator_dirname):
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
        print(m2m[0])
        l2l = np.array(l2l)
        print("Saving M2M & L2L Operators")
        save_array_to_hdf5(operator_dirname, 'm2m', m2m)
        save_array_to_hdf5(operator_dirname, 'l2l', l2l)

    # Compute M2L operators

    if file_in_directory('m2l', operator_dirname):
        print(f"Already Computed M2L Operators of Order {order}")

    else:
        print(f"Computing M2L Operators of Order {order}")
        m2l = []
        # Centre cube at level 3

        x0 = np.array([[0, 0, 0]])
        r0 = 1
        target_level = 3

        center_key = fmm.hilbert.get_key_from_point(x0, target_level, x0, r0)
        center_4d_idx = fmm.hilbert.get_4d_index_from_key(center_key)

        interaction_list = compute_interaction_list(center_key)

        source_to_target_vecs = np.zeros(shape=(189, 5))
        for source_idx, source in enumerate(interaction_list):

            source_4d_idx = fmm.hilbert.get_4d_index_from_key(source)

            diff = source_4d_idx[:3] - center_4d_idx[:3]
            magnitude = np.linalg.norm(diff)

            source_to_target_vecs[source_idx][:3] = diff
            source_to_target_vecs[source_idx][3] = magnitude
            source_to_target_vecs[source_idx][4] = source

        # Sort based on relative distance between sources and target
        source_to_target_vecs = \
            source_to_target_vecs[source_to_target_vecs[:, 3].argsort()]

        # Only need to compute M2L operators for unique distance vector
        distances_considered = []

        kernel_function = CONFIG_OBJECTS['kernel_functions'][kernel]

        loading = '.'
        for idx, source_to_target_vec in enumerate(source_to_target_vecs):
            if source_to_target_vec[3] not in distances_considered:
                print(loading)
                loading += '.'
                # Compute source surface
                distances_considered.append(source_to_target_vec[3])
                source_key = int(source_to_target_vec[-1])
                source_center = fmm.hilbert.get_center_from_key(source_key, x0, r0)
                source_level = target_level

                source_upward_equivalent_surface = scale_surface(
                    surface, r0, source_level, source_center, alpha_inner
                )

                # Compute target surfaces
                target_upward_check_surface = scale_surface(
                    surface, r0, target_level, x0, alpha_outer
                )

                target_upward_equivalent_surface = scale_surface(
                    surface, r0, target_level, x0, alpha_inner
                )

                uc2e_v, uc2e_u, dc2e_v, dc2e_u = compute_check_to_equivalent(
                    kernel_function,
                    target_upward_check_surface,
                    target_upward_equivalent_surface
                )

                s2tc = gram_matrix(
                    kernel_function,
                    source_upward_equivalent_surface,
                    target_upward_check_surface
                )

                tmp = np.matmul(uc2e_u, s2tc)
                m2l.append(np.matmul(uc2e_v, tmp))

        m2l = np.array(m2l)
        print("Saving M2L Operators")
        save_array_to_hdf5(operator_dirname, 'm2l', m2l)


if __name__ == "__main__":
    config = load_json('config', '../')
    main(**config)