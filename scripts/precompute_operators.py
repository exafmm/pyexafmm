"""
Script to precompute and store M2M/L2L and M2L operators.
"""
import os
import pathlib
import sys

import numpy as np

from fmm.fmm import Laplace
from fmm.octree import Octree
from fmm.operator import (
    compute_surface, gram_matrix, scale_surface, compute_check_to_equivalent_inverse
    )
import fmm.hilbert

import utils.data as data

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent


CONFIG_OBJECTS = {
    'kernel_functions': {
        'laplace': Laplace()
    }
}


def main(
        operator_dirname,
        surface_filename,
        order,
        alpha_inner,
        alpha_outer,
        kernel,
        data_dirname,
        source_filename,
        target_filename,
        octree_max_level
        ):

    data_dirpath = PARENT / f"{data_dirname}/"

    operator_dirpath = PARENT / f"{operator_dirname}/"

    # Step 0: onstruct Octree and load Python config objs
    print("source filename", data_dirpath)

    sources = data.load_hdf5_to_array('sources', source_filename, data_dirpath)
    targets = data.load_hdf5_to_array('targets', target_filename, data_dirpath)
    octree = Octree(sources, targets, octree_max_level)

    # Load required Python objects
    kernel_function = CONFIG_OBJECTS['kernel_functions'][kernel]

    # Step 1: Compute a surface of a given order
    # Check if surface already exists
    if data.file_in_directory(surface_filename, operator_dirpath):
        print(f"Already Computed Surface of Order {order}")
        print(f"Loading ...")
        surface = data.load_hdf5_to_array(surface_filename, surface_filename, operator_dirpath)

    else:
        print(f"Computing Surface of Order {order}")
        surface = compute_surface(order)
        print("Saving Surface to HDF5")
        data.save_array_to_hdf5(operator_dirpath, f'{surface_filename}', surface)

    # Step 2: Use surfaces to compute inverse of check to equivalent Gram matrix.
    # This is a useful quantity that will form the basis of most operators.

    if data.file_in_directory('uc2e_u', operator_dirpath):
        print(f"Already Computed Inverse of Check To Equivalent Kernel of Order {order}")
        print("Loading...")

        # Upward check to upward equivalent
        uc2e_u = data.load_hdf5_to_array('uc2e_u', 'uc2e_u', operator_dirpath)
        uc2e_v = data.load_hdf5_to_array('uc2e_v', 'uc2e_v', operator_dirpath)

        # Downward check to downward equivalent
        dc2e_u = data.load_hdf5_to_array('dc2e_u', 'dc2e_u', operator_dirpath)
        dc2e_v = data.load_hdf5_to_array('dc2e_v', 'dc2e_v', operator_dirpath)

    else:
        print(f"Computing Inverse of Check To Equivalent Gram Matrix of Order {order}")

        # Compute upward check surface and upward equivalent surface
        # These are computed in a decomposed from the SVD of the Gram matrix
        # of these two surfaces

        upward_equivalent_surface = scale_surface(
            surface=surface,
            radius=octree.radius,
            level=0,
            center=octree.center,
            alpha=alpha_inner
        )

        upward_check_surface = scale_surface(
            surface=surface,
            radius=octree.radius,
            level=0,
            center=octree.center,
            alpha=alpha_outer
        )

        uc2e_v, uc2e_u, dc2e_v, dc2e_u = compute_check_to_equivalent_inverse(
            kernel_function, upward_check_surface, upward_equivalent_surface
        )

        # Save matrices
        print("Saving SVD Decompositions")
        data.save_array_to_hdf5(operator_dirpath, 'uc2e_v', uc2e_v)
        data.save_array_to_hdf5(operator_dirpath, 'uc2e_u', uc2e_u)
        data.save_array_to_hdf5(operator_dirpath, 'dc2e_v', dc2e_v)
        data.save_array_to_hdf5(operator_dirpath, 'dc2e_u', dc2e_u)

    # Compute M2M operator
    if (
            data.file_in_directory('m2m', operator_dirpath)
            and data.file_in_directory('l2l', operator_dirpath)
       ):
        print(f"Already Computed M2M & L2L Operators of Order {order}")

    else:
        parent_center = octree.center
        parent_radius = octree.radius
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

        scale = (1/2)**(child_level)

        print(f"Computing M2M & L2L Operators of Order {order}")
        for child_idx, child_center in enumerate(child_centers):
            print(loading)

            child_upward_equivalent_surface = scale_surface(
                surface, parent_radius, child_level, child_center, alpha_inner
            )

            pc2ce = gram_matrix(
                kernel_function,
                child_upward_equivalent_surface,
                parent_upward_check_surface,
            )


            # Compute M2M operator for this octant
            tmp = np.matmul(uc2e_u, pc2ce)
            m2m.append(np.matmul(uc2e_v, tmp))

            # Compute L2L operator for this octant
            pc2ce = pc2ce.T
            tmp = np.matmul(pc2ce, scale*dc2e_v)
            l2l.append(np.matmul(tmp, dc2e_u))

            loading += '.'

        # Save m2m & l2l operators, index is equivalent to their Hilbert key
        m2m = np.array(m2m)
        l2l = np.array(l2l)
        print("Saving M2M & L2L Operators")
        data.save_array_to_hdf5(operator_dirpath, 'm2m', m2m)
        data.save_array_to_hdf5(operator_dirpath, 'l2l', l2l)

    # Compute M2L operators
    if data.file_in_directory('m2l', operator_dirpath):
        print(f"Already Computed M2L Operators of Order {order}")

    else:
        print(f"Computing M2L Operators of Order {order}")
        m2l = []
        # Consider central cube at level 3, with a dense interaction list.
        # This will be used to calculate all the m2l operators for a given
        # Octree.

        x0 = octree.center
        r0 = octree.radius
        target_level = source_level = 3

        center_key = fmm.hilbert.get_key_from_point(x0, target_level, x0, r0)
        center_4d_index = fmm.hilbert.get_4d_index_from_key(center_key)

        # Compute interaction list for the central node
        interaction_list = fmm.hilbert.compute_interaction_list(center_key)

        # Data structure to hold relative vector of sources 2 target, distance
        # of the source node from the target in units of box width, and the key
        # of the source node.
        sources_relative_to_targets = np.zeros(shape=(189, 5))

        for source_idx, source_key in enumerate(interaction_list):

            source_4d_idx = fmm.hilbert.get_4d_index_from_key(source_key)

            source_relative_to_target = source_4d_idx[:3] - center_4d_index[:3]
            magnitude = np.linalg.norm(source_relative_to_target)

            sources_relative_to_targets[source_idx][:3] = source_relative_to_target
            sources_relative_to_targets[source_idx][3] = magnitude
            sources_relative_to_targets[source_idx][4] = source_key

        loading = '.'

        for idx, source_to_target_vec in enumerate(sources_relative_to_targets):
            print(loading)

            source_key = int(source_to_target_vec[-1])
            source_center = fmm.hilbert.get_center_from_key(source_key, x0, r0)
            source_level = target_level
            source_upward_equivalent_surface = scale_surface(
                surface, r0, source_level, source_center, alpha_inner
                )

            target_center = fmm.hilbert.get_center_from_key(center_key, x0, r0)

            # Compute target check surface
            target_downward_check_surface = scale_surface(
                surface, r0, target_level, target_center, alpha_inner
            )

            scale = (1/2)**(target_level)

            se2tc = gram_matrix(
                kernel_function,
                source_upward_equivalent_surface,
                target_downward_check_surface
            )

            tmp = np.matmul(dc2e_u, se2tc)
            m2l.append(np.matmul(scale*dc2e_v, tmp))

            loading += '.'

        m2l = np.array(m2l)
        print("Saving M2L Operators")

        # Indexes in m2l array are related to the `sources_relative_to_targets`
        # datastructure via corresponding index pointer.
        data.save_array_to_hdf5(operator_dirpath, 'm2l', m2l)

        data.save_array_to_hdf5(
            operator_dirpath,
            'sources_relative_to_targets',
            sources_relative_to_targets
        )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError(
            f'Must Specify Config Filepath!\
                e.g. `python precompute_operators.py /path/to/config.json`')
    else:
        config_filepath = sys.argv[1]
        config = data.load_json(config_filepath)
        main(**config)