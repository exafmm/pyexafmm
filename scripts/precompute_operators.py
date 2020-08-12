"""
Script to precompute and store M2M/L2L and M2L operators.
"""
import os
import pathlib
import sys

import numpy as np

import fmm.hilbert as hilbert
from fmm.kernel import KERNELS
from fmm.octree import Octree
import fmm.operator as operator
import utils.data as data

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent


def main(**config):
    """
    Main script, configure using config.json file in module root.
    """

    data_dirpath = PARENT / f"{config['data_dirname']}/"
    operator_dirpath = PARENT / f"{config['operator_dirname']}/"

    # Step 0: Construct Octree and load Python config objs
    print("source filename", data_dirpath)

    sources = data.load_hdf5_to_array(
        config['source_filename'],
        config['source_filename'],
        data_dirpath
        )

    targets = data.load_hdf5_to_array(
        config['target_filename'],
        config['target_filename'],
        data_dirpath
        )

    source_densities = data.load_hdf5_to_array(
        config['source_densities_filename'],
        config['source_densities_filename'],
        data_dirpath
        )

    octree = Octree(
        sources, targets, config['octree_max_level'], source_densities
        )

    # Load required Python objects
    kernel = KERNELS[config['kernel']]()

    # Step 1: Compute a surface of a given order
    # Check if surface already exists
    if data.file_in_directory(config['surface_filename'], operator_dirpath):
        print(f"Already Computed Surface of Order {config['order']}")
        print(f"Loading ...")
        surface = data.load_hdf5_to_array(
            config['surface_filename'],
            config['surface_filename'],
            operator_dirpath
            )

    else:
        print(f"Computing Surface of Order {config['order']}")
        surface = operator.compute_surface(config['order'])

        print("Saving Surface to HDF5")
        data.save_array_to_hdf5(
            operator_dirpath,
            config['surface_filename'],
            surface
            )

    # Step 2: Use surfaces to compute inverse of check to equivalent Gram matrix.
    # This is a useful quantity that will form the basis of most operators.

    if data.file_in_directory('uc2e_u', operator_dirpath):
        print(f"Already Computed Inverse of Check To Equivalent Kernel of Order {config['order']}")
        print("Loading...")

        # Upward check to upward equivalent
        uc2e_u = data.load_hdf5_to_array('uc2e_u', 'uc2e_u', operator_dirpath)
        uc2e_v = data.load_hdf5_to_array('uc2e_v', 'uc2e_v', operator_dirpath)

        # Downward check to downward equivalent
        dc2e_u = data.load_hdf5_to_array('dc2e_u', 'dc2e_u', operator_dirpath)
        dc2e_v = data.load_hdf5_to_array('dc2e_v', 'dc2e_v', operator_dirpath)

    else:
        print(f"Computing Inverse of Check To Equivalent Gram Matrix of Order {config['order']}")

        # Compute upward check surface and upward equivalent surface
        # These are computed in a decomposed from the SVD of the Gram matrix
        # of these two surfaces

        upward_equivalent_surface = operator.scale_surface(
            surface=surface,
            radius=octree.radius,
            level=0,
            center=octree.center,
            alpha=config['alpha_inner']
        )

        upward_check_surface = operator.scale_surface(
            surface=surface,
            radius=octree.radius,
            level=0,
            center=octree.center,
            alpha=config['alpha_outer']
        )

        uc2e_v, uc2e_u,= operator.compute_check_to_equivalent_inverse(
            kernel_function=kernel,
            check_surface=upward_check_surface,
            equivalent_surface=upward_equivalent_surface,
            cond=None
        )

        dc2e_v, dc2e_u = operator.compute_check_to_equivalent_inverse(
            kernel_function=kernel,
            check_surface=upward_equivalent_surface,
            equivalent_surface=upward_check_surface,
            cond=None
        )

        # Save matrices
        print("Saving SVD Decompositions")
        data.save_array_to_hdf5(operator_dirpath, 'uc2e_v', uc2e_v)
        data.save_array_to_hdf5(operator_dirpath, 'uc2e_u', uc2e_u)
        data.save_array_to_hdf5(operator_dirpath, 'dc2e_v', dc2e_v)
        data.save_array_to_hdf5(operator_dirpath, 'dc2e_u', dc2e_u)

    # Compute M2M/L2L operators
    if (
            data.file_in_directory('m2m', operator_dirpath)
            and data.file_in_directory('l2l', operator_dirpath)
       ):
        print(f"Already Computed M2M & L2L Operators of Order {config['order']}")

    else:
        parent_center = octree.center
        parent_radius = octree.radius
        parent_level = 0
        child_level = 1

        child_centers = [
            hilbert.get_center_from_key(child, parent_center, parent_radius)
            for child in hilbert.get_children(0)
        ]

        parent_upward_check_surface = operator.scale_surface(
            surface=surface,
            radius=octree.radius,
            level=parent_level,
            center=octree.center,
            alpha=config['alpha_outer']
            )

        m2m = []
        l2l = []

        loading = len(child_centers)

        scale = (1/kernel.scale)**(child_level)

        print(f"Computing M2M & L2L Operators of Order {config['order']}")
        for child_idx, child_center in enumerate(child_centers):
            print(f'Computed ({child_idx+1}/{loading}) M2L/L2L operators')

            child_upward_equivalent_surface = operator.scale_surface(
                surface=surface,
                radius=octree.radius,
                level=child_level,
                center=child_center,
                alpha=config['alpha_inner']
            )

            pc2ce = operator.gram_matrix(
                kernel_function=kernel,
                targets=parent_upward_check_surface,
                sources=child_upward_equivalent_surface,
            )

            # Compute M2M operator for this octant
            tmp = np.matmul(uc2e_u, pc2ce)
            m2m.append(np.matmul(uc2e_v, tmp))

            # Compute L2L operator for this octant
            # cc2pe = pc2ce.T
            cc2pe = operator.gram_matrix(
                kernel_function=kernel,
                targets=child_upward_equivalent_surface,
                sources=parent_upward_check_surface
            )

            tmp = np.matmul(dc2e_u, cc2pe)
            l2l.append(np.matmul(scale*dc2e_v, tmp))

        # Save m2m & l2l operators, index is equivalent to their Hilbert key
        m2m = np.array(m2m)
        l2l = np.array(l2l)
        print("Saving M2M & L2L Operators")
        data.save_array_to_hdf5(operator_dirpath, 'm2m', m2m)
        data.save_array_to_hdf5(operator_dirpath, 'l2l', l2l)

    # Compute M2L operators

    # Create sub-directory to store m2l computations
    m2l_dirpath = operator_dirpath
    current_level = 2

    already_computed = False

    while current_level <= config['octree_max_level']:

        m2l_filename = f'm2l_level_{current_level}'

        if data.file_in_directory(m2l_filename, operator_dirpath, ext='pkl'):
            already_computed = True

        if already_computed:
            print(f"Already Computed M2L operators for level {current_level}")

        else:

            print(f"Computing M2L Operators for Level {current_level}")

            leaves = np.arange(
                hilbert.get_level_offset(current_level),
                hilbert.get_level_offset(current_level+1)
                )

            loading = 0

            m2l = [
                [] for leaf in range(len(leaves))
            ]

            index_to_key = [
                None for leaf in range(len(leaves))
            ]

            index_to_key_filename = f'index_to_key_level_{current_level}'

            for target_idx, target in enumerate(leaves):

                print(f"Computed M2L operators for ({loading}/{len(leaves)}) leaves")
                loading += 1

                interaction_list = hilbert.get_interaction_list(target)

                target_center = hilbert.get_center_from_key(
                    key=target,
                    x0=octree.center,
                    r0=octree.radius
                )

                target_check_surface = operator.scale_surface(
                    surface=surface,
                    radius=octree.radius,
                    level=current_level,
                    center=target_center,
                    alpha=config['alpha_inner']
                )

                # Create index mapping for looking up the m2l operator
                index_to_key[target_idx] = interaction_list

                for source in interaction_list:

                    source_center = hilbert.get_center_from_key(
                        key=source,
                        x0=octree.center,
                        r0=octree.radius
                    )

                    source_equivalent_surface = operator.scale_surface(
                        surface=surface,
                        radius=octree.radius,
                        level=current_level,
                        center=source_center,
                        alpha=config['alpha_inner']
                    )

                    se2tc = operator.gram_matrix(
                        kernel_function=kernel,
                        sources=source_equivalent_surface,
                        targets=target_check_surface
                    )

                    scale = (1/kernel.scale)**(current_level)

                    tmp = np.matmul(dc2e_u, se2tc)

                    m2l_matrix = np.matmul(scale*dc2e_v, tmp)

                    m2l[target_idx].append(m2l_matrix)

            m2l = np.array(m2l)

            print("Saving M2L matrices")
            data.save_pickle(
                m2l, m2l_filename, m2l_dirpath
            )

            data.save_pickle(
                index_to_key, index_to_key_filename, m2l_dirpath
            )

        current_level += 1
        already_computed = False


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError(
            f'Must Specify Config Filepath!\
                e.g. `python precompute_operators.py /path/to/config.json`')
    else:
        config_filepath = sys.argv[1]
        config = data.load_json(config_filepath)
        main(**config)
