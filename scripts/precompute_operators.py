"""
Script to precompute and store M2M/L2L and M2L operators. M2L computation
accellerated via multiprocessing.
"""
import os
import pathlib
import sys
import time

import numpy as np

import adaptoctree.morton as morton
import adaptoctree.tree as tree

import fmm.hilbert as hilbert
from fmm.kernel import KERNELS
from fmm.octree import Octree
import fmm.operator as operator
import utils.data as data
import utils.multiproc as multiproc
import utils.time


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent


# Setup a global variable for M2L counter
M2L_COUNTER = multiproc.setup_counter(default=9)
M2L_LOCK = multiproc.setup_lock()


# Incremeneter for M2L counter
def increment_m2l_counter():
    """
    Increment the M2L counter.
    """
    with M2L_LOCK:
        M2L_COUNTER.value += 1


def log_m2l_progress(counter_value, level):
    """
    Log M2L progress to stdout.
    """
    max_calculations = sum([8**(level) for level in range(0, level)])
    calculation_number = counter_value - max_calculations

    print(f"Computed {calculation_number}/{8**level} M2L Operators")


def compute_m2l_matrices(
        target, kernel, surface, alpha_inner, x0, r0, dc2e_v, dc2e_u, interaction_list,
    ):
    """
    Data has to be passed to each process in order to compute the m2l matrices
        associated with a given target key. This method takes the required data
        and computes all the m2l matrices for a given target.

    Parameters:
    -----------
    target : np.int64
        Target Hilbert Key.
    kernel : function
    surface : np.array(shape=(n, 3), dtype=np.float64)
    alpha_inner : np.float64
        Ratio between side length of shifted/scaled node and original node.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Center of Octree's root node.
    r0 : np.float64
        Half side length of Octree's root node.
    dc2e_v : np.array(shape=(n, n))
        First component of the inverse of the downward-check-to-equivalent
        Gram matrix at level 0.
    dc2e_v : np.array(shape=(n, n))
        Second component of the inverse of the downward-check-to-equivalent
        Gram matrix at level 0.

    Returns:
    --------
    list[np.array(shape=(n, n))]
        List of all m2l matrices associated with this target.
    """
    # Get level
    level = hilbert.get_level(target)

    # Container for results
    m2l_matrices = []

    # Increment counter, and print progress
    increment_m2l_counter()
    log_m2l_progress(M2L_COUNTER.value, level)

    # Compute target check surface
    target_center = hilbert.get_center_from_key(
        key=target,
        x0=x0,
        r0=r0
    )

    target_check_surface = operator.scale_surface(
        surface=surface,
        radius=r0,
        level=level,
        center=target_center,
        alpha=alpha_inner
    )

    # Compute m2l matrices
    for source in interaction_list:

        source_center = hilbert.get_center_from_key(
            key=source,
            x0=x0,
            r0=r0
        )

        source_equivalent_surface = operator.scale_surface(
            surface=surface,
            radius=r0,
            level=level,
            center=source_center,
            alpha=alpha_inner
        )

        se2tc = operator.gram_matrix(
            kernel_function=kernel,
            sources=source_equivalent_surface,
            targets=target_check_surface
        )

        scale = (1/kernel.scale)**(level)

        tmp = np.matmul(dc2e_u, se2tc)

        m2l_matrix = np.matmul(scale*dc2e_v, tmp)

        m2l_matrices.append(m2l_matrix)

    return m2l_matrices


def compute_surface(config, db):
    
    order = config['order']
    surface = operator.compute_surface(order)
    
    print(f"Computing Surface of Order {order}")

    if 'surface' in db.keys():
        del db['surface']
        db['surface'] = surface

    else:
        db['surface'] = surface

    return surface


def compute_octree(config, db):

    max_level = config['max_level']
    max_points = config['max_points']
    start_level = 1

    sources = db['particle_data']['sources'][...]
    targets = db['particle_data']['targets'][...]
    points = np.vstack((sources, targets))
    
    # Compute Octree 
    max_bound, min_bound = morton.find_bounds(points)
    x0 = morton.find_center(max_bound, min_bound)
    r0 = morton.find_radius(x0, max_bound, min_bound)
    
    unbalanced = tree.build(targets, max_level, max_points, start_level)
    u_depth = tree.find_depth(unbalanced)
    octree = tree.balance(unbalanced, u_depth)
    depth = tree.find_depth(octree)
    
    if 'octree' in db.keys():
        del db['octree']['keys']
        del db['octree']['depth']
        del db['octree']['x0']
        del db['octree']['r0']

        db['octree']['keys'] = octree
        db['octree']['depth'] = np.array([depth], np.int64)
        db['octree']['x0'] = np.array([x0], np.float64)
        db['octree']['r0'] = np.array([r0], np.float64)

    else:
        db.create_group('octree')

        db['octree']['keys'] = octree
        db['octree']['depth'] = np.array([depth], np.int64)
        db['octree']['x0'] = np.array([x0], np.float64)
        db['octree']['r0'] = np.array([r0], np.float64)

    return x0, r0, depth, octree


def compute_inv_c2e(config, db, kernel, surface, x0, r0):

    print(f"Computing Inverse of Check To Equivalent Gram Matrix of Order {config['order']}")
    
    upward_equivalent_surface = operator.scale_surface(
        surface=surface,
        radius=r0,
        level=0,
        center=x0,
        alpha=config['alpha_inner']
    )

    upward_check_surface = operator.scale_surface(
        surface=surface,
        radius=r0,
        level=0,
        center=x0,
        alpha=config['alpha_outer']
    )

    uc2e_v, uc2e_u = operator.compute_check_to_equivalent_inverse(
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

    if 'uc2e' in db.keys() and 'dc2e' in db.keys():

        del db['uc2e']
        del db['dc2e']

        db['uc2e']['u'] = uc2e_u
        db['uc2e']['v'] = uc2e_v
        db['dc2e']['u'] = dc2e_u
        db['dc2e']['v'] = dc2e_v

    else:

        db.create_group(f"uc2e")
        db.create_group(f"dc2e")

        db['uc2e']['u'] = uc2e_u
        db['uc2e']['v'] = uc2e_v
        db['dc2e']['u'] = dc2e_u
        db['dc2e']['v'] = dc2e_v

    return uc2e_u, uc2e_v, dc2e_u, dc2e_v


def main(**config):
    """
    Main script, configure using config.json file in module root.
    """
    start = time.time()

    # Setup Multiproc
    processes = os.cpu_count()
    pool = multiproc.setup_pool(processes=processes)

    # Step 0: Construct Octree and load Python config objs
    db = data.load_hdf5(config['experiment'], PARENT, 'a')
    x0, r0, depth, octree = compute_octree(config, db)

    # Load required Python objects
    kernel = KERNELS[config['kernel']]()

    # Step 1: Compute a surface of a given order
    surface = compute_surface(config, db)

    # # Step 2: Use surfaces to compute inverse of check to equivalent Gram matrix.
    # # This is a useful quantity that will form the basis of most operators.
    uc2e_u, uc2e_v, dc2e_u, dc2e_v = compute_inv_c2e(config, db, kernel, surface, x0, r0)

    # # Step 3: Compute M2M/L2L operators
    # if (
    #         data.file_in_directory('m2m', operator_dirpath)
    #         and data.file_in_directory('l2l', operator_dirpath)
    #    ):
    #     print(f"Already Computed M2M & L2L Operators of Order {config['order']}")

    # else:
    #     parent_center = octree.center
    #     parent_radius = octree.radius
    #     parent_level = 0
    #     child_level = 1

    #     child_centers = [
    #         hilbert.get_center_from_key(child, parent_center, parent_radius)
    #         for child in hilbert.get_children(0)
    #     ]

    #     parent_upward_check_surface = operator.scale_surface(
    #         surface=surface,
    #         radius=octree.radius,
    #         level=parent_level,
    #         center=octree.center,
    #         alpha=config['alpha_outer']
    #         )

    #     m2m = []
    #     l2l = []

    #     loading = len(child_centers)

    #     scale = (1/kernel.scale)**(child_level)

    #     print(f"Computing M2M & L2L Operators of Order {config['order']}")
    #     for child_idx, child_center in enumerate(child_centers):
    #         print(f'Computed ({child_idx+1}/{loading}) M2L/L2L operators')

    #         child_upward_equivalent_surface = operator.scale_surface(
    #             surface=surface,
    #             radius=octree.radius,
    #             level=child_level,
    #             center=child_center,
    #             alpha=config['alpha_inner']
    #         )

    #         pc2ce = operator.gram_matrix(
    #             kernel_function=kernel,
    #             targets=parent_upward_check_surface,
    #             sources=child_upward_equivalent_surface,
    #         )

    #         # Compute M2M operator for this octant
    #         tmp = np.matmul(uc2e_u, pc2ce)
    #         m2m.append(np.matmul(uc2e_v, tmp))

    #         # Compute L2L operator for this octant
    #         cc2pe = operator.gram_matrix(
    #             kernel_function=kernel,
    #             targets=child_upward_equivalent_surface,
    #             sources=parent_upward_check_surface
    #         )

    #         tmp = np.matmul(dc2e_u, cc2pe)
    #         l2l.append(np.matmul(scale*dc2e_v, tmp))

    #     # Save m2m & l2l operators, index is equivalent to their Hilbert key
    #     m2m = np.array(m2m)
    #     l2l = np.array(l2l)
    #     print("Saving M2M & L2L Operators")
    #     data.save_array_to_hdf5(operator_dirpath, 'm2m', m2m)
    #     data.save_array_to_hdf5(operator_dirpath, 'l2l', l2l)

    # # Step 4: Compute M2L operators

    # # Create sub-directory to store m2l computations
    # m2l_dirpath = operator_dirpath
    # current_level = 2

    # already_computed = False

    # while current_level <= config['octree_max_level']:

    #     m2l_filename = f'm2l_level_{current_level}'

    #     if data.file_in_directory(m2l_filename, operator_dirpath, ext='pkl'):
    #         already_computed = True

    #     if already_computed:
    #         print(f"Already Computed M2L operators for level {current_level}")

    #     else:

    #         print(f"Computing M2L Operators for Level {current_level}")

    #         leaves = np.arange(
    #             hilbert.get_level_offset(current_level),
    #             hilbert.get_level_offset(current_level+1)
    #             )

    #         loading = 0

    #         m2l = [
    #             [] for leaf in range(len(leaves))
    #         ]

    #         index_to_key = [
    #             None for leaf in range(len(leaves))
    #         ]

    #         index_to_key_filename = f'index_to_key_level_{current_level}'

    #         args = []

    #         # Gather arguments needed to send out to processes, and create index
    #         # mapping
    #         for target_idx, target in enumerate(leaves):

    #             interaction_list = hilbert.get_interaction_list(target)

    #             # Create index mapping for looking up the m2l operator
    #             index_to_key[target_idx] = interaction_list

    #             # Add arg to args for parallel mapping
    #             arg = (
    #                 target, kernel, surface, config['alpha_inner'],
    #                 octree.center, octree.radius, dc2e_v, dc2e_u, interaction_list
    #                 )

    #             args.append(arg)

    #         # Submit tasks to process pool
    #         m2l = pool.starmap(compute_m2l_matrices, args)

    #         # Convert results to matrix
    #         m2l = np.array([np.array(l) for l in m2l])

    #         print(f"Saving Dense M2L Operators for level {current_level}")
    #         data.save_pickle(
    #             m2l, m2l_filename, m2l_dirpath
    #         )
    #         data.save_pickle(
    #             index_to_key, index_to_key_filename, m2l_dirpath
    #         )

    #     current_level += 1
    #     already_computed = False

    # minutes, seconds = utils.time.seconds_to_minutes(time.time() - start)
    # print(f"Total time elapsed {minutes:.0f} minutes and {seconds:.0f} seconds")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError(
            f'Must Specify Config Filepath!\
                e.g. `python precompute_operators.py /path/to/config.json`')
    else:
        config_filepath = sys.argv[1]
        config = data.load_json(config_filepath)
        main(**config)
