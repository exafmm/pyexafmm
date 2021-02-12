"""
Script to precompute and store M2M/L2L and M2L operators. M2L computation
accellerated via multiprocessing.
"""
import os
import pathlib
import sys
import time

import cupy as cp
import numpy as np

import adaptoctree.morton as morton
import adaptoctree.tree as tree

from fmm.kernel import KERNELS
import fmm.operator as operator
import utils.data as data
import utils.multiproc as multiproc
import utils.time


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent
CUDA_FILEPATH = PARENT / 'fmm/cuda'

TPB = 32 # Threads Per Block for GPU
BLOCK_DIMENSIONS = (TPB, TPB)


def compute_dense_m2l_cpu(
        target, kernel, surface, alpha_inner, x0,
        r0, dc2e_v, dc2e_u, interaction_list,
    ):
    # Get level
    level = morton.find_level(target)
    scale = kernel.scale(level)

    # Container for results
    m2l_matrices = []

    # Compute target check surface
    target_center = morton.find_physical_center_from_key(
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

        source_center = morton.find_physical_center_from_key(
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
            kernel_function=kernel.eval,
            sources=source_equivalent_surface,
            targets=target_check_surface
        )

        tmp = np.matmul(dc2e_u, se2tc)

        m2l_matrix = np.matmul(scale*dc2e_v, tmp)

        m2l_matrices.append(m2l_matrix)

    return m2l_matrices


def compute_dense_m2l(
        target, kernel, surface, alpha_inner, x0, r0,
        dc2e_v, dc2e_u, interaction_list, gram_matrix_gpu
    ):
    """
    Compute Dense M2L matrix.
    """
    # Get level and scale
    level = morton.find_level(target)
    scale = kernel.scale(level)

    # Block matrices, one block per interaction list item.
    n_blocks = len(interaction_list)
    x_dim, y_dim = dc2e_u.shape
    dc2e_v = scale*dc2e_v

    # Compute target check surface
    target_center = morton.find_physical_center_from_key(
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

    # Transfer target check surface to GPU
    target_check_surface_gpu = cp.asarray(target_check_surface)

    # Allocate space for all source equivalent surfaces on the GPU
    source_equivalent_surfaces_gpu = cp.zeros(shape=(n_blocks*x_dim, 3), dtype=np.float64)

    # Allocate a block matrix for the se2tc matrices on the GPU
    se2tc_block_gpu = cp.zeros(shape=(n_blocks*x_dim, y_dim), dtype=np.float64)

    # Allocate space to store results
    m2l_matrix_gpu = cp.zeros(shape=(n_blocks*x_dim, y_dim), dtype=np.float64)

    # Compute equivalent surface for each source, and transfer to GPU
    for i in range(n_blocks):

        # Block indices
        l_idx = i*x_dim
        r_idx = (i+1)*x_dim

        source = interaction_list[i]

        # Compute surface
        source_center = morton.find_physical_center_from_key(
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

        # Transfer to GPU
        source_equivalent_surfaces_gpu[l_idx:r_idx, :] = cp.asarray(source_equivalent_surface)

    # GPU grid dimensions derived from size of problem
    grid_dimensions = (int(np.ceil(x_dim/32)), int(np.ceil(y_dim/32)))

    # start = time.time()
    # Compute gram matrix for each source/target pair in a loop
    for i in range(n_blocks):
        l_idx = i*x_dim
        r_idx = (i+1)*x_dim

        # Index sources with which to compute gram matrix
        gram_matrix_gpu(
            grid_dimensions,
            BLOCK_DIMENSIONS,
            (
                source_equivalent_surfaces_gpu[l_idx:r_idx,:],
                target_check_surface_gpu,
                se2tc_block_gpu[l_idx:r_idx, :],
                x_dim,
                y_dim
            )
        )

    # print(f'First Loop time {time.time()-start:.5f} s')

    # Transfer dc2e components to GPU
    dc2e_u_gpu = cp.asarray(dc2e_u)
    dc2e_v_gpu = cp.asarray(dc2e_v)

    # Compute dense M2L matrix on GPU
    # start = time.time()
    for i in range(n_blocks):
        # Block indices
        l_idx = i*x_dim
        r_idx = (i+1)*x_dim

        tmp_gpu = cp.matmul(dc2e_u_gpu, se2tc_block_gpu[l_idx:r_idx, :])
        m2l_matrix_gpu[l_idx:r_idx, :] = cp.matmul(dc2e_v_gpu, tmp_gpu)

    # print(f'Second Loop time {time.time()-start:.5f} s')

    # Transfer result back to CPU
    return m2l_matrix_gpu.get()


def compute_surface(config, db):

    order = config['order']
    surface = operator.compute_surface(order)

    print(f"Computing Surface of Order {order}")

    if 'surface' in db.keys():
        del db['surface']

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
    complete = tree.complete_tree(octree)
    u, x, v, w = tree.find_interaction_lists(octree, complete, depth)

    sources_to_keys = tree.points_to_keys(sources, octree, depth, x0, r0)
    targets_to_keys = tree.points_to_keys(targets, octree, depth, x0, r0)

    if 'octree' in db.keys():
        del db['octree']['keys']
        del db['octree']['depth']
        del db['octree']['x0']
        del db['octree']['r0']
        del db['octree']['complete']
        del db['particle_data']['sources_to_keys']
        del db['particle_data']['targets_to_keys']
        del db['interaction_lists']['u']
        del db['interaction_lists']['x']
        del db['interaction_lists']['v']
        del db['interaction_lists']['w']

        for i in range(len(complete)):
            node = str(complete[i])
            del db['key_to_index'][node]

    else:
        db.create_group('octree')
        db.create_group('interaction_lists')
        db.create_group('key_to_index')

    db['octree']['keys'] = octree
    db['octree']['depth'] = np.array([depth], np.int64)
    db['octree']['x0'] = np.array([x0], np.float64)
    db['octree']['r0'] = np.array([r0], np.float64)
    db['octree']['complete'] = complete

    db['particle_data']['sources_to_keys'] = sources_to_keys
    db['particle_data']['targets_to_keys'] = targets_to_keys

    for i in range(len(complete)):
        node = str(complete[i])
        db['key_to_index'][node] = np.array([i])

    db['interaction_lists']['u'] = u
    db['interaction_lists']['x'] = x
    db['interaction_lists']['v'] = v
    db['interaction_lists']['w'] = w

    return x0, r0, depth, octree, complete, u, x, v, w


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

        del db['uc2e']['u']
        del db['dc2e']['u']
        del db['uc2e']['v']
        del db['dc2e']['v']

    else:

        db.create_group('uc2e')
        db.create_group('dc2e')

    db['uc2e']['u'] = uc2e_u
    db['uc2e']['v'] = uc2e_v
    db['dc2e']['u'] = dc2e_u
    db['dc2e']['v'] = dc2e_v

    return uc2e_u, uc2e_v, dc2e_u, dc2e_v


def compute_m2m_and_l2l(
        config, db, surface, kernel, uc2e_u, uc2e_v, dc2e_u, dc2e_v,
        parent_center, parent_radius
    ):
    parent_level = 0
    child_level = 1

    child_centers = [
        morton.find_physical_center_from_key(child, parent_center, parent_radius)
        for child in morton.find_children(0)
    ]

    parent_upward_check_surface = operator.scale_surface(
        surface=surface,
        radius=parent_radius,
        level=parent_level,
        center=parent_center,
        alpha=config['alpha_outer']
        )

    m2m = []
    l2l = []

    loading = len(child_centers)

    kernel_function = kernel.eval
    scale = kernel.scale(child_level)

    print(f"Computing M2M & L2L Operators of Order {config['order']}")
    for child_idx, child_center in enumerate(child_centers):
        print(f'Computed ({child_idx+1}/{loading}) M2L/L2L operators')

        child_upward_equivalent_surface = operator.scale_surface(
            surface=surface,
            radius=parent_radius,
            level=child_level,
            center=child_center,
            alpha=config['alpha_inner']
        )

        pc2ce = operator.gram_matrix(
            kernel_function=kernel_function,
            targets=parent_upward_check_surface,
            sources=child_upward_equivalent_surface,
        )

        # Compute M2M operator for this octant
        tmp = np.matmul(uc2e_u, pc2ce)
        m2m.append(np.matmul(uc2e_v, tmp))

        # Compute L2L operator for this octant
        cc2pe = operator.gram_matrix(
            kernel_function=kernel_function,
            targets=child_upward_equivalent_surface,
            sources=parent_upward_check_surface
        )

        tmp = np.matmul(dc2e_u, cc2pe)
        l2l.append(np.matmul(scale*dc2e_v, tmp))

    # Save m2m & l2l operators
    m2m = np.array(m2m)
    l2l = np.array(l2l)

    print("Saving M2M & L2L Operators")

    if 'm2m' in db.keys() and 'l2l' in db.keys():
        del db['m2m']
        del db['l2l']

    db['m2m'] = m2m
    db['l2l'] = l2l


def compute_m2l(config, db, kernel, surface, x0, r0, dc2e_v, dc2e_u, complete):

    # Instantiate GPU kernel object
    kernel_src_filepath = CUDA_FILEPATH / f"{config['kernel']}.cu"

    with open(kernel_src_filepath, 'r') as f:
        kernel_src = f.read()

    module = cp.RawModule(code=kernel_src)
    gram_matrix_gpu = module.get_function('gram_matrix')

    if 'm2l' in db.keys():
        del db['m2l']

    else:
        db.create_group('m2l')
        group = db['m2l']
        group.create_group('dense')


    # This whole loop can be multi-processed
    progress = 0
    n_complete = len(complete)
    for i in range(n_complete):

        node = complete[i]
        node_str = str(node)
        node_idx = db['key_to_index'][node_str][0]
        v_list = db['interaction_lists']['v'][node_idx]
        v_list = v_list[v_list != -1]

        if len(v_list) > 0:
            m2l_matrix = compute_dense_m2l(
                node, kernel, surface, config['alpha_inner'],
                x0, r0, dc2e_v, dc2e_u, v_list, gram_matrix_gpu
            )

        else:
            m2l_matrix = np.array([-1])

        db['m2l']['dense'][node_str] = m2l_matrix

        progress += 1
        print(f'computed {progress}/{n_complete} m2l operators')


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
    x0, r0, depth, octree, complete, u, x, v, w = compute_octree(config, db)

    # Load required Python objects
    kernel = KERNELS[config['kernel']]()

    # Step 1: Compute a surface of a given order
    surface = compute_surface(config, db)

    # # Step 2: Use surfaces to compute inverse of check to equivalent Gram matrix.
    # # This is a useful quantity that will form the basis of most operators.
    uc2e_u, uc2e_v, dc2e_u, dc2e_v = compute_inv_c2e(config, db, kernel, surface, x0, r0)

    # Step 3: Compute M2M/L2L operators
    compute_m2m_and_l2l(
        config, db, surface, kernel, uc2e_u, uc2e_v, dc2e_u, dc2e_v,
        x0, r0
    )

    # Step 4: Compute M2L operators
    compute_m2l(config, db, kernel, surface, x0, r0, dc2e_v, dc2e_u, complete)

    minutes, seconds = utils.time.seconds_to_minutes(time.time() - start)
    print(f"Total time elapsed {minutes:.0f} minutes and {seconds:.0f} seconds")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError(
            f'Must Specify Config Filepath!\
                e.g. `python precompute_operators.py /path/to/config.json`')
    else:
        config_filepath = sys.argv[1]
        config = data.load_json(config_filepath)
        main(**config)
