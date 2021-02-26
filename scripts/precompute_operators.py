"""
Precompute and store M2M/L2L/M2L operators for a given points dataset.

We use a Randomised SVD to compress the effect of all M2L operator matrices on
a single target node. This is accellerated with a GPU. Furthermore, dense
interactions are computed on-the-fly, rather than being stored in a dense
interaction matrix.
"""
import os
import pathlib
import sys
import time

import cupy as cp
import numpy as np
import scipy.linalg as linalg

import adaptoctree.morton as morton
import adaptoctree.tree as tree

from fmm.kernel import KERNELS, BLOCK_WIDTH, BLOCK_HEIGHT
import fmm.operator as operator
import utils.data as data
import utils.time


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent

TPB = BLOCK_HEIGHT


def compute_compressed_gram_matrix(
        node, surface_inner, surface_outer, alpha_inner, x0, r0, v_list,
        k, implicit_gram_matrix
    ):
    """
    Compute compressed representation of Gram matrices, using Randomised SVD.

    Strategy: We implicitly compute the Gram matrix between sources and targets
        to reduce memory writes of dense matrices.
    """

    # Get level
    level = morton.find_level(node)

    # Compute target check surface
    target_center = morton.find_physical_center_from_key(
        key=node,
        x0=x0,
        r0=r0
    )

    # Convert to single precision before transferring to GPU
    target_check_surface = operator.scale_surface(
        surface=surface_outer,
        radius=r0,
        level=level,
        center=target_center,
        alpha=alpha_inner
    ).astype(np.float32)

    # Allocate space on the GPU for sources and targets
    n_targets_per_node = len(surface_outer)
    n_sources_per_node = len(surface_inner)
    n_source_nodes = len(v_list)

    dtargets = cp.asarray(target_check_surface)

    csources = np.zeros((n_sources_per_node*n_source_nodes, 3), dtype=np.float32)

    # Compute source equivalent surfaces and transfer to GPU
    for idx in range(n_source_nodes):

        source = v_list[idx]

        source_center = morton.find_physical_center_from_key(
            key=source,
            x0=x0,
            r0=r0
        )

        source_equivalent_surface = operator.scale_surface(
            surface=surface_inner,
            radius=r0,
            level=level,
            center=source_center,
            alpha=alpha_inner
        ).astype(np.float32)

        lidx = idx*n_sources_per_node
        ridx = (idx+1)*n_sources_per_node

        csources[lidx:ridx] = source_equivalent_surface

    dsources = cp.asarray(csources)

    # Height and width of Gram matrix
    height = n_sources_per_node*n_source_nodes
    width = n_targets_per_node

    bpg = (int(np.ceil(width/BLOCK_WIDTH)), int(np.ceil(height/BLOCK_HEIGHT)))

    # Result data, in the language of RSVD
    dY = cp.zeros((height, k)).astype(np.float32)

    # Random matrix, for compressed basis, premultiplied by transpose of
    # Inverse components needed for M2L operator, so that implicit Kernel
    # matrix can be applied
    dOmega = cp.random.rand(n_targets_per_node, k).astype(np.float32)
    dOmegaT = dOmega.T

    # Perform implicit matrix matrix product between Gram matrix and random
    # matrix Omega
    for idx in range(k):
        implicit_gram_matrix[bpg, TPB](
            dsources, dtargets, dOmegaT, dY, height, width, idx
        )

    # Perform QR decomposition on the GPU
    dQ, _ = cp.linalg.qr(dY)
    dQT = dQ.T

    # Perform transposed matrix-matrix multiplication implicitly
    height = n_targets_per_node
    width = n_sources_per_node*n_source_nodes

    dBT = cp.zeros((n_targets_per_node, k)).astype(np.float32)

    # Blocking is transposed
    bpg = (int(np.ceil(width/BLOCK_WIDTH)), int(np.ceil(height/BLOCK_HEIGHT)))

    for idx in range(k):
        implicit_gram_matrix[bpg, TPB](dtargets, dsources, dQT, dBT, height, width, idx)

    # Perform SVD on reduced matrix
    du, dS, dVT = cp.linalg.svd(dBT.T, full_matrices=False)
    dU = cp.matmul(dQ, du)

    # Return compressed SVD components
    return (dU.get(), dS.get(), dVT.get())


def compute_surfaces(config, db):
    """
    Compute inner and outer surfaces. Check surfaces with a larger number
        of discretisation points tends to lead to better conditioning.
    """
    order_equivalent = config['order_equivalent']
    order_check = config['order_check']
    equivalent_surface = operator.compute_surface(order_equivalent)
    check_surface = operator.compute_surface(order_check)

    print(f"Computing Inner Surface of Order {order_equivalent}")
    print(f"Computing Outer Surface of Order {order_check}")

    if 'surface' in db.keys():
        del db['surface']

    db.create_group('surface')

    db['surface']['equivalent'] = equivalent_surface
    db['surface']['check'] = check_surface

    return equivalent_surface, check_surface


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

    return x0, r0, complete


def compute_inv_c2e(config, db, kernel, equivalent_surface, check_surface, x0, r0):

    gram_matrix = KERNELS[kernel]['dense_gram']

    print("Computing Inverse of Check To Equivalent Gram Matrix")

    upward_equivalent_surface = operator.scale_surface(
        surface=equivalent_surface,
        radius=r0,
        level=0,
        center=x0,
        alpha=config['alpha_inner']
    )

    upward_check_surface = operator.scale_surface(
        surface=check_surface,
        radius=r0,
        level=0,
        center=x0,
        alpha=config['alpha_outer']
    )

    downward_equivalent_surface = operator.scale_surface(
        surface=equivalent_surface,
        radius=r0,
        level=0,
        center=x0,
        alpha=config['alpha_outer']
    )

    downward_check_surface = operator.scale_surface(
        surface=check_surface,
        radius=r0,
        level=0,
        center=x0,
        alpha=config['alpha_inner']
    )

    uc2e = gram_matrix(
        targets=upward_check_surface,
        sources=upward_equivalent_surface,
    )

    uc2e_inv = linalg.pinv2(uc2e)

    dc2e = gram_matrix(
        targets=downward_check_surface,
        sources=downward_equivalent_surface,
    )

    dc2e_inv = linalg.pinv2(dc2e)

    if 'uc2e_inv' in db.keys() and 'dc2e_inv' in db.keys():

        del db['uc2e_inv']
        del db['dc2e_inv']

    db['uc2e_inv']= uc2e_inv
    db['dc2e_inv']= dc2e_inv

    return uc2e_inv, dc2e_inv


def compute_m2m_and_l2l(
        config, db, equivalent_surface, check_surface, kernel, uc2e_inv, dc2e_inv,
        parent_center, parent_radius
    ):

    parent_level = 0
    child_level = 1

    child_centers = [
        morton.find_physical_center_from_key(child, parent_center, parent_radius)
        for child in morton.find_children(0)
    ]

    parent_upward_check_surface = operator.scale_surface(
        surface=check_surface,
        radius=parent_radius,
        level=parent_level,
        center=parent_center,
        alpha=config['alpha_outer']
    )

    parent_downward_equivalent_surface = operator.scale_surface(
        surface=equivalent_surface,
        radius=parent_radius,
        level=parent_level,
        center=parent_center,
        alpha=config['alpha_outer']
    )


    m2m = []
    l2l = []

    loading = len(child_centers)

    gram_matrix = KERNELS[kernel]['dense_gram']
    kernel_scale = KERNELS[kernel]['scale']
    scale = kernel_scale(child_level)

    print("Computing M2M & L2L Operators")

    for child_idx, child_center in enumerate(child_centers):
        print(f'Computed ({child_idx+1}/{loading}) M2M/L2L operators')

        child_upward_equivalent_surface = operator.scale_surface(
            surface=equivalent_surface,
            radius=parent_radius,
            level=child_level,
            center=child_center,
            alpha=config['alpha_inner']
        )

        child_downward_check_surface = operator.scale_surface(
            surface=check_surface,
            radius=parent_radius,
            level=child_level,
            center=child_center,
            alpha=config['alpha_inner']
        )

        pc2ce = gram_matrix(
            targets=parent_upward_check_surface,
            sources=child_upward_equivalent_surface,
        )

        # Compute M2M operator for this octant
        m2m.append(np.matmul(uc2e_inv, pc2ce))

        # Compute L2L operator for this octant
        cc2pe = gram_matrix(
            targets=child_downward_check_surface,
            sources=parent_downward_equivalent_surface
        )

        l2l.append(np.matmul(scale*dc2e_inv, cc2pe))

    # Save m2m & l2l operators
    m2m = np.array(m2m)
    l2l = np.array(l2l)

    if 'm2m' in db.keys() and 'l2l' in db.keys():
        del db['m2m']
        del db['l2l']

    db['m2m'] = m2m
    db['l2l'] = l2l


def compute_m2l(config, db, kernel, equivalent_surface, check_surface, x0, r0, complete):

    # Get required GPU kernel
    implicit_gram_matrix = KERNELS[kernel]['implicit_gram']

    # Required config, not explicitly passed
    k = config['target_rank']

    if 'm2l' in db.keys():
        del db['m2l']

    else:
        db.create_group('m2l')

    group = db['m2l']

    progress = 0
    n_complete = len(complete)

    for i in range(n_complete):

        node = complete[i]
        node_str = str(node)
        node_idx = db['key_to_index'][node_str][0]
        v_list = db['interaction_lists']['v'][node_idx]
        v_list = v_list[v_list != -1]

        if len(v_list) > 0 and node != 0:

            U, S, VT = compute_compressed_gram_matrix(
                node, equivalent_surface, check_surface, config['alpha_inner'],
                x0, r0, v_list, k, implicit_gram_matrix
            )

            if node_str in group.keys():
                del db['m2l'][node_str]

            else:
                group.create_group(node_str)

            db['m2l'][node_str]['U'] = U
            db['m2l'][node_str]['S'] = S
            db['m2l'][node_str]['VT'] = VT

        progress += 1

        print(f'Computed ({progress}/{n_complete}) M2L operators')


def main(**config):
    """
    Main script, configure using config.json file in module root.
    """
    start = time.time()

    # Step 0: Construct Octree and load Python config objs
    db = data.load_hdf5(config['experiment'], PARENT, 'a')
    x0, r0, complete = compute_octree(config, db)

    # Load required Python objects
    kernel = config['kernel']

    # Step 1: Compute a surface of a given order
    equivalent_surface, check_surface = compute_surfaces(config, db)

    # # Step 2: Use surfaces to compute inverse of check to equivalent Gram matrix.
    # # This is a useful quantity that will form the basis of most operators.
    uc2e_inv, dc2e_inv = compute_inv_c2e(config, db, kernel, equivalent_surface, check_surface, x0, r0)

    # Step 3: Compute M2M/L2L operators
    compute_m2m_and_l2l(
        config, db, equivalent_surface, check_surface, kernel, uc2e_inv, dc2e_inv, x0, r0
    )

    # # Step 4: Compute M2L operators
    compute_m2l(config, db, kernel, equivalent_surface, check_surface, x0, r0, complete)

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
