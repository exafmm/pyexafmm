"""
Precompute and store M2M/L2L/M2L operators for a given points dataset.
"""
import os
import pathlib
import sys
import time

import cupy as cp
import h5py
import numpy as np
import scipy.linalg as linalg

import adaptoctree.morton as morton
import adaptoctree.tree as tree

from fmm.kernel import KERNELS, BLOCK_WIDTH, BLOCK_HEIGHT
import fmm.surface as surface
from fmm.parameters import DIGEST_SIZE

import utils.data as data
import utils.time


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent
WORKING_DIR = pathlib.Path(os.getcwd())

TPB = BLOCK_HEIGHT


def compress_m2l_gram_matrix(
        level, x0, r0, depth, alpha_inner, check_surface,
        equivalent_surface, k, implicit_gram_matrix, digest_size=DIGEST_SIZE
    ):
    """
    Compute compressed representation of unique Gram matrices for targets and
    sources at a given level of the octree, specified by their unique transfer
    vectors. Compression is computed using the randomised-SVD of Halko et. al.
    (2011), and accelerated using CUDA. The Gram matrix is potentially extremely
    large for meshes of useful order of discretisation, therefore it is never
    computed explicitly, instead matrix elements are computed on the fly and
    applied to a given RHS where needed.

    Parameters:
    -----------
    level : np.int64
        Octree level at which M2L operators are being calculated.
    x0 : np.array(shape=(1, 3), dtype=np.float32)
        Physical center of octree root node.
    r0 : np.float32
        Half side length of octree root node.
    alpha_inner : np.float32
        Relative size of inner surface
    check_surface : np.array(shape=(n_check, 3), dtype=np.float32)
        Discretised check surface.
    equivalent_surface: np.array(shape=(n_equivalent, 3), dtype=np.float32)
        Discretised equivalent surface.
    k : np.int32
        Target compression rank.
    implicit_gram_matrix : CUDA JIT function handle.
        Function to apply gram matrix implicitly to a given RHS.
    digest_size : np.int64
        Size of the hash computed for each transfer vector.

    Returns:
    --------
    (
        np.array((nu, k), np.float32),
        np.array(ns, dtype=np.float32),
        np.array((k, k), dtype=np.float32)
    )
        SVD of Gram matrices corresponding to all unique transfer vectors for
        each level. For a given level, these are indexed by the hash of the
        transfer vectors.
    """
    sources, targets, hashes = tree.find_unique_v_list_interactions(
        level, x0, r0, depth, digest_size=digest_size
    )

    n_targets_per_node = len(check_surface)
    n_sources_per_node = len(equivalent_surface)
    n_targets = len(targets)
    n_sources = len(sources)

    dsources = cp.zeros((n_sources*n_sources_per_node, 3)).astype(np.float32)
    dtargets = cp.zeros((n_targets*n_targets_per_node, 3)).astype(np.float32)

    for idx in range(len(targets)):

        target = targets[idx]
        source = sources[idx]

        target_center = morton.find_physical_center_from_key(
            key=target,
            x0=x0,
            r0=r0
        )

        source_center = morton.find_physical_center_from_key(
            key=source,
            x0=x0,
            r0=r0
        )

        lidx_targets = (idx)*n_targets_per_node
        ridx_targets = (idx+1)*n_targets_per_node

        lidx_sources = (idx)*n_sources_per_node
        ridx_sources = (idx+1)*n_sources_per_node

        target_coordinates = surface.scale_surface(
            surf=check_surface,
            radius=np.float32(r0),
            level=np.int32(level),
            center=target_center.astype(np.float32),
            alpha=np.float32(alpha_inner)
        )

        source_coordinates = surface.scale_surface(
            surf=equivalent_surface,
            radius=np.float32(r0),
            level=np.int32(level),
            center=source_center.astype(np.float32),
            alpha=np.float32(alpha_inner)
        )

        # Compress the Gram matrix between these sources/targets
        dsources[lidx_sources:ridx_sources, :] = cp.asarray(source_coordinates)
        dtargets[lidx_targets:ridx_targets, :] = cp.asarray(target_coordinates)


    # Height and width of Gram matrix
    height = n_targets_per_node*n_targets
    sub_height = n_targets_per_node

    width = n_sources_per_node
    sub_width = n_sources_per_node

    bpg = (int(np.ceil(width/BLOCK_WIDTH)), int(np.ceil(height/BLOCK_HEIGHT)))

    # Result data, in the language of RSVD
    dY = cp.zeros((height, k)).astype(np.float32)

    # Random matrix, for compressed basis
    dOmega = cp.random.rand(n_sources_per_node, k).astype(np.float32)
    dOmegaT = dOmega.T

    # Perform implicit matrix matrix product between Gram matrix and random
    # matrix Omega
    for idx in range(k):
        implicit_gram_matrix[bpg, TPB](
            dsources, dtargets, dOmegaT, dY, height, width, sub_height, sub_width, idx
        )

    # Perform QR decomposition on the GPU
    dQ, _ = cp.linalg.qr(dY)
    dQT = dQ.T

    # Perform transposed matrix-matrix multiplication implicitly
    height = n_sources_per_node
    sub_height = n_sources_per_node
    width = n_targets_per_node*n_targets
    sub_width = n_targets_per_node

    dBT = cp.zeros((n_sources_per_node, k)).astype(np.float32)

    # Blocking is transposed
    bpg = (int(np.ceil(width/BLOCK_WIDTH)), int(np.ceil(height/BLOCK_HEIGHT)))

    for idx in range(k):
        implicit_gram_matrix[bpg, TPB](
            dtargets, dsources, dQT, dBT, height, width, sub_height, sub_width, idx
        )

    # Perform SVD on reduced matrix
    du, dS, dVT = cp.linalg.svd(dBT.T, full_matrices=False)
    dU = cp.matmul(dQ, du)

    # Return compressed SVD components
    return dU.get(), dS.get(), dVT.get(), hashes


def compute_surfaces(config, db):
    """
    Compute equivalent and check surfaces, and save to disk.

    Parameters:
    -----------
    config : dict
        Config object, loaded from config.json.
    db : hdf5.File
        HDF5 file handle containing all experimental data.

    Returns:
    --------
    np.array(shape=(n_equivalent, 3), dtype=np.float32)
        Discretised equivalent surface.
    np.array(shape=(n_check, 3), dtype=np.float32)
        Discretised check surface.
    """
    order_equivalent = config['order_equivalent']
    order_check = config['order_check']
    equivalent_surface = surface.compute_surface(order_equivalent)
    check_surface = surface.compute_surface(order_check)

    print(f"Computing Inner Surface of Order {order_equivalent}")
    print(f"Computing Outer Surface of Order {order_check}")

    if 'surface' in db.keys():
        del db['surface']

    db.create_group('surface')

    db['surface']['equivalent'] = equivalent_surface
    db['surface']['check'] = check_surface

    return equivalent_surface, check_surface


def compute_octree(config, db):
    """
    Compute balanced as well as completed octree and all interaction  lists,
        and save to disk.

    Parameters:
    -----------
    config : dict
        Config object, loaded from config.json.
    db : hdf5.File
        HDF5 file handle containing all experimental data.

    Returns:
    --------
    np.array(shape=(1, 3), dtype=np.float64)
        Physical center of octree root node.
    np.float64
        Half side length of octree root node.
    np.int64
        Depth of octree.
    """
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
    db['octree']['depth'] = np.array([depth], np.int32)
    db['octree']['x0'] = np.array([x0], np.float32)
    db['octree']['r0'] = np.array([r0], np.float32)
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

    return x0, r0, depth


def compute_inv_c2e(
        config, db, dense_gram_matrix, equivalent_surface, check_surface, x0, r0
    ):
    """
    Compute inverses of upward and downward check to equivalent Gram matrices
        at the level of the root node, and save to disk.

    Parameters:
    -----------
    config : dict
        Config object, loaded from config.json.
    db : hdf5.File
        HDF5 file handle containing all experimental data.
    dense_gram_matrix : Numba JIT function handle
        Function to calculate dense gram matrix on CPU
    check_surface : np.array(shape=(n_check, 3), dtype=np.float32)
        Discretised check surface.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Physical center of octree root node.
    r0 : np.float64
        Half side length of octree root node.

    Returns:
    --------
    (
        np.array(shape=(n_check, n_equivalent), dtype=np.float64),
        np.array(shape=(n_equivalent, n_check), dtype=np.float64)
    )
        Tuple of downward-check-to-equivalent Gram matrix inverse, and the
        upward-check-to-equivalent Gram matrix inverse.
    """

    print("Computing Inverse of Check To Equivalent Gram Matrix")

    upward_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface,
        radius=np.float32(r0),
        level=np.int32(0),
        center=x0.astype(np.float32),
        alpha=np.float32(config['alpha_inner'])
    )

    upward_check_surface = surface.scale_surface(
        surf=check_surface,
        radius=np.float32(r0),
        level=np.int32(0),
        center=x0.astype(np.float32),
        alpha=np.float32(config['alpha_outer'])
    )

    downward_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface,
        radius=np.float32(r0),
        level=np.int32(0),
        center=x0.astype(np.float32),
        alpha=np.float32(config['alpha_outer'])
    )

    downward_check_surface = surface.scale_surface(
        surf=check_surface,
        radius=np.float32(r0),
        level=np.int32(0),
        center=x0.astype(np.float32),
        alpha=np.float32(config['alpha_inner'])
    )

    uc2e = dense_gram_matrix(
        targets=upward_check_surface,
        sources=upward_equivalent_surface,
    )

    dc2e = dense_gram_matrix(
        targets=downward_check_surface,
        sources=downward_equivalent_surface,
    )

    uc2e_inv = linalg.pinv2(uc2e)
    dc2e_inv = linalg.pinv2(dc2e)

    if 'uc2e_inv' in db.keys() and 'dc2e_inv' in db.keys():

        del db['uc2e_inv']
        del db['dc2e_inv']

    db['uc2e_inv']= uc2e_inv
    db['dc2e_inv']= dc2e_inv

    return uc2e_inv, dc2e_inv


def compute_m2m_and_l2l(
        config, db, equivalent_surface, check_surface, dense_gram_matrix,
        kernel_scale, uc2e_inv, dc2e_inv, parent_center, parent_radius
    ):
    """
    Compute M2M and L2L operators at level of root node, and its children, and
        save to disk.

    config : dict
        Config object, loaded from config.json.
    db : hdf5.File
        HDF5 file handle containing all experimental data.
    equivalent_surface: np.array(shape=(n_equivalent, 3), dtype=np.float32)
        Discretised equivalent surface.
    check_surface : np.array(shape=(n_check, 3), dtype=np.float32)
        Discretised check surface.
    dense_gram_matrix : Numba JIT function handle
        Function to calculate dense gram matrix on CPU.
    kernel_scale : Numba JIT function handle
        Function to calculate the scale of the kernel for a given level.
    uc2e_inv : np.array(shape=(n_check, n_equivalent), dtype=np.float64)
    dc2e_inv : np.array(shape=(n_equivalent, n_check), dtype=np.float64)
    parent_center : np.array(shape=(3,), dtype=np.float64)
        Operators are calculated wrt to root node, so corresponds to x0.
    parent_radius : np.float64
        Operators are calculated wrt to root node, so corresponds to r0.
    """

    parent_level = 0
    child_level = 1

    child_centers = [
        morton.find_physical_center_from_key(child, parent_center, parent_radius)
        for child in morton.find_children(0)
    ]

    parent_upward_check_surface = surface.scale_surface(
        surf=check_surface,
        radius=np.float32(parent_radius),
        level=np.int32(parent_level),
        center=parent_center.astype(np.float32),
        alpha=np.float32(config['alpha_outer'])
    )

    parent_downward_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface,
        radius=np.float32(parent_radius),
        level=np.int32(parent_level),
        center=parent_center.astype(np.float32),
        alpha=np.float32(config['alpha_outer'])
    )

    m2m = []
    l2l = []

    # Calculate scale
    scale = kernel_scale(child_level)

    print("Computing M2M & L2L Operators")

    loading = len(child_centers)
    for child_idx, child_center in enumerate(child_centers):
        print(f'Computed ({child_idx+1}/{loading}) M2M/L2L operators')

        child_upward_equivalent_surface = surface.scale_surface(
            surf=equivalent_surface,
            radius=np.float32(parent_radius),
            level=np.int32(child_level),
            center=child_center.astype(np.float32),
            alpha=np.float32(config['alpha_inner'])
        )

        child_downward_check_surface = surface.scale_surface(
            surf=check_surface,
            radius=np.float32(parent_radius),
            level=np.int32(child_level),
            center=child_center.astype(np.float32),
            alpha=np.float32(config['alpha_inner'])
        )

        pc2ce = dense_gram_matrix(
            targets=parent_upward_check_surface,
            sources=child_upward_equivalent_surface,
        )

        # Compute M2M operator for this octant
        m2m.append(np.matmul(uc2e_inv, pc2ce))

        # Compute L2L operator for this octant
        cc2pe = dense_gram_matrix(
            targets=child_downward_check_surface,
            sources=parent_downward_equivalent_surface
        )

        l2l.append(np.matmul(scale*dc2e_inv, cc2pe))

    # Save M2M & L2L operators
    m2m = np.array(m2m)
    l2l = np.array(l2l)

    if 'm2m' in db.keys() and 'l2l' in db.keys():
        del db['m2m']
        del db['l2l']

    db['m2m'] = m2m
    db['l2l'] = l2l


def compute_m2l(
        config, db, implicit_gram_matrix, k,  equivalent_surface, check_surface,
        x0, r0, depth
    ):
    """
    Compute RSVD compressed Gram matrices for M2L operators, and save to disk.

    Parameters:
    -----------
    config : dict
        Config object, loaded from config.json.
    db : hdf5.File
        HDF5 file handle containing all experimental data.
    implicit_gram_matrix : CUDA JIT function handle.
        Function to apply gram matrix implicitly to a given RHS.
    k : np.int64
        Target SVD compression rank of M2L matrix.
    equivalent_surface: np.array(shape=(n_equivalent, 3), dtype=np.float64)
        Discretised equivalent surface.
    check_surface : np.array(shape=(n_check, 3), dtype=np.float64)
        Discretised check surface.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Physical center of octree root node.
    r0 : np.float64
        Half side length of octree root node.
    depth : np.int64
        Depth of octree.
    """

    if 'm2l' in db.keys():
        del db['m2l']

    else:
        db.create_group('m2l')

    group = db['m2l']

    progress = 0

    for level in range(2, depth+1):

        str_level = str(level)

        if str_level in group.keys():
            del db['m2l'][str_level]

        else:
            group.create_group(str_level)

        u, s, vt, hashes = compress_m2l_gram_matrix(
            level, x0, r0, depth, config['alpha_inner'], check_surface,
            equivalent_surface, k, implicit_gram_matrix
        )

        db['m2l'][str_level]['u'] = u
        db['m2l'][str_level]['s'] = s
        db['m2l'][str_level]['vt'] = vt
        db['m2l'][str_level]['hashes'] = hashes.astype(np.int64)

        progress += 1

        print(f'Computed operators for ({progress}/{depth-1}) M2L Levels')


def main(**config):
    """
    Main script, configure using config.json file in module root.
    """
    start = time.time()

    # Step 0: Construct Octree and load Python config objs
    db = h5py.File(WORKING_DIR / f"{config['experiment']}.hdf5", 'a')
    x0, r0, depth = compute_octree(config, db)

    # Required config, not explicitly passed
    kernel = config['kernel']
    k = config['target_rank']

    implicit_gram_matrix = KERNELS[kernel]['implicit_gram_blocked']
    dense_gram_matrix = KERNELS[kernel]['dense_gram']
    kernel_scale = KERNELS[kernel]['scale']

    # Step 1: Compute a surface of a given order
    equivalent_surface, check_surface = compute_surfaces(config, db)

    # Step 2: Use surfaces to compute inverse of check to equivalent Gram matrix.
    # This is a useful quantity that will form the basis of most operators.
    uc2e_inv, dc2e_inv = compute_inv_c2e(
        config, db, dense_gram_matrix, equivalent_surface, check_surface,
        x0, r0
    )

    # Step 3: Compute M2M/L2L operators
    compute_m2m_and_l2l(
        config, db, equivalent_surface, check_surface, dense_gram_matrix,
        kernel_scale, uc2e_inv, dc2e_inv, x0, r0
    )

    # Step 4: Compute M2L operators for each level, and their transfer vectors
    compute_m2l(
        config, db, implicit_gram_matrix, k,  equivalent_surface, check_surface,
        x0, r0, depth
    )

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
