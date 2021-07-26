"""
Precompute and store M2M/L2L/M2L operators for a given points dataset.
"""
import os
import pathlib
import sys
import time

import h5py
import numba
import numpy as np
from sklearn.utils.extmath import randomized_svd

import adaptoctree.morton as morton
import adaptoctree.tree as tree

from fmm.dtype import NUMPY
from fmm.kernel import KERNELS
import fmm.linalg as linalg
import fmm.surface as surface

import utils.data as data
import utils.time

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent
WORKING_DIR = pathlib.Path(os.getcwd())


def compress_m2l_gram_matrix(
        dtype, level, x0, r0, depth, alpha_inner, check_surface,
        equivalent_surface, k, dense_gram
    ):
    """
    Compute compressed representation of unique Gram matrices for targets and
    sources at a given level of the octree, specified by their unique transfer
    vectors. Compression is computed using the randomised-SVD of Halko et. al.
    (2011).

    Parameters:
    -----------
    level : int
        Octree level at which M2L operators are being calculated.
    x0 : np.array(shape=(1, 3), dtype=float)
        Physical center of octree root node.
    r0 : float
        Half side length of octree root node.
    alpha_inner : float
        Relative size of inner surface
    check_surface : np.array(shape=(n_check, 3), dtype=float)
        Discretised check surface.
    equivalent_surface: np.array(shape=(n_equivalent, 3), dtype=float)
        Discretised equivalent surface.
    k : int
        Target compression rank.

    Returns:
    --------
    (
        np.array((nu, k), float),
        np.array(ns, dtype=float),
        np.array((k, k), dtype=float)
    )
        SVD of Gram matrices corresponding to all unique transfer vectors for
        each level. For a given level, these are indexed by the hash of the
        transfer vectors.
    """
    sources, targets, hashes = tree.find_unique_v_list_interactions(
        level=level, x0=x0, r0=r0, depth=depth
    )

    n_targets_per_node = len(check_surface)
    n_sources_per_node = len(equivalent_surface)
    n_sources = len(sources)

    se2tc = np.zeros(
        shape=(n_targets_per_node, n_sources*n_sources_per_node),
        dtype=dtype
    )

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

        lidx_sources = idx*n_sources_per_node
        ridx_sources = lidx_sources+n_sources_per_node

        target_check_surface = surface.scale_surface(
            surf=check_surface,
            radius=r0,
            level=level,
            center=target_center,
            alpha=alpha_inner
        )

        source_equivalent_surface = surface.scale_surface(
            surf=equivalent_surface,
            radius=r0,
            level=level,
            center=source_center,
            alpha=alpha_inner
        )

        se2tc[:, lidx_sources:ridx_sources] = dense_gram(
            sources=source_equivalent_surface,
            targets=target_check_surface
        )

    u, s, vt = randomized_svd(se2tc, k)

    return u.astype(dtype), s.astype(dtype), vt.astype(dtype), hashes


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
    np.array(shape=(n_equivalent, 3), dtype=float)
        Discretised equivalent surface.
    np.array(shape=(n_check, 3), dtype=float)
        Discretised check surface.
    """
    order_equivalent = config['order_equivalent']
    order_check = config['order_check']
    precision = config['precision']
    dtype = NUMPY[precision]
    equivalent_surface = surface.compute_surface(order_equivalent, dtype)
    check_surface = surface.compute_surface(order_check, dtype)

    print(f"Computing Inner Surface of Order {order_equivalent} at {dtype} precision")
    print(f"Computing Outer Surface of Order {order_check} at {dtype} precision")

    if 'surface' in db.keys():
        del db['surface']

    db.create_group('surface')

    db['surface']['equivalent'] = equivalent_surface
    db['surface']['check'] = check_surface

    return equivalent_surface, check_surface


@numba.njit(cache=True)
def compute_index_pointer(keys, points_indices):
    """
    Compute index pointers for argsorted list of keys.

    Parameters:
    -----------
    keys : np.int64
        Keys, not necessarily sorted.
    points_indices : np.array(np.int64)
        Indices that will sort the keys.

    Returns:
    --------
    np.array(np.int64)
        Index pointer for sorted keys.
    """
    sorted_keys = keys[points_indices]

    curr_idx = 0
    curr_key = sorted_keys[0]

    nkeys = len(sorted_keys)

    index_pointer = [curr_idx]

    for idx in range(nkeys):
        next_key = sorted_keys[idx]
        if next_key != curr_key:
            index_pointer.append(idx)
        curr_key = next_key

    index_pointer.append(nkeys)

    return np.array(index_pointer)


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
    np.array(shape=(1, 3), dtype=float)
        Physical center of octree root node.
    float
        Half side length of octree root node.
    int
        Depth of octree.
    """
    max_level = config['max_level']
    max_points = config['max_points']
    precision = config['precision']
    dtype = NUMPY[precision]
    start_level = 1

    # AdaptOctree restricted to double precision, hence cast for compatibility
    sources = db['particle_data']['sources'][...].astype(np.float64)
    targets = db['particle_data']['targets'][...].astype(np.float64)
    points = np.vstack((sources, targets))

    print("Computing octree")
    max_bound, min_bound = morton.find_bounds(points)
    x0 = morton.find_center(max_bound, min_bound)
    r0 = morton.find_radius(max_bound, min_bound)

    unbalanced = tree.build(targets, max_level, max_points, start_level)
    u_depth = tree.find_depth(unbalanced)
    octree = tree.balance(unbalanced, u_depth)
    octree = np.sort(octree, kind='stable')
    depth = tree.find_depth(octree)
    sources_to_keys = tree.points_to_keys(sources, octree, depth, x0, r0)
    targets_to_keys = tree.points_to_keys(targets, octree, depth, x0, r0)

    # Overwrite octree with non-empty leaf nodes
    octree = np.unique(np.hstack((sources_to_keys, targets_to_keys)))
    complete = tree.complete_tree(octree)
    complete = np.sort(complete, kind='stable')
    u, x, v, w = tree.find_interaction_lists(octree, complete, depth)

    # Impose order
    source_indices = np.argsort(sources_to_keys, kind='stable')
    source_index_pointer = compute_index_pointer(sources_to_keys, source_indices)
    target_indices = np.argsort(targets_to_keys, kind='stable')
    target_index_pointer = compute_index_pointer(targets_to_keys, target_indices)

    if 'octree' in db.keys():
        del db['octree']['keys']
        del db['octree']['depth']
        del db['octree']['x0']
        del db['octree']['r0']
        del db['octree']['complete']
        del db['particle_data']['sources_to_keys']
        del db['particle_data']['source_indices']
        del db['particle_data']['source_index_pointer']
        del db['particle_data']['targets_to_keys']
        del db['particle_data']['target_indices']
        del db['particle_data']['target_index_pointer']
        del db['interaction_lists']['u']
        del db['interaction_lists']['x']
        del db['interaction_lists']['v']
        del db['interaction_lists']['w']

    else:
        db.create_group('octree')
        db.create_group('interaction_lists')

    db['octree']['keys'] = np.sort(octree)
    db['octree']['complete'] = np.sort(complete)
    db['octree']['depth'] = np.array([depth])
    db['octree']['x0'] = np.array([x0])
    db['octree']['r0'] = np.array([r0])

    # Save source to index mappings
    db['particle_data']['sources_to_keys'] = sources_to_keys
    db['particle_data']['source_indices'] = source_indices
    db['particle_data']['source_index_pointer'] = source_index_pointer
    db['particle_data']['targets_to_keys'] = targets_to_keys
    db['particle_data']['target_indices'] = target_indices
    db['particle_data']['target_index_pointer'] = target_index_pointer

    # Save interaction lists
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
    check_surface : np.array(shape=(n_check, 3), dtype=float)
        Discretised check surface.
    x0 : np.array(shape=(1, 3), dtype=float)
        Physical center of octree root node.
    r0 : float
        Half side length of octree root node.

    Returns:
    --------
    (
        np.array(shape=(n_check, n_equivalent), dtype=float),
        np.array(shape=(n_equivalent, n_check), dtype=float)
    )
        Tuple of downward-check-to-equivalent Gram matrix inverse, and the
        upward-check-to-equivalent Gram matrix inverse.
    """

    print("Computing Inverse of Check To Equivalent Gram Matrix")

    order_e = config['order_equivalent']
    order_c = config['order_check']
    precision = config['precision']
    alpha_inner = config['alpha_inner']
    alpha_outer = config['alpha_outer']
    max_level = config['max_level']
    dtype = NUMPY[precision]

    equivalent_surface = surface.compute_surface(order_e, dtype)
    check_surface = surface.compute_surface(order_c, dtype)
    level = 0

    upward_equivalent_surfaces = []
    upward_check_surfaces = []
    downward_equivalent_surfaces = []
    downward_check_surfaces = []

    for level in range(0, max_level+1):

        upward_equivalent_surfaces.append(
            surface.scale_surface(
                surf=equivalent_surface,
                radius=r0,
                level=level,
                center=x0,
                alpha=alpha_inner
            )
        )

        upward_check_surfaces.append(
            surface.scale_surface(
                surf=check_surface,
                radius=r0,
                level=level,
                center=x0,
                alpha=alpha_outer
            )
        )

        downward_equivalent_surfaces.append(
            surface.scale_surface(
                surf=equivalent_surface,
                radius=r0,
                level=level,
                center=x0,
                alpha=alpha_outer
            )
        )

        downward_check_surfaces.append(
            surface.scale_surface(
                surf=check_surface,
                radius=r0,
                level=level,
                center=x0,
                alpha=alpha_inner
            )
        )

    uc2e = dense_gram_matrix(
        targets=upward_check_surfaces[0],
        sources=upward_equivalent_surfaces[0],
    )

    dc2e = dense_gram_matrix(
        targets=downward_check_surfaces[0],
        sources=downward_equivalent_surfaces[0],
    )

    uc2e_inv_a, uc2e_inv_b = linalg.pinv2(uc2e)
    dc2e_inv_a, dc2e_inv_b = linalg.pinv2(dc2e)

    if 'uc2e_inv' in db.keys() and 'dc2e_inv' in db.keys():
        del db['uc2e']
        del db['dc2e']
        del db['uc2e_inv_a']
        del db['uc2e_inv_b']
        del db['dc2e_inv_a']
        del db['dc2e_inv_b']

        for level in range(0, max_level+1):
            str_level = str(level)
            del db['surfaces'][str_level]['upward_equivalent_surface']
            del db['surfaces'][str_level]['upward_check_surface']
            del db['surfaces'][str_level]['downward_equivalent_surface']
            del db['surfaces'][str_level]['downward_check_surface']

    db['uc2e'] = uc2e
    db['dc2e'] = dc2e
    db['uc2e_inv_a'] = uc2e_inv_a
    db['uc2e_inv_b'] = uc2e_inv_b
    db['dc2e_inv_a'] = dc2e_inv_a
    db['dc2e_inv_b'] = dc2e_inv_b

    db.create_group('surfaces')

    group = db['surfaces']
    for level in range(0, max_level+1):
        str_level = str(level)

        group.create_group(str_level)

        group[str_level]['upward_equivalent_surface'] = upward_equivalent_surfaces[level]
        group[str_level]['upward_check_surface'] = upward_check_surfaces[level]
        group[str_level]['downward_equivalent_surface'] = downward_equivalent_surfaces[level]
        group[str_level]['downward_check_surface'] = downward_check_surfaces[level]

    return uc2e_inv_a, uc2e_inv_b, dc2e_inv_a, dc2e_inv_b


def compute_m2m_and_l2l(
        config, db, equivalent_surface, check_surface, dense_gram_matrix,
        kernel_scale, uc2e_inv_a, uc2e_inv_b, dc2e_inv_a,
        dc2e_inv_b, parent_center, parent_radius
    ):
    """
    Compute M2M and L2L operators at level of root node, and its children, and
        save to disk.

    config : dict
        Config object, loaded from config.json.
    db : hdf5.File
        HDF5 file handle containing all experimental data.
    equivalent_surface: np.array(shape=(n_equivalent, 3), dtype=float)
        Discretised equivalent surface.
    check_surface : np.array(shape=(n_check, 3), dtype=float)
        Discretised check surface.
    dense_gram_matrix : Numba JIT function handle
        Function to calculate dense gram matrix on CPU.
    kernel_scale : Numba JIT function handle
        Function to calculate the scale of the kernel for a given level.
    uc2e_inv : np.array(shape=(n_check, n_equivalent), dtype=float)
    dc2e_inv : np.array(shape=(n_equivalent, n_check), dtype=float)
    parent_center : np.array(shape=(3,), dtype=float)
        Operators are calculated wrt to root node, so corresponds to x0.
    parent_radius : float
        Operators are calculated wrt to root node, so corresponds to r0.
    """

    alpha_outer = config['alpha_outer']
    alpha_inner = config['alpha_inner']
    precision = config['precision']
    dtype = NUMPY[precision]

    parent_level = 0
    child_level = 1

    child_centers = [
        morton.find_physical_center_from_key(child, parent_center, parent_radius)
        for child in morton.find_children(0)
    ]

    parent_upward_check_surface = surface.scale_surface(
        surf=check_surface,
        radius=parent_radius,
        level=parent_level,
        center=parent_center,
        alpha=alpha_outer
    )

    parent_downward_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface,
        radius=parent_radius,
        level=parent_level,
        center=parent_center,
        alpha=alpha_outer
    )

    m2m = []
    l2l = []

    # Calculate scale
    scale = dtype(kernel_scale(child_level))

    print("Computing M2M & L2L Operators")

    loading = len(child_centers)
    for child_idx, child_center in enumerate(child_centers):
        print(f'Computed ({child_idx+1}/{loading}) M2M/L2L operators')

        child_upward_equivalent_surface = surface.scale_surface(
            surf=equivalent_surface,
            radius=parent_radius,
            level=child_level,
            center=child_center,
            alpha=alpha_inner
        )

        child_downward_check_surface = surface.scale_surface(
            surf=check_surface,
            radius=parent_radius,
            level=child_level,
            center=child_center,
            alpha=alpha_inner
        )

        pc2ce = dense_gram_matrix(
            targets=parent_upward_check_surface,
            sources=child_upward_equivalent_surface,
        )

        # Compute M2M operator for this octant
        m2m.append(uc2e_inv_a @ (uc2e_inv_b @ pc2ce))

        # Compute L2L operator for this octant
        cc2pe = dense_gram_matrix(
            targets=child_downward_check_surface,
            sources=parent_downward_equivalent_surface
        )

        l2l.append(scale*(dc2e_inv_a @ (dc2e_inv_b @ cc2pe)))

    # Save M2M & L2L operators
    m2m = np.array(m2m)
    l2l = np.array(l2l)

    if 'm2m' in db.keys() and 'l2l' in db.keys():
        del db['m2m']
        del db['l2l']

    db['m2m'] = m2m
    db['l2l'] = l2l


def compute_m2l(
        config, db, k,  equivalent_surface, check_surface, x0, r0, depth
    ):
    """
    Compute RSVD compressed Gram matrices for M2L operators, and save to disk.

    Parameters:
    -----------
    config : dict
        Config object, loaded from config.json.
    db : hdf5.File
        HDF5 file handle containing all experimental data.
        Function to apply gram matrix implicitly to a given RHS.
    k : np.int64
        Target SVD compression rank of M2L matrix.
    equivalent_surface: np.array(shape=(n_equivalent, 3), dtype=float)
        Discretised equivalent surface.
    check_surface : np.array(shape=(n_check, 3), dtype=float)
        Discretised check surface.
    x0 : np.array(shape=(1, 3), dtype=float)
        Physical center of octree root node.
    r0 : float
        Half side length of octree root node.
    depth : int
        Depth of octree.
    """
    dense_gram = KERNELS[config['kernel']]['dense_gram']
    dtype = NUMPY[config['precision']]
    alpha_inner = config['alpha_inner']

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
            dtype, level, x0, r0, depth, alpha_inner, check_surface,
            equivalent_surface, k, dense_gram
        )

        db['m2l'][str_level]['u'] = u
        db['m2l'][str_level]['s'] = s
        db['m2l'][str_level]['vt'] = vt
        db['m2l'][str_level]['hashes'] = hashes

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

    # implicit_gram_matrix = KERNELS[kernel]['implicit_gram_blocked']
    dense_gram_matrix = KERNELS[kernel]['dense_gram']
    kernel_scale = KERNELS[kernel]['scale']

    # Step 1: Compute a surface of a given order
    equivalent_surface, check_surface = compute_surfaces(config, db)

    # Step 2: Use surfaces to compute inverse of check to equivalent Gram matrix.
    # This is a useful quantity that will form the basis of most operators.
    uc2e_inv_a, uc2e_inv_b, dc2e_inv_a, dc2e_inv_b = compute_inv_c2e(
        config, db, dense_gram_matrix, equivalent_surface, check_surface,
        x0, r0
    )

    # Step 3: Compute M2M/L2L operators
    compute_m2m_and_l2l(
        config, db, equivalent_surface, check_surface, dense_gram_matrix,
        kernel_scale, uc2e_inv_a, uc2e_inv_b, dc2e_inv_a, dc2e_inv_b, x0, r0
    )

    # Step 4: Compute M2L operators for each level, and their transfer vectors
    compute_m2l(
        config, db, k,  equivalent_surface, check_surface,
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
