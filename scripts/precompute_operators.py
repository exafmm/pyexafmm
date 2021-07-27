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
from sklearn.utils.extmath import *
import scipy.linalg.lapack as lapack

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


def rsvd(M, n_components, *, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=0):
    """Computes a truncated randomized SVD.
    Parameters
    ----------
    M : {ndarray, sparse matrix}
        Matrix to decompose.
    n_components : int
        Number of singular values and vectors to extract.
    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.
        .. versionchanged:: 0.18
    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.
        .. versionadded:: 0.18
    transpose : bool or 'auto', default='auto'
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.
        .. versionchanged:: 0.18
    flip_sign : bool, default=True
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.
    random_state : int, RandomState instance or None, default=0
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).
    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 https://arxiv.org/abs/0909.4061
    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    if isinstance(M, (sparse.lil_matrix, sparse.dok_matrix)):
        warnings.warn("Calculating SVD of a {} is expensive. "
                      "csr_matrix is more efficient.".format(
                          type(M).__name__),
                      sparse.SparseEfficiencyWarning)

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    # Get datatype
    dtype = M.dtype.type

    Q = randomized_range_finder(
        M, size=n_random, n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_sparse_dot(Q.T, M)

    # compute the SVD on the thin matrix: (k + p) wide
    if dtype == np.float32:
        Uhat, s, Vt, _ = lapack.sgesvd(B, full_matrices=0)
    elif dtype == np.float64:
        Uhat, s, Vt, _ = lapack.dgesvd(B, full_matrices=0)

    del B
    U = np.dot(Q, Uhat)

    if flip_sign:
        if not transpose:
            U, Vt = svd_flip(U, Vt)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], Vt[:n_components, :]


def compress_m2l_gram_matrix(
        dense_gram_matrix, level, x0, r0, depth, alpha_inner, check_surface,
        equivalent_surface, k, dtype
    ):
    """
    Compute compressed representation of unique Gram matrices for targets and
    sources at a given level of the octree, specified by their unique transfer
    vectors. Compression is computed using the randomised-SVD of Halko et. al.
    (2011).

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
    check_surface : np.array(shape=(n_check, 3), dtype=float)
        Discretised check surface.
    equivalent_surface: np.array(shape=(n_equivalent, 3), dtype=float)
        Discretised equivalent surface.
    k : np.int32
        Target compression rank.

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
        level=level, x0=x0, r0=r0, depth=depth
    )

    n_targets_per_node = len(check_surface)
    n_sources_per_node = len(equivalent_surface)
    n_sources = len(sources)

    se2tc = np.zeros((n_targets_per_node, n_sources*n_sources_per_node), dtype)

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

        se2tc[:, lidx_sources:ridx_sources] =  dense_gram_matrix(
                sources=source_equivalent_surface, targets=target_check_surface
            )

    u, s, vt = rsvd(se2tc, k)

    return u, s, vt, hashes


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
    dtype = NUMPY[config['precision']]
    equivalent_surface = surface.compute_surface(order_equivalent, dtype)
    check_surface = surface.compute_surface(order_check, dtype)

    print(f"Computing Inner Surface of Order {order_equivalent}")
    print(f"Computing Outer Surface of Order {order_check}")

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

    # AdaptOctree restricted to double precision, cast for compatibility
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
    db['octree']['depth'] = np.array([depth])
    db['octree']['x0'] = np.array([x0])
    db['octree']['r0'] = np.array([r0])
    db['octree']['complete'] = np.sort(complete)

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
    check_surface : np.array(shape=(n_check, 3), dtype=np.float32)
        Discretised check surface.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Physical center of octree root node.
    r0 : np.float64
        Half side length of octree root node.

    Returns:
    --------
    (
        np.array(shape=(n_check, n_s), dtype=np.float64),
        np.array(shape=(n_s, n_equivalent), dtype=np.float64),
        np.array(shape=(n_check, n_s), dtype=np.float64),
        np.array(shape=(n_s, n_check), dtype=np.float64)
    )
        Tuple of downward-check-to-equivalent Gram matrix inverse, and the
        upward-check-to-equivalent Gram matrix inverse, returned in two
        components.
    """

    print("Computing Inverse of Check To Equivalent Gram Matrix")

    dtype = NUMPY[config['precision']]
    alpha_inner = config['alpha_inner']
    alpha_outer = config['alpha_outer']
    level = 0

    equivalent_surface = surface.compute_surface(config['order_equivalent'], dtype)
    check_surface = surface.compute_surface(config['order_check'], dtype)

    upward_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface,
        radius=r0,
        level=level,
        center=x0,
        alpha=alpha_inner
    )

    upward_check_surface = surface.scale_surface(
        surf=check_surface,
        radius=r0,
        level=level,
        center=x0,
        alpha=alpha_outer
    )

    downward_equivalent_surface = surface.scale_surface(
        surf=equivalent_surface,
        radius=r0,
        level=level,
        center=x0,
        alpha=alpha_outer
    )

    downward_check_surface = surface.scale_surface(
        surf=check_surface,
        radius=r0,
        level=level,
        center=x0,
        alpha=alpha_inner
    )

    uc2e = dense_gram_matrix(
        targets=upward_check_surface,
        sources=upward_equivalent_surface,
    )

    dc2e = dense_gram_matrix(
        targets=downward_check_surface,
        sources=downward_equivalent_surface,
    )

    uc2e_inv = linalg.pinv(uc2e)
    uc2e_inv_a, uc2e_inv_b = linalg.pinv2(uc2e)

    dc2e_inv = linalg.pinv(dc2e)
    dc2e_inv_a, dc2e_inv_b = linalg.pinv2(dc2e)

    if 'uc2e_inv' in db.keys() and 'dc2e_inv' in db.keys():
        del db['uc2e']
        del db['dc2e']
        del db['uc2e_inv']
        del db['uc2e_inv_a']
        del db['uc2e_inv_b']
        del db['dc2e_inv']
        del db['dc2e_inv_a']
        del db['dc2e_inv_b']

    db['uc2e'] = uc2e
    db['dc2e'] = dc2e
    db['uc2e_inv']= uc2e_inv
    db['uc2e_inv_a']= uc2e_inv_a
    db['uc2e_inv_b']= uc2e_inv_b
    db['dc2e_inv']= dc2e_inv
    db['dc2e_inv_a']= dc2e_inv_a
    db['dc2e_inv_b']= dc2e_inv_b

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

    dtype = NUMPY[config['precision']]
    alpha_inner = config['alpha_inner']
    alpha_outer = config['alpha_outer']

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
        config, db, k,  equivalent_surface, check_surface,
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
    equivalent_surface: np.array(shape=(n_equivalent, 3), dtype=float)
        Discretised equivalent surface.
    check_surface : np.array(shape=(n_check, 3), dtype=float)
        Discretised check surface.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Physical center of octree root node.
    r0 : np.float64
        Half side length of octree root node.
    depth : np.int64
        Depth of octree.
    """

    alpha_inner = config['alpha_inner']
    dtype = NUMPY[config['precision']]
    dense_gram_matrix = KERNELS[config['kernel']]['dense_gram']

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
            dense_gram_matrix, level, x0, r0, depth, alpha_inner, check_surface,
            equivalent_surface, k, dtype
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
