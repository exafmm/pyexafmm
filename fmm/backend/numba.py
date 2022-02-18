"""
Compute operators, accelerated with Numba.
"""
import numba
import numpy as np

import adaptoctree.morton as morton

import fmm.surface as surface


@numba.njit(cache=True, parallel=True)
def prepare_p2m_data(
        leaves,
        nleaves,
        sources,
        source_densities,
        source_index_pointer,
        key_to_leaf_index,
        x0,
        r0,
        alpha_outer,
        check_surface,
        ncheck_points,
        p2p_function,
        scale_function,
        dtype,
    ):
    """
    Create aligned vector of scales, and check potentials indexed by leaves.
        This maximizes cache re-use in the application of the P2M operator.

    Parameters:
    -----------
    leaves: np.array(shape=(nleaves), dtype=np.int64)
        Octree leaves.
    nleaves : int
        Number of leaves.
    sources : np.array(shape=(nsources, 3), dtype=float)
        All source coordinates.
    source_densities : np.array(shape=(nsources), dtype=float)
        Densities at source coordinates.
    source_index_pointer : np.array(shape=(nleaves+1), dtype=int)
    key_to_to_leaf_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Center of octree root node.
    r0 : np.float64
        Half side length of octree root node.
    alpha_outer : float
        Relative size of outer surface.
    check_surface : np.array(shape=(ncheck_points, 3), dtype=float)
        Discretized check surface.
    ncheck_points : int
        Number of quadrature points on check_surface.
    p2p_function : function
        Serial P2P function handle for this kernel.
    scale_function : function
        Scale function handle for this kernel.
    dtype : type
        Corresponds to precision of experiment ∈ {np.float32, np.float64}.

    Returns:
    --------
    (np.array(float), np.array(float))
        Tuple of scales and check potentials (ordered by leaf index) respectively.
    """
    scales = np.zeros(shape=nleaves, dtype=dtype)
    check_potentials = np.zeros(shape=(nleaves*ncheck_points), dtype=dtype)

    for thread_idx in numba.prange(nleaves):

        leaf = leaves[thread_idx]

        # Lookup leaf sources, and leaf source densities
        leaf_idx = key_to_leaf_index[leaf]
        leaf_sources = sources[source_index_pointer[leaf_idx]:source_index_pointer[leaf_idx+1]]
        leaf_source_densities = source_densities[source_index_pointer[leaf_idx]:source_index_pointer[leaf_idx+1]]

        # Compute center of leaf box in cartesian coordinates
        leaf_center = morton.find_physical_center_from_key(
            key=leaf, x0=x0, r0=r0
        )

        leaf_level = morton.find_level(leaf)

        upward_check_surface = surface.scale_surface(
            surf=check_surface,
            radius=r0,
            level=leaf_level,
            center=leaf_center,
            alpha=alpha_outer,
        )

        check_potential = p2p_function(
            targets=upward_check_surface,
            sources=leaf_sources,
            source_densities=leaf_source_densities,
        )

        lidx = thread_idx*ncheck_points
        ridx = lidx+ncheck_points
        check_potentials[lidx:ridx] += check_potential
        scales[thread_idx] += scale_function(leaf_level)

    return scales, check_potentials


@numba.njit(cache=True, parallel=True)
def p2m_core(
        leaves,
        nleaves,
        key_to_index,
        nequivalent_points,
        ncheck_points,
        uc2e_inv_a,
        uc2e_inv_b,
        scales,
        multipole_expansions,
        check_potentials
    ):
    """
    Parallelized loop applying P2M operator over all leaves.

    Parameters:
    -----------
    leaves: np.array(shape=(nleaves), dtype=np.int64)
        Octree leaves.
    nleaves : int
        Number of leaves.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
	    Map from key to global index.
    nequivalent_points : int
	    Number of quadrature points on equivalent_surface.
    ncheck_points : int
        Number of quadrature points on check_surface.
    uc2e_inv_a : np.array(dtype=float)
	    First component of inverse of upward check to equivalent Gram matrix.
    uc2e_inv_b : np.array(dtype=float)
    	Second component of inverse of upward check to equivalent Gram matrix.
    scales : np.array(shape=(nleaves), dtype=float)
        Scales calculated in `prepare_p2m_data` function
    multipole_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Multipole expansions, aligned by global index from `key_to_index`.
    check_potentials : np.array(shape=(nleaves*ncheck_points), dtype=float)
        Check potentials calculated in `prepare_p2p_data` function.
    """
    for i in numba.prange(nleaves):

        scale = scales[i]

        check_lidx = i*ncheck_points
        check_ridx = check_lidx+ncheck_points
        check_potential = check_potentials[check_lidx:check_ridx]

        leaf = leaves[i]
        lidx = key_to_index[leaf]*nequivalent_points
        ridx = lidx+nequivalent_points

        multipole_expansions[lidx:ridx] += scale*(uc2e_inv_a @ (uc2e_inv_b @ check_potential))


@numba.njit(cache=True)
def p2m(
        leaves,
        nleaves,
        key_to_index,
        key_to_leaf_index,
        sources,
        source_densities,
        source_index_pointer,
        multipole_expansions,
        nequivalent_points,
        x0,
        r0,
        alpha_outer,
        check_surface,
        ncheck_points,
        uc2e_inv_a,
        uc2e_inv_b,
        p2p_function,
        scale_function,
        dtype
    ):
    """
    P2M operator. Compute the multipole expansion from the sources at each
    leaf node.

    Parameters:
    -----------
    leaves: np.array(shape=(nleaves), dtype=np.int64)
        Octree leaves.
    nleaves : int
        Number of leaves.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    key_to_leaf_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    sources : np.array(shape=(nsources, 3), dtype=float)
        All source coordinates.
    source_densities : np.array(shape=(nsources), dtype=float)
        Densities at source coordinates.
    source_index_pointer : np.array(shape=(nleaves+1), dtype=int)
        Index pointer aligning sources by leaf.
    multipole_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Multipole expansions, aligned by global index from `key_to_index`.
    nequivalent_points : int
        Number of quadrature points on equivalent_surface.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Center of octree root node.
    r0 : np.float64
        Half side length of octree root node.
    alpha_outer : float
        Relative size of outer surface.
    check_surface : np.array(shape=(ncheck_points, 3), dtype=float)
        Discretized check surface.
    ncheck_points : int
        Number of quadrature points on check_surface.
    uc2e_inv_a : np.array(dtype=float)
        First component of inverse of upward check to equivalent Gram matrix.
    uc2e_inv_b : np.array(dtype=float)
        Second component of inverse of upward check to equivalent Gram matrix.
    p2p_function : function
        Serial P2P function handle for this kernel.
    scale_function : function
        Scale function handle for this kernel.
    dtype : type
        Corresponds to precision of experiment ∈ {np.float32, np.float64}.
    """

    scales, check_potentials = prepare_p2m_data(
        leaves=leaves,
        nleaves=nleaves,
        sources=sources,
        source_densities=source_densities,
        source_index_pointer=source_index_pointer,
        key_to_leaf_index=key_to_leaf_index,
        x0=x0,
        r0=r0,
        alpha_outer=alpha_outer,
        check_surface=check_surface,
        ncheck_points=ncheck_points,
        p2p_function=p2p_function,
        scale_function=scale_function,
        dtype=dtype
    )

    p2m_core(
        leaves=leaves,
        nleaves=nleaves,
        key_to_index=key_to_index,
        nequivalent_points=nequivalent_points,
        ncheck_points=ncheck_points,
        uc2e_inv_a=uc2e_inv_a,
        uc2e_inv_b=uc2e_inv_b,
        scales=scales,
        multipole_expansions=multipole_expansions,
        check_potentials=check_potentials
    )



@numba.njit(cache=True, parallel=True)
def m2m(
        keys,
        multipole_expansions,
        m2m,
        key_to_index,
        nequivalent_points,
    ):
    """
    M2M operator serially applied over keys in a given level. Parallelization
    doesn't offer much benefit due to blocking writes to the parent multipole
    expansion.

    Notes:
    ------
    As it's not possible to know group siblings contained in the tree apriori
    due to the relative cost of searching the tree for siblings, it's not
    possible to parallelize the operation over groups of siblings.

    Parameters:
    -----------
    keys : np.array(shape=(nkeys), dtype=np.int64)
        All nodes at a given level of the octree.
    multipole_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Multipole expansions, aligned by global index from `key_to_index`.
    m2m : np.array(shape=(8, ncheck_points, nequivalent_points), dtype=float)
	    Precomputed M2M operators.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    nequivalent_points : int
        Number of quadrature points on equivalent_surface.
    """
    nkeys = len(keys)

    for i in numba.prange(nkeys):
        child = keys[i]
        parent = morton.find_parent(child)

        # Compute operator index
        operator_idx = child == morton.find_siblings(child)

        # Get parent and child expansion indices
        key_idx = key_to_index[child]
        child_lidx = key_idx*nequivalent_points
        child_ridx = child_lidx+nequivalent_points

        parent_idx = key_to_index[parent]
        parent_lidx = parent_idx*nequivalent_points
        parent_ridx = parent_lidx+nequivalent_points

        # Add child contribution to parent multipole expansion
        multipole_expansions[parent_lidx:parent_ridx] += (
            m2m[operator_idx][0] @ multipole_expansions[child_lidx:child_ridx]
        )


@numba.njit(cache=True, parallel=False)
def m2l_core(
        key,
        v_list,
        u,
        s,
        vt,
        dc2e_inv_a,
        dc2e_inv_b,
        local_expansions,
        multipole_expansions,
        nequivalent_points,
        key_to_index,
        hash_to_index,
        scale
):
    """
    Application of the M2L operator over the V list of a given node.

    Parameters:
    -----------
    key : np.int64
	    Operator applied to this key.
    v_list : np.array(shape=(nv_list), dtype=np.int64)
	    V list of this key.
    u : np.array(shape=(ncheck_points, k), dtype=float)
        Left singular vectors of M2L matrix for all transfer vectors at this level.
    s : np.array(shape=(k, k), dtype=float)
        Singular values of M2L matrix for all transfer vectors at this level.
    vt : np.array(shape=(k, nequivalent_points*316))
        Right singular values of M2L matrix for all transfer vectors at this level.
    dc2e_inv_a : np.array(dtype=float)
        First component of inverse of downward check to equivalent Gram matrix.
    dc2e_inv_b : np.array(dtype=float)
        Second component of inverse of downward check to equivalent Gram matrix.
    multipole_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Multipole expansions, aligned by global index from `key_to_index`.
    local_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Local expansions, aligned by global index from `key_to_index`.
    nequivalent_points : int
        Number of quadrature points on equivalent_surface.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    hash_to_index : numba.typed.Dict.empty(key_type=numba.types.int64, value_type=numba.types.int64)
        Map between transfer vector hash and index of right singular vector
        components at a given level.
    scale : float
        Kernel scale for keys at this level.
    """
    nv_list = len(v_list)

    # Indices of local expansion
    l_lidx = key_to_index[key]*nequivalent_points
    l_ridx = l_lidx+nequivalent_points

    for i in range(nv_list):

        source = v_list[i]

        # Locd correct components of compressed M2L matrix
        transfer_vector = morton.find_transfer_vector(key, source)
        v_idx = hash_to_index[transfer_vector]
        v_lidx = v_idx*nequivalent_points
        v_ridx = v_lidx+nequivalent_points
        vt_sub = np.copy(vt[:, v_lidx:v_ridx])

        # Indices of multipole expansion
        m_lidx = key_to_index[source]*nequivalent_points
        m_ridx = m_lidx+nequivalent_points

        local_expansions[l_lidx:l_ridx] += scale*(
            dc2e_inv_a @ (
                dc2e_inv_b @ (
                    u @ (s @ (vt_sub @ multipole_expansions[m_lidx:m_ridx]))
                )
            )
        )


@numba.njit(cache=True, parallel=True)
def m2l(
        keys,
        v_lists,
        u,
        s,
        vt,
        dc2e_inv_a,
        dc2e_inv_b,
        multipole_expansions,
        local_expansions,
        nequivalent_points,
        key_to_index,
        hash_to_index,
        scale
    ):
    """
    M2L operator. Parallelized over all keys in a given level.

    Parameters:
    -----------
    keys : np.array(shape=(nkeys), dtype=np.int64)
        All nodes at a given level of the octree.
    v_lists : np.array(shape=(ncomplete, nv_list), dtype=np.int64)
	    All V lists for nodes in octree.
    u : np.array(shape=(ncheck_points, k), dtype=float)
        Left singular vectors of M2L matrix for all transfer vectors at this level.
    s : np.array(shape=(k, k), dtype=float)
        Singular values of M2L matrix for all transfer vectors at this level.
    vt : np.array(shape=(k, nequivalent_points*316))
        Right singular values of M2L matrix for all transfer vectors at this level.
    dc2e_inv_a : np.array(dtype=float)
        First component of inverse of downward check to equivalent Gram matrix.
    dc2e_inv_b : np.array(dtype=float)
        Second component of inverse of downward check to equivalent Gram matrix.
    multipole_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Multipole expansions, aligned by global index from `key_to_index`.
    local_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Local expansions, aligned by global index from `key_to_index`.
    nequivalent_points : int
        Number of quadrature points on equivalent_surface.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    hash_to_index : numba.typed.Dict.empty(key_type=numba.types.int64, value_type=numba.types.int64)
        Map between transfer vector hash and index of right singular vector
        components at a given level.
    scale : float
        Kernel scale for keys at this level.
    """
    nkeys = len(keys)

    for i in numba.prange(nkeys):
        key = keys[i]

        # Pick out the v list
        v_list = v_lists[key_to_index[key]]

        # Filter v list
        v_list = v_list[v_list != -1]
        v_list = v_list[v_list != 0]

        m2l_core(
            key=key,
            v_list=v_list,
            u=u,
            s=s,
            vt=vt,
            dc2e_inv_a=dc2e_inv_a,
            dc2e_inv_b=dc2e_inv_b,
            local_expansions=local_expansions,
            multipole_expansions=multipole_expansions,
            nequivalent_points=nequivalent_points,
            key_to_index=key_to_index,
            hash_to_index=hash_to_index,
            scale=scale
        )


@numba.njit(cache=True)
def l2l_core(
        key,
        local_expansions,
        l2l,
        key_to_index,
        nequivalent_points
     ):
    """
    L2L operator applied to a parent key.

    Parameters:
    -----------
    key : np.int64
        Operator applied to this key.
    local_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
	    Local expansions, aligned by global index from `key_to_index`.
    l2l : np.array(shape=(8, ncheck_points, nequivalent_points), dtype=float)
        Precomputed L2L operators.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    nequivalent_points : int
        Number of quadrature points on equivalent_surface.
    """
    parent = morton.find_parent(key)
    parent_idx = key_to_index[parent]
    parent_lidx = parent_idx*nequivalent_points
    parent_ridx = parent_lidx+nequivalent_points
    parent_equivalent_density = local_expansions[parent_lidx:parent_ridx]

    # Compute expansion index
    child_idx = key_to_index[key]
    child_lidx = child_idx*nequivalent_points
    child_ridx = child_lidx+nequivalent_points

    # Compute operator index
    operator_idx = key == morton.find_siblings(key)

    # Compute contribution to local expansion of child from parent
    local_expansions[child_lidx:child_ridx] += l2l[operator_idx][0] @ parent_equivalent_density


@numba.njit(cache=True, parallel=True)
def l2l(
    keys,
    local_expansions,
    l2l,
    key_to_index,
    nequivalent_points
):
    nkeys = len(keys)
    for i in numba.prange(nkeys):
        key = keys[i]
        l2l_core(key, local_expansions, l2l, key_to_index, nequivalent_points)


@numba.njit(cache=True, parallel=True)
def s2l(
        leaves,
        nleaves,
        sources,
        source_densities,
        source_index_pointer,
        key_to_index,
        key_to_leaf_index,
        x_lists,
        local_expansions,
        x0,
        r0,
        alpha_inner,
        check_surface,
        nequivalent_points,
        dc2e_inv_a,
        dc2e_inv_b,
        scale_function,
        p2p_function,
        dtype
    ):
    """
    S2L operator, parallelized simply over leaves.

    Notes:
    ------
    Parallelization of the S2L operator doesn't maximize cache re-use
    (c.f. P2M, L2T) This decision is taken as X lists are usually small, with
    relatively few nodes having X lists either. This makes the savings due to
    cache re-use competitive with the cost of array allocation.

    Parameters:
    -----------
    leaves: np.array(shape=(nleaves), dtype=np.int64)
        Octree leaves.
    nleaves : int
        Number of leaves.
    sources : np.array(shape=(nsources, 3), dtype=float)
        All source coordinates.
    source_densities : np.array(shape=(nsources), dtype=float)
        Densities at source coordinates.
    source_index_pointer : np.array(shape=(nleaves+1), dtype=int)
        Index pointer aligning sources by leaf.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    key_to_leaf_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    x_lists : np.array(shape=(ncomplete, nx_list), dtype=np.int64)
	    All X lists for nodes in octree.
    local_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Local expansions, aligned by global index from `key_to_index`.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Center of octree root node.
    r0 : np.float64
        Half side length of octree root node.
    alpha_inner : float
        Relative size of inner surface.`
    check_surface : np.array(shape=(ncheck_points, 3), dtype=float)
        Discretized check surface.
    nequivalent_points : int
        Number of quadrature points on equivalent_surface.
    dc2e_inv_a : np.array(dtype=float)
        First component of inverse of downward check to equivalent Gram matrix.
    dc2e_inv_b : np.array(dtype=float)
        Second component of inverse of downward check to equivalent Gram matrix.
    scale_function : function
        Scale function handle for this kernel.
    p2p_function : function
        Serial P2P function handle for this kernel.
    dtype : type
        Corresponds to precision of experiment ∈ {np.float32, np.float64}.
    """
    nleaves = len(leaves)

    for i in numba.prange(nleaves):

        # Pick out leaf
        leaf = leaves[i]

        # Calculate downward check surface
        level = morton.find_level(leaf)
        scale = dtype(scale_function(level))
        center = morton.find_physical_center_from_key(leaf, x0, r0)

        downward_check_surface = surface.scale_surface(
            surf=check_surface,
            radius=r0,
            level=level,
            center=center,
            alpha=alpha_inner
        )

        # Pick out X list
        key_idx = key_to_index[leaf]
        key_lidx = key_idx*nequivalent_points
        key_ridx = key_lidx+nequivalent_points

        x_list = x_lists[key_idx]
        x_list = x_list[x_list != -1]

        # Apply S2L operator over X list
        for source in x_list:

            source_index = key_to_leaf_index[source]
            coordinates = sources[source_index_pointer[source_index]:source_index_pointer[source_index+1]]
            densities = source_densities[source_index_pointer[source_index]:source_index_pointer[source_index+1]]

            downward_check_potential = p2p_function(
                sources=coordinates,
                targets=downward_check_surface,
                source_densities=densities
            )

            local_expansions[key_lidx:key_ridx] += scale*(dc2e_inv_a @ (dc2e_inv_b @ downward_check_potential))


@numba.njit(cache=True, parallel=True)
def m2t(
        leaves,
        nleaves,
        w_lists,
        targets,
        target_index_pointer,
        key_to_index,
        key_to_leaf_index,
        target_potentials,
        multipole_expansions,
        x0,
        r0,
        alpha_inner,
        equivalent_surface,
        nequivalent_points,
        p2p_function,
        gradient_function
    ):
    """
    M2T operator parallelized over leaves.

    Notes:
    ------
    The potential size of W lists make pre-allocation (c.f. L2T, near_field*)
    very expensive for the M2T operator, hence we opt for simple parallelization
    over leaves.

    Parameters:
    -----------
    leaves: np.array(shape=(nleaves), dtype=np.int64)
        Octree leaves.
    nleaves : int
        Number of leaves.
    w_lists : np.array(shape=(ncomplete, nw_list), dtype=np.int64)
        All W lists for nodes in octree.
    targets : np.array(shape=(ntargets, 3), dtype=float)
        All target coordinates.
    target_index_pointer : np.array(shape=(nleaves+1), dtype=int)
        Index pointer aligning targets by leaf.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    key_to_leaf_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    target_potentials : np.array(shape=(ntargets, 4), dtype=float)
        Target potentials (component 0), and x, y, and z components of
        potential gradient (components 1, 2, 3 resp.).
    multipole_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Multipole expansions, aligned by global index from `key_to_index`.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Center of octree root node.
    r0 : np.float64
        Half side length of octree root node.
    alpha_inner : float
        Relative size of inner surface.
    equivalent_surface : np.array(shape=(nequivalent_points, 3), dtype=float)
	    Discretized equivalent surface.
    nequivalent_points : int
	    Number of quadrature points on equivalent_surface.
    p2p_function : function
	    Serial P2P function handle for this kernel.
    grad_function : function
        Serial gradient function handle for this kernel.
    """
    for i in numba.prange(nleaves):
        target_key = leaves[i]
        global_idx = key_to_index[target_key]
        leaf_idx = key_to_leaf_index[target_key]
        w_list = w_lists[global_idx]
        w_list = w_list[w_list != -1]

        # Coordinates of targets within leaf node
        target_coordinates = targets[
            target_index_pointer[leaf_idx]:target_index_pointer[leaf_idx+1]
        ]

        for source in w_list:
            source_idx = key_to_index[source]
            source_lidx = source_idx*nequivalent_points
            source_ridx = source_lidx+nequivalent_points

            source_level = morton.find_level(source)
            source_center = morton.find_physical_center_from_key(source, x0, r0)

            upward_equivalent_surface = surface.scale_surface(
                surf=equivalent_surface,
                radius=r0,
                level=source_level,
                center=source_center,
                alpha=alpha_inner
            )

            target_idx = key_to_leaf_index[target_key]

            target_potentials[target_index_pointer[target_idx]:target_index_pointer[target_idx+1], 0] += p2p_function(
                sources=upward_equivalent_surface,
                targets=target_coordinates,
                source_densities=multipole_expansions[source_lidx:source_ridx]
            )

            target_potentials[target_index_pointer[target_idx]:target_index_pointer[target_idx+1], 1:] += gradient_function(
                sources=upward_equivalent_surface,
                targets=target_coordinates,
                source_densities=multipole_expansions[source_lidx:source_ridx]
            )


@numba.njit(cache=True)
def prepare_l2t_data(
        leaves,
        nleaves,
        targets,
        target_index_pointer,
        equivalent_surface,
        nequivalent_points,
        x0,
        r0,
        alpha_outer,
        key_to_index,
        key_to_leaf_index,
        local_expansions,
        dtype,
    ):
    """
    Prepare L2T data to maximise cache re-use by allocating aligned arrays
    aligning source and target coordinates.

    Parameters:
    -----------
    leaves: np.array(shape=(nleaves), dtype=np.int64)
        Octree leaves.
    nleaves : int
        Number of leaves.
    targets : np.array(shape=(ntargets, 3), dtype=float)
        All target coordinates.
    target_index_pointer : np.array(shape=(nleaves+1), dtype=int)
        Index pointer aligning targets by leaf.
    equivalent_surface : np.array(shape=(nequivalent_points, 3), dtype=float)
        Discretized equivalent surface.
    nequivalent_points : int
        Number of quadrature points on equivalent_surface.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Center of octree root node.
    r0 : np.float64
        Half side length of octree root node.
    alpha_outer : float
        Relative size of outer surface.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    key_to_leaf_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    local_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Local expansions, aligned by global index from `key_to_index`.
    dtype : type
	    Corresponds to precision of experiment ∈ {np.float32, np.float64}.
    """
    local_sources = np.zeros((nequivalent_points*nleaves, 3), dtype=dtype)
    local_source_densities = np.zeros((nequivalent_points*nleaves), dtype=dtype)
    local_targets = np.zeros((nequivalent_points*nleaves, 3), dtype=dtype)
    local_source_index_pointer = np.zeros(nleaves+1, np.int64)
    local_target_index_pointer = np.zeros(nleaves+1, np.int64)

    source_ptr = 0
    target_ptr = 0
    local_source_index_pointer[0] = source_ptr
    local_target_index_pointer[0] = target_ptr

    for i in range(nleaves):
        target = leaves[i]
        level = morton.find_level(target)
        center = morton.find_physical_center_from_key(target, x0, r0)
        target_leaf_index = key_to_leaf_index[target]

        targets_at_node = targets[
            target_index_pointer[target_leaf_index]:target_index_pointer[target_leaf_index+1]
        ]

        sources_at_node = surface.scale_surface(
            equivalent_surface,
            r0,
            level,
            center,
            alpha_outer
        )

        source_idx = key_to_index[target]
        source_lidx = source_idx*nequivalent_points
        source_ridx = source_lidx+nequivalent_points

        source_densities_at_node = local_expansions[source_lidx:source_ridx]

        ntargets_at_node = len(targets_at_node)
        new_target_ptr = target_ptr+ntargets_at_node

        local_targets[target_ptr:new_target_ptr] = targets_at_node
        target_ptr = new_target_ptr

        local_target_index_pointer[i+1] = target_ptr

        nsources_at_node = len(sources_at_node)
        new_source_ptr = source_ptr+nsources_at_node

        local_sources[source_ptr:new_source_ptr] = sources_at_node
        local_source_densities[source_ptr:new_source_ptr] = source_densities_at_node
        source_ptr = new_source_ptr

        local_source_index_pointer[i+1] = source_ptr

    return local_sources, local_targets, local_source_densities, local_source_index_pointer, local_target_index_pointer


@numba.njit(cache=True)
def l2t(
        leaves,
        nleaves,
        key_to_index,
        key_to_leaf_index,
        targets,
        target_potentials,
        target_index_pointer,
        local_expansions,
        x0,
        r0,
        alpha_outer,
        equivalent_surface,
        nequivalent_points,
        p2p_parallel_function,
        dtype
    ):
    """
    Parameters:
    -----------
    leaves: np.array(shape=(nleaves), dtype=np.int64)
        Octree leaves.
    nleaves : int
        Number of leaves.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    key_to_leaf_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    targets : np.array(shape=(ntargets, 3), dtype=float)
        All target coordinates.
    target_potentials : np.array(shape=(ntargets, 4), dtype=float)
        Target potentials (component 0), and x, y, and z components of
        potential gradient (components 1, 2, 3 resp.).
    target_index_pointer : np.array(shape=(nleaves+1), dtype=int)
        Index pointer aligning targets by leaf.
    local_expansions : np.array(shape=(nequivalent_points*ncomplete), dtype=float)
        Local expansions, aligned by global index from `key_to_index`.
    x0 : np.array(shape=(1, 3), dtype=np.float64)
        Center of octree root node.
    r0 : np.float64
        Half side length of octree root node.
    alpha_outer : float
        Relative size of outer surface.
    equivalent_surface : np.array(shape=(nequivalent_points, 3), dtype=float)
        Discretized equivalent surface.
    nequivalent_points : int
        Number of quadrature points on equivalent_surface.
    p2p_parallel_function : function
	    Parallel P2P function handle for this kernel.
    dtype : type
	    Corresponds to precision of experiment ∈ {np.float32, np.float64}.
    """
    local_sources, local_targets, local_source_densities, local_source_index_pointer, local_target_index_pointer = prepare_l2t_data(
        leaves,
        nleaves,
        targets,
        target_index_pointer,
        equivalent_surface,
        nequivalent_points,
        x0,
        r0,
        alpha_outer,
        key_to_index,
        key_to_leaf_index,
        local_expansions,
        dtype
    )

    target_potentials_vec = p2p_parallel_function(
        sources=local_sources,
        targets=local_targets,
        source_densities=local_source_densities,
        source_index_pointer=local_source_index_pointer,
        target_index_pointer=local_target_index_pointer
    )

    for i in range(nleaves):
        res = target_potentials_vec[local_target_index_pointer[i]:local_target_index_pointer[i+1]]
        leaf = leaves[i]
        leaf_idx = key_to_leaf_index[leaf]
        target_potentials[target_index_pointer[leaf_idx]:target_index_pointer[leaf_idx+1], :] += res


@numba.njit(cache=True, parallel=True)
def near_field(
    leaves,
    nleaves,
    key_to_leaf_index,
    key_to_index,
    targets,
    u_lists,
    target_index_pointer,
    sources,
    source_densities,
    source_index_pointer,
    target_potentials,
    p2p_function,
    p2p_gradient_function
):
    """
    Parameters:
    -----------
    leaves: np.array(shape=(nleaves), dtype=np.int64)
	    Octree leaves.
    nleaves : int
    	Number of leaves.
    key_to_leaf_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to global index.
    targets : np.array(shape=(ntargets, 3), dtype=float)
        All target coordinates.
    u_lists : np.array(shape=(ncomplete, nu_list), dtype=np.int64)
        All U lists for nodes in octree.
    target_index_pointer : np.array(shape=(nleaves+1), dtype=int)
        Index pointer aligning targets by leaf.
    sources : np.array(shape=(nsources, 3), dtype=float)
        All source coordinates.
    source_densities : np.array(shape=(nsources), dtype=float)
        Densities at source coordinates.
    source_index_pointer : np.array(shape=(nleaves+1), dtype=int)
        Index pointer aligning sources by leaf.
    target_potentials : np.array(shape=(ntargets, 4), dtype=float)
        Target potentials (component 0), and x, y, and z components of
        potential gradient (components 1, 2, 3 resp.).
    """
    for i in numba.prange(nleaves):
        target = leaves[i]
        target_leaf_index = key_to_leaf_index[target]
        target_index = key_to_index[target]
        targets_at_node = targets[
            target_index_pointer[target_leaf_index]:target_index_pointer[target_leaf_index+1]
        ]

        u_list = u_lists[target_index]
        u_list = u_list[u_list != -1]

        # single threaded over inner loop over u list!
        for j in range(len(u_list)):
            source = u_list[j]
            source_leaf_index = key_to_leaf_index[source]

            sources_at_node = sources[
                source_index_pointer[source_leaf_index]:source_index_pointer[source_leaf_index+1]
            ]

            source_densities_at_node = source_densities[
                source_index_pointer[source_leaf_index]:source_index_pointer[source_leaf_index+1]
            ]

            target_potentials[target_index_pointer[target_leaf_index]:target_index_pointer[target_leaf_index+1],1:] += \
                p2p_gradient_function(sources_at_node, targets_at_node, source_densities_at_node)
            target_potentials[target_index_pointer[target_leaf_index]:target_index_pointer[target_leaf_index+1], 0] += \
               p2p_function(sources_at_node, targets_at_node, source_densities_at_node)


        # Now compute contribution due to sources in the node itself
        sources_at_node = sources[
            source_index_pointer[target_leaf_index]:source_index_pointer[target_leaf_index+1]
        ]

        source_densities_at_node = source_densities[
            source_index_pointer[target_leaf_index]:source_index_pointer[target_leaf_index+1]
        ]

        target_potentials[target_index_pointer[target_leaf_index]:target_index_pointer[target_leaf_index+1],1:] += \
            p2p_gradient_function(sources_at_node, targets_at_node, source_densities_at_node)
        target_potentials[target_index_pointer[target_leaf_index]:target_index_pointer[target_leaf_index+1], 0] += \
           p2p_function(sources_at_node, targets_at_node, source_densities_at_node)
