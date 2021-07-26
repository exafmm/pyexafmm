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
        scale_function
    ):
    """
    Create vector of scales, and check potentials, required for the P2M operator.

    Parameters:
    -----------
    leaves : np.array(nleaves, np.int64)
    nleaves : int
    sources : np.array((nsources, 3), float)
    source_densities : np.array(nsources, float)
    source_index_pointer : np.array(nleaves+1, int)
    key_to_to_leaf_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    x0 : np.array(shape=(1, 3), dtype=float)
        Physical center of octree root node.
    r0 : float
        Half side length of octree root node.
    alpha_outer: float
        Relative size of outer surface
    check_surface : np.array(shape=(n_check, 3), dtype=float)
        Discretised check surface.
    ncheck_points : int
        Number of quadrature points on the check surface.
    p2p_function : function handle
        Serial P2P function.
    scale_function : function handle
        Scaling function for kernel.

    Returns:
    --------
    (np.array(float), np.array(float))
        Tuple of scales and check potentials (ordered by leaf index) respectively.
    """

    dtype = sources.dtype

    scales = np.zeros(nleaves, dtype=dtype)
    check_potentials = np.zeros(nleaves*ncheck_points, dtype=dtype)

    for thread_idx in numba.prange(nleaves):

        leaf = leaves[thread_idx]

        # Lookup leaf sources, and leaf source densities
        idx = key_to_leaf_index[leaf]
        leaf_sources = sources[source_index_pointer[idx]:source_index_pointer[idx+1]]
        leaf_source_densities = source_densities[source_index_pointer[idx]:source_index_pointer[idx+1]]

        # Compute center of leaf box in cartesian coordinates
        leaf_center = morton.find_physical_center_from_key(
            key=leaf, x0=x0, r0=r0
        )

        leaf_level = morton.find_level(leaf)

        upward_check_surface = surface.scale_surface(
            surf=check_surface,
            radius=r0,
            level=np.int32(leaf_level),
            center=leaf_center.astype(np.float32),
            alpha=alpha_outer,
        )

        check_potential = p2p_function(
            targets=upward_check_surface,
            sources=leaf_sources,
            source_densities=leaf_source_densities,
        )

        idx = thread_idx*ncheck_points
        check_potentials[idx:idx+ncheck_points] += check_potential
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
    Core loop of P2M operator.

    Parameters:
    -----------
    leaves : np.array(nleaves, np.int64)
    nleaves : np.int32
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to complete tree index.
    nequivalent_points : np.int32
        Number of quadrature points on the equivalent surface.
    ncheck_points : np.int32
        Number of quadrature points on the check surface.
    uc2e_inv_a : np.array(shape=(n_equivalent, n_equivalent), dtype=np.float64)
    uc2e_inv_b : np.array(shape=(n_equivalent, n_check), dtype=np.float64)
    multipole_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all multipole expansions.
    """

    for thread_idx in numba.prange(nleaves):

        leaf = leaves[thread_idx]
        leaf_lidx = key_to_index[leaf]*nequivalent_points

        scale = scales[thread_idx]

        check_lidx = thread_idx*ncheck_points
        check_potential = check_potentials[check_lidx:check_lidx+ncheck_points]
        multipole_expansions[leaf_lidx:leaf_lidx+nequivalent_points] += scale*(uc2e_inv_a @ (uc2e_inv_b @ check_potential))


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
        scale_function
    ):
    """
    P2M operator. Compute the multipole expansion from the sources at each
    leaf node. Composed of two numba-fied operators.

    Parameters:
    -----------
    leaves : np.array(nleaves, np.int64)
    nleaves : np.int32
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to complete tree index.
    key_to_to_leaf_index : numba.types.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    sources : np.array((nsources, 3), np.float32)
    source_densities : np.array(nsources, np.float32)
    source_index_pointer : np.array(nleaves+1,np.float32)
    multipole_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all multipole expansions.
    nequivalent_points : np.int32
        Number of quadrature points on the equivalent surface.
    x0 : np.array(shape=(1, 3), dtype=np.float32)
        Physical center of octree root node.
    r0 : np.float32
        Half side length of octree root node.
    alpha_outer: np.float32
        Relative size of outer surface
    check_surface : np.array(shape=(n_check, 3), dtype=np.float32)
        Discretised check surface.
    ncheck_points : np.int32
        Number of quadrature points on the check surface.
    uc2e_inv_a : np.array(shape=(n_equivalent, n_equivalent), dtype=np.float64)
    uc2e_inv_b : np.array(shape=(n_equivalent, n_check), dtype=np.float64)
    p2p_function : function handle
        Serial P2P function.
    scale_function : function handle
        Scaling function for kernel.
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
        scale_function=scale_function
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


@numba.njit(cache=True)
def m2m(
        keys,
        multipole_expansions,
        m2m,
        key_to_index,
        nequivalent_points,
    ):
    """
    M2M operator. Add the contribution of the multipole expansions of a given
        source node's children to it's own multipole expansion for all nodes
        on a given level.

    Parameters:
    -----------
    keys : np.int64
        Morton keys of source nodes at this level.
    multipole_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all multipole expansions.
    m2m : np.array(shape=(8, n_equivalent, n_equivalent), dtype=np.float32)
        Unscaled pre-computed M2M operators for all children. Implicitly
            indexed by order of Morton encoding from
            adaptoctree.morton.find_children.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
    nequivalent_points : np.int64
    """

    for i in range(len(keys)):
        key = keys[i]

        parent = morton.find_parent(key)
        siblings = morton.find_siblings(key)

        # Compute operator index
        operator_idx = np.where(siblings == key)[0]

        # Get child equivalent density
        key_idx = key_to_index[key]
        child_lidx = (key_idx)*nequivalent_points
        child_ridx = (key_idx+1)*nequivalent_points

        # Add to source data
        parent_idx = key_to_index[parent]
        parent_lidx = parent_idx*nequivalent_points
        parent_ridx = (parent_idx+1)*nequivalent_points
        multipole_expansions[parent_lidx:parent_ridx] += (
            m2m[operator_idx][0] @ multipole_expansions[child_lidx:child_ridx]
        )


@numba.njit(cache=True, parallel=False)
def m2l_core(
        target,
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
    Core loop of M2L operator.

    Parameters:
    -----------
    target : np.int64
    v_list : np.array(np.int64)
    u : np.array(np.float32)
        Compressed left singular vectors of SVD of M2L Gram matrix for nodes at this level.
    s : np.array(np.float32)
        Compressed singular values of SVD of M2L Gram matrix for nodes at this level.
    vt : np.array(np.float32)
        Compressed right singular vectors of SVD of M2L Gram matrix for nodes at this level.
    dc2e_inv_a : np.array(shape=(n_equivalent, n_equivalent), dtype=np.float64)
    dc2e_inv_b : np.array(shape=(n_equivalent, n_check), dtype=np.float64)
    local_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all local expansions.
    multipole_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all multipole expansions.
    nequivalent_points: np.int32
        Number of points discretising the equivalent surface.
    ncheck_points : np.int32
        Number of points discretising the check surface.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
    hash_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map between hashes and indices of transfer vectors.
    scale : np.float32
        Precomputed kernel scale for this level.
    """
    nv_list = len(v_list)

    # Index of local expansion
    lidx = key_to_index[target]*nequivalent_points

    for i in range(nv_list):

        source = v_list[i]

        transfer_vector = morton.find_transfer_vector(target, source)
        v_idx = hash_to_index[transfer_vector]
        v_lidx = v_idx*nequivalent_points
        v_ridx = v_lidx+nequivalent_points
        vt_sub = np.copy(vt[:, v_lidx:v_ridx])

        m_lidx = key_to_index[source]*nequivalent_points
        m_ridx = m_lidx+nequivalent_points
        local_expansions[lidx:lidx+nequivalent_points] += scale*(dc2e_inv_a @ (dc2e_inv_b @ (u @ (s @ (vt_sub @ multipole_expansions[m_lidx:m_ridx])))))


@numba.njit(cache=True, parallel=True)
def m2l(
        targets,
        v_lists,
        key_to_index,
        u,
        s,
        vt,
        dc2e_inv_a,
        dc2e_inv_b,
        local_expansions,
        multipole_expansions,
        nequivalent_points,
        hash_to_index,
        scale
    ):
    """
    M2L operator. Parallelised over all targets in a given level.

    Parameters:
    -----------
    targets : np.array(np.int64)
    v_lists : np.array(shape=(n_v_list, ncomplete), dtype=np.int64)
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
    u : np.array(np.float32)
        Compressed left singular vectors of SVD of M2L Gram matrix for nodes at this level.
    s : np.array(np.float32)
        Compressed singular values of SVD of M2L Gram matrix for nodes at this level.`
    vt : np.array(np.float32)
        Compressed right singular vectors of SVD of M2L Gram matrix for nodes at this level.
    dc2e_inv_a : np.array(shape=(n_equivalent, n_equivalent), dtype=np.float32)
    dc2e_inv_b : np.array(shape=(n_equivalent, n_check), dtype=np.float32)
    local_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all local expansions.
    multipole_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all multipole expansions.
    nequivalent_points: np.int32
        Number of points discretising the equivalent surface.
    ncheck_points : np.int32
        Number of points discretising the check surface.
    hash_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
    scale : np.float32
        Precomputed kernel scale for this level.
    """
    # range over targets on a given level
    ntargets = len(targets)

    # Call innner loop
    for i in numba.prange(ntargets):

        # Pick out the target
        target = targets[i]

        # Pick out the v list
        v_list = v_lists[key_to_index[target]]

        # Filter v list
        v_list = v_list[v_list != -1]
        v_list = v_list[v_list != 0]

        m2l_core(
            target=target,
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
def l2l(
        key,
        local_expansions,
        l2l,
        key_to_index,
        nequivalent_points
     ):
    """
    L2L operator. Translate the local expansion of a given node's parent to itself.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    local_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all local expansions.
    l2l : np.array(shape=(8, n_check, n_check), dtype=np.float32)
        Unscaled pre-computed L2L operators for all children. Implicitly
            indexed by order of Morton encoding from
            adaptoctree.morton.find_children.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
    nequivalent_points : np.int64
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


@numba.njit(cache=True)
def s2l(
        key,
        sources,
        source_densities,
        source_index_pointer,
        key_to_index,
        key_to_leaf_index,
        x_list,
        local_expansions,
        x0,
        r0,
        alpha_inner,
        check_surface,
        nequivalent_points,
        dc2e_inv_a,
        dc2e_inv_b,
        scale_function,
        p2p_function
    ):
    """
    S2L operator. For source nodes in a target node's X list, the multipole
        expansion of the source node doesn't apply, as the target node lies
        within it's upward check surface, therefore the sources are used to
        compute the contribution to the local expansion of the target node
        directly.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    sources : np.array(shape=(nsources, 3), dtype=np.float32)
        Source coordinates.
    source_densities : np.array(shape=(nsources, 1), dtype=np.float32)
        Charge densities at source points.
    source_index_pointer : np.array(np.int64)
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
    key_to_to_leaf_index : numba.types.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    x_list : np.array(shape=(n_x_list, 1), dtype=np.int64)
        Morton keys of X list members.
    local_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all local expansions.
    x0 : np.array(shape=(1, 3), dtype=np.float32)
        Physical center of octree root node.
    r0 : np.float32
        Half side length of octree root node.
    alpha_inner: np.float32
        Relative size of inner surface
    check_surface : np.array(shape=(n_check, 3), dtype=np.float32)
        Discretised check surface.
    nequivalent_points : np.int64
    dc2e_inv_a : np.array(shape=(n_equivalent, n_equivalent), dtype=np.float32)
    dc2e_inv_b : np.array(shape=(n_equivalent, n_check), dtype=np.float32)
    scale_function : function
        Function handle for kernel scaling.
    p2p_function : function
        Function handle for kernel P2P.
    """

    level = morton.find_level(key)
    scale = scale_function(level)
    center = morton.find_physical_center_from_key(key, x0, r0)

    key_idx = key_to_index[key]
    key_lidx = key_idx*nequivalent_points
    key_ridx = key_lidx+nequivalent_points

    downward_check_surface = surface.scale_surface(
        surf=check_surface,
        radius=r0,
        level=level,
        center=center,
        alpha=alpha_inner
    )

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


@numba.njit(cache=True)
def m2t(
        target_key,
        target_index_pointer,
        key_to_index,
        key_to_leaf_index,
        w_list,
        target_coordinates,
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
    M2T operator. M2L translations aren't applicable, as the source nodes in
        the W list are not outside of the downward equivalent surface of the
        target node.

    Parameters:
    -----------
    target_key : np.int64
        Morton key of source node.
    target_index_pointer : np.array(np.int32)
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to complete index.
    key_to_to_leaf_index : numba.types.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    w_list : np.array(shape=(n_v_list, 1), dtype=np.int64)
        Morton keys of W list members.
    target_coordinates : np.array(shape=(ntargets, 3), dtype=np.float32)
        Target coordinates of particles in target node.
    targets_potentials : np.array(shape=(ntargets,), dtype=np.float32)
        Potentials at all target points, due to all source points.
    multipole_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all multipole expansions.
    x0 : np.array(shape=(1, 3), dtype=np.float32)
        Physical center of octree root node.
    r0 : np.float32
        Half side length of octree root node.
    alpha_inner : np.float32
        Relative size of inner surface
    equivalent_surface : np.array(shape=(n_equivalent, 3), dtype=np.float32)
        Discretised equivalent surface.
    p2p_function : function
        Function handle for kernel P2P.
    gradient_function : function
        Function handle for kernel gradient P2P.
    """

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

        target_potentials[target_index_pointer[target_idx]:target_index_pointer[target_idx+1],0] += p2p_function(
            sources=upward_equivalent_surface,
            targets=target_coordinates,
            source_densities=multipole_expansions[source_lidx:source_ridx]
        )

        target_potentials[
                target_index_pointer[target_idx]:target_index_pointer[target_idx+1], 1:
            ] += gradient_function(
                sources=upward_equivalent_surface,
                targets=target_coordinates,
                source_densities=multipole_expansions[source_lidx:source_ridx]
            )


@numba.njit(cache=True)
def l2t(
        key,
        key_to_index,
        key_to_leaf_index,
        target_coordinates,
        target_potentials,
        target_index_pointer,
        local_expansions,
        x0,
        r0,
        alpha_outer,
        equivalent_surface,
        nequivalent_points,
        p2p_function,
        gradient_function,
    ):
    """
    L2T operator. Evaluate the local expansion at the target points in a given
        target node.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to complete index.
    key_to_to_leaf_index : numba.types.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    target_coordinates : np.array(shape=(ntargets, 3), dtype=np.float32)
        Target coordinates of particles in target node.
    targets_potentials : np.array(shape=(ntargets,), dtype=np.float32)
        Potentials at all target points, due to all source points.
    target_index_pointer : np.array(np.int32)
    local_expansions : np.array(shape=(ncomplete*nequivalent_points, dtype=np.float32)
        Array of all local expansions.
    x0 : np.array(shape=(1, 3), dtype=np.float32)
        Physical center of octree root node.
    r0 : np.float32
        Half side length of octree root node.
    alpha_outer : np.float32
        Relative size of outer surface
    equivalent_surface : np.array(shape=(n_equivalent, 3), dtype=np.float32)
        Discretised equivalent surface.
    p2p_function : function
        Function handle for kernel P2P.
    gradient_function : function
        Function handle for kernel gradient P2P.
    """
    source_idx = key_to_index[key]
    source_lidx = source_idx*nequivalent_points
    source_ridx = source_lidx+nequivalent_points

    level = morton.find_level(key)
    center = morton.find_physical_center_from_key(key, x0, r0)

    downward_equivalent_surface = surface.scale_surface(
        equivalent_surface,
        r0,
        level,
        center,
        alpha_outer
    )

    target_idx = key_to_leaf_index[key]

    target_potentials[target_index_pointer[target_idx]:target_index_pointer[target_idx+1],0] += p2p_function(
        sources=downward_equivalent_surface,
        targets=target_coordinates,
        source_densities=local_expansions[source_lidx:source_ridx]
    )

    target_potentials[
            target_index_pointer[target_idx]:target_index_pointer[target_idx+1], 1:
        ] += gradient_function(
            sources=downward_equivalent_surface,
            targets=target_coordinates,
            source_densities=local_expansions[source_lidx:source_ridx]
        )


@numba.njit(cache=True)
def prepare_u_list_data(
        leaves,
        targets,
        target_index_pointer,
        sources,
        source_densities,
        source_index_pointer,
        key_to_index,
        key_to_leaf_index,
        u_lists,
        max_points,
    ):
    """
    Create batched (in terms of index pointer) sources and targets in order to
        run the P2P computation in parallel over all targets at the leaf level,
        and all sources in their U lists.

    Parameters:
    -----------
    leaves : np.array(nleaves, np.int64)
    targets : np.array((ntargets, 3), np.int32)
        All target coordinates.
    target_index_pointer : np.array(np.int32)
    sources : np.array((nsources, 3), np.int32)
        All source coordinates.
    source_densities : np.array(nsources, np.int32)
    source_index_pointer : np.array(np.int32)
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to complete index.
    key_to_to_leaf_index : numba.types.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    u_lists : np.array(shape=(n_u_list, ncomplete), dtype=np.int64)
    max_points : np.int64
        Max points per node.

    Returns:
    --------
    5-tuple containing the re-batched sources, targets, source densities, source index pointers
    and target index pointers respectively.
    """

    local_sources = np.zeros((max_points*26*len(leaves), 3), np.float32)
    local_source_densities = np.zeros((max_points*26*len(leaves)), np.float32)
    local_targets = np.zeros((max_points*len(leaves), 3), np.float32)
    local_source_index_pointer = np.zeros(len(leaves)+1, np.int64)
    local_target_index_pointer = np.zeros(len(leaves)+1, np.int64)

    source_ptr = 0
    target_ptr = 0
    local_source_index_pointer[0] = source_ptr
    local_target_index_pointer[0] = target_ptr

    for i in range(len(leaves)):
        target = leaves[i]
        target_leaf_index = key_to_leaf_index[target]
        target_index = key_to_index[target]

        targets_at_node = targets[
            target_index_pointer[target_leaf_index]:target_index_pointer[target_leaf_index+1]
        ]

        ntargets_at_node = len(targets_at_node)
        new_target_ptr = target_ptr+ntargets_at_node

        local_targets[target_ptr:new_target_ptr] = targets_at_node
        target_ptr = new_target_ptr

        local_target_index_pointer[i+1] = target_ptr

        u_list = u_lists[target_index]
        u_list = u_list[u_list != -1]


        for j in range(len(u_list)):

            source = u_list[j]
            source_leaf_index = key_to_leaf_index[source]
            sources_at_node = sources[
                source_index_pointer[source_leaf_index]:source_index_pointer[source_leaf_index+1]
            ]
            nsources_at_node = len(sources_at_node)

            source_densities_at_node = source_densities[
                source_index_pointer[source_leaf_index]:source_index_pointer[source_leaf_index+1]
                ]

            new_source_ptr = source_ptr+nsources_at_node

            local_sources[source_ptr:new_source_ptr] = sources_at_node
            local_source_densities[source_ptr:new_source_ptr] = source_densities_at_node

            source_ptr = new_source_ptr

        local_source_index_pointer[i+1] = source_ptr

    local_sources = local_sources[local_source_index_pointer[0]:local_source_index_pointer[-1]]
    local_source_densities = local_source_densities[local_source_index_pointer[0]:local_source_index_pointer[-1]]
    local_targets = local_targets[local_target_index_pointer[0]:local_target_index_pointer[-1]]

    return local_sources, local_targets, local_source_densities, local_source_index_pointer, local_target_index_pointer


@numba.njit(cache=True)
def near_field_u_list(
        u_lists,
        leaves,
        targets,
        target_index_pointer,
        sources,
        source_densities,
        source_index_pointer,
        key_to_index,
        key_to_leaf_index,
        max_points,
        target_potentials,
        p2p_parallel_function
    ):
    """
    Evaluate all near field particles for source nodes within a given target
        node's U list directly.

    Parameters:
    -----------
    u_lists : np.array(shape=(n_u_list, ncomplete), dtype=np.int64)
    leaves : np.array(nleaves, np.int64)
    targets : np.array((ntargets, 3), np.int32)
        All target coordinates.
    target_index_pointer : np.array(np.int32)
    sources : np.array((nsources, 3), np.int32)
        All source coordinates.
    source_densities : np.array(nsources, np.int32)
    source_index_pointer : np.array(np.int32)
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to complete index.
    key_to_to_leaf_index : numba.types.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    max_points : np.int64
        Max points per node.
    targets_potentials : np.array(shape=(ntargets,), dtype=np.float32)
        Potentials at all target points, due to all source points.
    p2p_parallel_function : function
    """

    local_sources, local_targets, local_source_densities, local_source_index_pointer, local_target_index_pointer = prepare_u_list_data(
            leaves=leaves,
            targets=targets,
            target_index_pointer=target_index_pointer,
            sources=sources,
            source_densities=source_densities,
            source_index_pointer=source_index_pointer,
            key_to_index=key_to_index,
            key_to_leaf_index=key_to_leaf_index,
            u_lists=u_lists,
            max_points=max_points,
        )

    target_potentials_vec = p2p_parallel_function(
        sources=local_sources,
        targets=local_targets,
        source_densities=local_source_densities,
        source_index_pointer=local_source_index_pointer,
        target_index_pointer=local_target_index_pointer
    )

    nleaves = len(local_target_index_pointer) - 1

    for i in range(nleaves):
        res = target_potentials_vec[local_target_index_pointer[i]:local_target_index_pointer[i+1]]
        leaf = leaves[i]
        leaf_idx = key_to_leaf_index[leaf]
        target_potentials[target_index_pointer[leaf_idx]:target_index_pointer[leaf_idx+1], :] += res


def near_field_node(
        key,
        key_to_leaf_index,
        source_coordinates,
        source_densities,
        target_coordinates,
        target_index_pointer,
        target_potentials,
        p2p_function,
        gradient_function,
    ):
    """
    Evaluate all near field particles for source particles within a given
        target node

    Parameters:
    -----------
    key : np.int64
        Target key.
    key_to_to_leaf_index : numba.types.Dict(key_type=np.int64, value_type=np.int64)
        Map from key to leaf index.
    source_coordinates: np.array(shape=(nsources, 3), dtype=np.float32)
        Source coordinates of particles in the target node.
    source_densities : np.array(nsources, np.int32)
    target_coordinates : np.array(shape=(ntargets, 3), dtype=np.float32)
        Target coordinates of particles in target node.
    target_index_pointer : np.array(np.int32)
    targets_potentials : np.array(shape=(ntargets,), dtype=np.float32)
        Potentials at all target points, due to all source points.
    p2p_function: function
        Function handle for kernel P2P.
    gradient_function : function
        Function handle for kernel gradient P2P.
    """
    idx = key_to_leaf_index[key]

    target_potentials[
        target_index_pointer[idx]:target_index_pointer[idx+1],0
        ] += p2p_function(
                    sources=source_coordinates,
                    targets=target_coordinates,
                    source_densities=source_densities
                   )

    target_potentials[
        target_index_pointer[idx]:target_index_pointer[idx+1],1:
        ] += gradient_function(
                    sources=source_coordinates,
                    targets=target_coordinates,
                    source_densities=source_densities
                   )
