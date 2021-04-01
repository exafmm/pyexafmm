"""
Compute operators, accelerated with Numba.
"""
import numba
import numpy as np

import adaptoctree.morton as morton
from adaptoctree.utils import deterministic_checksum

import fmm.surface as surface
from fmm.parameters import DIGEST_SIZE


@numba.njit(cache=True, parallel=True)
def prepare_p2m_data(
        leaves,
        nleaves,
        sources,
        source_densities,
        sources_to_keys,
        x0,
        r0,
        alpha_outer,
        check_surface,
        ncheck_points,
        p2p_function,
        scale_function
    ):

    scales = np.zeros(nleaves, dtype=np.float32)
    check_potentials = np.zeros(nleaves*ncheck_points, np.float32)

    for thread_idx in numba.prange(nleaves):

        leaf = leaves[thread_idx]

        # Source indices in a given leaf
        source_indices = sources_to_keys == leaf

        # Find leaf sources, and leaf source densities
        leaf_sources = sources[source_indices]
        leaf_source_densities = source_densities[source_indices]

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
def p2m_subroutine(
        leaves,
        nleaves,
        key_to_index,
        nequivalent_points,
        ncheck_points,
        uc2e_inv,
        scales,
        multipole_expansions,
        check_potentials
    ):

    for thread_idx in numba.prange(nleaves):

        leaf = leaves[thread_idx]
        leaf_idx = key_to_index[leaf]
        lidx = leaf_idx*nequivalent_points
        ridx = lidx+nequivalent_points

        scale = scales[thread_idx]

        idx = thread_idx*ncheck_points
        check_potential = check_potentials[idx:idx+ncheck_points]
        multipole_expansions[lidx:ridx] += scale*(uc2e_inv @ (check_potential))


@numba.njit(cache=True)
def p2m(
        leaves,
        nleaves,
        key_to_index,
        sources,
        source_densities,
        sources_to_keys,
        multipole_expansions,
        nequivalent_points,
        x0,
        r0,
        alpha_outer,
        check_surface,
        ncheck_points,
        uc2e_inv,
        p2p_function,
        scale_function
    ):

    scales, check_potentials = prepare_p2m_data(
        leaves=leaves,
        nleaves=nleaves,
        sources=sources,
        source_densities=source_densities,
        sources_to_keys=sources_to_keys,
        x0=x0,
        r0=r0,
        alpha_outer=alpha_outer,
        check_surface=check_surface,
        ncheck_points=ncheck_points,
        p2p_function=p2p_function,
        scale_function=scale_function
    )

    p2m_subroutine(
        leaves=leaves,
        nleaves=nleaves,
        key_to_index=key_to_index,
        nequivalent_points=nequivalent_points,
        ncheck_points=ncheck_points,
        uc2e_inv=uc2e_inv,
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
    key : np.int64
        Morton key of source node.
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


@numba.njit(cache=True, parallel=True)
def m2l(
        keys,
        key_to_index,
        v_lists,
        multipole_expansions,
        local_expansions,
        dc2e_inv,
        nequivalent_points,
        ncheck_points,
        u,
        s,
        vt,
        hashes,
        scale_function,
        depth,
    ):

    for i in numba.prange(len(keys)):

        # Pick out key
        key = keys[i]

        # Compute indices to lookup target local expansion
        target_idx = key_to_index[key]

        # Pick out V list
        v_list = v_lists[target_idx]
        v_list = v_list[v_list != -1]

        if len(v_list) > 0:
            target_lidx = target_idx*nequivalent_points
            target_ridx = (target_idx+1)*nequivalent_points


            level = morton.find_level(key)
            scale = np.float32(scale_function(level))

            # Compute the hasehes of transfer vectors in the target's V List.
            transfer_vectors = morton.find_transfer_vectors(key, v_list, depth)

            # Use the hashes to compute the index of the M2L Gram matrix at this level
            m2l_lidxs = np.zeros(len(v_list), np.int32)
            m2l_ridxs = np.zeros(len(v_list), np.int32)

            for j in range(len(transfer_vectors)):
                hash_vector = deterministic_checksum(transfer_vectors[j])
                m2l_idx = np.where(hash_vector == hashes)[0][0]
                m2l_lidxs[j] += m2l_idx*ncheck_points
                m2l_ridxs[j] += (m2l_idx+1)*ncheck_points

            for idx in range(len(v_list)):
                # Find source densities for this source
                source = v_list[idx]

                # Pick out compressed right singular vector for this M2L gram matrix
                u_sub = u[m2l_lidxs[idx]:m2l_ridxs[idx]]

                # Compute indices to lookup source multipole expansions
                source_idx = key_to_index[source]
                source_lidx = source_idx*nequivalent_points
                source_ridx = (source_idx+1)*nequivalent_points

                # Compute contribution from source, to the local expansion
                local_expansions[target_lidx:target_ridx] += scale*(dc2e_inv @ (u_sub @ (s @ (vt @ multipole_expansions[source_lidx:source_ridx]))))


@numba.njit(cache=True)
def l2l(
        key,
        local_expansions,
        l2l,
        key_to_index,
        nequivalent_points
     ):
    """
    L2L operator. Translate the local expansion of a parent node, to each of
        it's children.

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
    parent_idx = key_to_index[key]
    parent_lidx = parent_idx*nequivalent_points
    parent_ridx = (parent_idx+1)*nequivalent_points
    parent_equivalent_density = local_expansions[parent_lidx:parent_ridx]

    children = morton.find_children(key)

    for child in children:

        if child in key_to_index:

            # Compute expansion index
            child_idx = key_to_index[child]
            child_lidx = child_idx*nequivalent_points
            child_ridx = (child_idx+1)*nequivalent_points

            # Compute operator index
            operator_idx = child == children

            # Compute contribution to local expansion of child from parent
            local_expansions[child_lidx:child_ridx] += l2l[operator_idx][0] @ parent_equivalent_density


@numba.njit(cache=True)
def s2l(
        key,
        key_to_index,
        x_list,
        sources,
        source_densities,
        sources_to_keys,
        local_expansions,
        x0,
        r0,
        alpha_inner,
        check_surface,
        nequivalent_points,
        dc2e_inv,
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
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
    x_list : np.array(shape=(n_x_list, 1), dtype=np.int64)
        Morton keys of X list members.
    sources : np.array(shape=(nsources, 3), dtype=np.float32)
        Source coordinates.
    source_densities : np.array(shape=(nsources, 1), dtype=np.float32)
        Charge densities at source points.
    sources_to_keys : np.array(shape=(nsources, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) source lies.
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
    dc2e_inv : np.array(shape=(n_equivalent, n_check), dtype=np.float32)
    scale_function : function
        Function handle for kernel scaling.
    p2p_function : function
        Function handle for kernel P2P.
    """
    level = np.int32(morton.find_level(key))
    scale = np.int32(scale_function(level))
    center = morton.find_physical_center_from_key(key, x0, r0).astype(np.float32)

    key_idx = key_to_index[key]
    key_lidx = key_idx*nequivalent_points
    key_ridx = (key_idx+1)*nequivalent_points

    downward_check_surface = surface.scale_surface(
        surf=check_surface,
        radius=r0,
        level=level,
        center=center,
        alpha=alpha_inner
    )

    for source in x_list:

        source_indices = sources_to_keys == source

        if np.any(source_indices == True):
            source_coodinates = sources[source_indices]
            densities = source_densities[source_indices]
            downward_check_potential = p2p_function(
                sources=source_coodinates,
                targets=downward_check_surface,
                source_densities=densities
            )

            local_expansions[key_lidx:key_ridx] += (scale*(dc2e_inv @ (downward_check_potential)))


@numba.njit(cache=True)
def m2t(
        key,
        key_to_index,
        w_list,
        targets,
        targets_to_keys,
        target_potentials,
        multipole_expansions,
        x0,
        r0,
        alpha_inner,
        equivalent_surface,
        nequivalent_points,
        p2p_function
    ):
    """
    M2T operator. M2L translations aren't applicable, as the source nodes in
        the W list are not outside of the downward equivalent surface of the
        target node.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
    w_list : np.array(shape=(n_v_list, 1), dtype=np.int64)
        Morton keys of W list members.
    targets : np.array(shape=(ntargets, 3), dtype=np.float32)
        Target coordinates.
    targets_to_keys: np.array(shape=(ntargets, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) target lies.
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
    """

    # Find target particles
    target_indices = targets_to_keys == key

    if (len(target_indices) > 0):
        target_coordinates = targets[target_indices]

        for source in w_list:

            source_idx = key_to_index[source]
            source_lidx = source_idx*nequivalent_points
            source_ridx = (source_idx+1)*nequivalent_points

            source_level = np.int32(morton.find_level(source))
            source_center = morton.find_physical_center_from_key(source, x0, r0).astype(np.float32)

            upward_equivalent_surface = surface.scale_surface(
                surf=equivalent_surface,
                radius=r0,
                level=source_level,
                center=source_center,
                alpha=alpha_inner
            )

            target_potentials[target_indices] += p2p_function(
                sources=upward_equivalent_surface,
                targets=target_coordinates,
                source_densities=multipole_expansions[source_lidx:source_ridx]
            )


@numba.njit(cache=True)
def l2t(
        key,
        key_to_index,
        targets,
        targets_to_keys,
        target_potentials,
        local_expansions,
        x0,
        r0,
        alpha_outer,
        equivalent_surface,
        nequivalent_points,
        p2p_function,
    ):
    """
    L2T operator. Evaluate the local expansion at the target points in a given
        target node.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    key_to_index : numba.typed.Dict(key_type=np.int64, value_type=np.int64)
    targets : np.array(shape=(ntargets, 3), dtype=np.float32)
        Target coordinates.
    targets_to_keys: np.array(shape=(ntargets, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) target lies.
    targets_potentials : np.array(shape=(ntargets,), dtype=np.float32)
        Potentials at all target points, due to all source points.
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
    """
    target_idxs = key == targets_to_keys

    if len(target_idxs) > 0:

        source_idx = key_to_index[key]
        source_lidx = (source_idx)*nequivalent_points
        source_ridx = (source_idx+1)*nequivalent_points

        level = np.int32(morton.find_level(key))
        center = morton.find_physical_center_from_key(key, x0, r0).astype(np.float32)

        downward_equivalent_surface = surface.scale_surface(
            equivalent_surface,
            r0,
            level,
            center,
            alpha_outer
        )

        target_coordinates = targets[target_idxs]

        target_potentials[target_idxs] += p2p_function(
            sources=downward_equivalent_surface,
            targets=target_coordinates,
            source_densities=local_expansions[source_lidx:source_ridx]
        )


@numba.njit(cache=True)
def near_field(
        key,
        u_list,
        targets,
        targets_to_keys,
        target_potentials,
        sources,
        source_densities,
        sources_to_keys,
        p2p_function
    ):
    """
    Evaluate all near field particles for source nodes within a given target
        node's U list directly.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    u_list : np.array(shape=(n_u_list, 1), dtype=np.int64)
        Morton keys of U list members.
    targets : np.array(shape=(ntargets, 3), dtype=np.float32)
        Target coordinates.
    targets_to_keys: np.array(shape=(ntargets, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) target lies.
    targets_potentials : np.array(shape=(ntargets,), dtype=np.float32)
        Potentials at all target points, due to all source points.
    sources : np.array(shape=(nsources, 3), dtype=np.float32)
        Source coordinates.
    source_densities : np.array(shape=(nsources, 1), dtype=np.float32)
        Charge densities at source points.
    sources_to_keys : np.array(shape=(nsources, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) source lies.
    p2p_function : function
        Function handle for kernel P2P.
    """
    target_indices = targets_to_keys == key

    if len(target_indices) > 0:

        target_coordinates = targets[target_indices]

        # Sources in U list
        for source in u_list:

            source_indices = sources_to_keys == source
            source_coordinates = sources[source_indices]
            densities = source_densities[source_indices]

            target_potentials[target_indices] += p2p_function(
                sources=source_coordinates,
                targets=target_coordinates,
                source_densities=densities
            )

        # Sources in target node
        local_source_indices = sources_to_keys == key
        local_source_coordinates = sources[local_source_indices]
        local_densities = source_densities[local_source_indices]

        target_potentials[target_indices] += p2p_function(
            sources=local_source_coordinates,
            targets=target_coordinates,
            source_densities=local_densities
        )
