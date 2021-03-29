"""
Compute operators, accelerated with Numba.
"""
import numba
import numpy as np

import adaptoctree.morton as morton
from adaptoctree.utils import deterministic_hash

import fmm.surface as surface
from fmm.kernel import KERNELS
from fmm.parameters import DIGEST_SIZE


@numba.njit(cache=True)
def find_physical_center_from_anchor(anchor, x0, r0):
    xmin = x0 - r0
    level = anchor[3]
    side_length = 2 * r0 / (1 << level)

    side_length = np.float64(side_length)
    anchor = anchor.astype(np.float64)

    return (anchor[:3] + 0.5) * side_length + xmin


@numba.njit(cache=True)
def find_physical_center_from_key(key, x0, r0):
    anchor = morton.decode_key(key)
    return find_physical_center_from_anchor(anchor, x0, r0)


@numba.njit(cache=True, parallel=True)
def p2m(
        leaves,
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
        uc2e_inv,
        kernel
    ):

    # Configure kernel
    p2p_function = KERNELS[kernel]['p2p']
    scale_function = KERNELS[kernel]['scale']

    nleaves = len(leaves)

    for thread_idx in numba.prange(nleaves):

        leaf = leaves[thread_idx]
        leaf_idx = key_to_index[leaf]

        lidx = leaf_idx*nequivalent_points
        ridx = (leaf_idx+1)*nequivalent_points

        # Source indices in a given leaf
        source_indices = sources_to_keys == leaf

        # Find leaf sources, and leaf source densities
        leaf_sources = sources[source_indices]
        leaf_source_densities = source_densities[source_indices]

        # # Compute center of leaf box in cartesian coordinates
        leaf_center = find_physical_center_from_key(
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

        scale = np.float32(scale_function(leaf_level))

        check_potential = p2p_function(
            targets=upward_check_surface,
            sources=leaf_sources,
            source_densities=leaf_source_densities,
        )

        multipole_expansions[lidx:ridx] += scale*(uc2e_inv @ (check_potential))


@numba.njit(cache=True)
def m2m(
        key,
        multipole_expansions,
        nequivalent_points,
        m2m,
        key_to_index,
    ):
    """
    M2M operator. Add the contribution of the multipole expansions of a given
        source node's children to it's own multipole expansion.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    multipole_expansions : {np.int64: np.array(shape=(nequivalent_points), dtype=np.float32)}
        Dictionary containing multipole expansions, indexed by Morton key of
        source nodes.
    m2m : np.array(shape=(8, n_equivalent, n_equivalent), dtype=np.float32)
        Unscaled pre-computed M2M operators for all children. Implicitly
            indexed by order of Morton encoding from
            adaptoctree.morton.find_children.
    """
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


@numba.njit(cache=True)
def m2l(
        key,
        key_to_index,
        depth,
        v_list,
        multipole_expansions,
        local_expansions,
        dc2e_inv,
        nequivalent_points,
        ncheck_points,
        kernel,
        u,
        s,
        vt,
        hashes
    ):
    """
    M2L operator. Translate the multipole expansion of all source nodes in a
        given target node's V list, into a local expansion centered on the
        target node.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    depth : np.int64
        Maximum depth of the octree, used to find transfer vectors.
    v_list : np.array(shape=(n_v_list, 1), dtype=np.int64)
        Morton keys of V list members.
    multipole_expansions : {np.int64: np.array(shape=(nequivalent_points), dtype=np.float32)}
        Dictionary containing multipole expansions, indexed by Morton key of
        source nodes.
    local_expansions : {np.int64: np.array(shape=(ncheck_points), dtype=np.float32)}
        Dictionary containing local expansions, indexed by Morton key of
        target nodes.
    dc2e_inv : np.array(shape=(n_equivalent, n_check), dtype=np.float64)
    ncheck_points : np.int32
        Number of points discretising the check surface.
    scale_function : function
        Function handle for kernel scaling.
    u : np.array(np.float32)
        Compressed left singular vectors of SVD of M2L Gram matrix for nodes at this level.
    s : np.array(np.float32)
        Compressed singular values of SVD of M2L Gram matrix for nodes at this level.
    vt : np.array(np.float32)
        Compressed right singular vectors of SVD of M2L Gram matrix for nodes at this level.
    """
    scale_function = KERNELS[kernel]['scale']

    if len(v_list) > 0:

        # Compute indices to lookup target local expansion
        target_idx = key_to_index(key)
        target_lidx = target_idx*ncheck_points
        target_ridx = (target_idx+1)*ncheck_points

        level = morton.find_level(key)
        scale = np.float32(scale_function(level))

        # Compute the hasehes of transfer vectors in the target's V List.
        transfer_vectors = morton.find_transfer_vectors(key, v_list, depth)
        hash_vectors = np.zeros(len(transfer_vectors), dtype=np.int64)

        # Use the hashes to compute the index of the M2L Gram matrix at this
        # level
        m2l_lidxs = np.zeros(len(v_list), np.int32)
        m2l_ridxs = np.zeros(len(v_list), np.int32)

        for i in range(len(transfer_vectors)):
            hash_vectors[i] = deterministic_hash(transfer_vectors[i], digest_size=DIGEST_SIZE)
            m2l_idx = np.where(hash_vectors[i] == hashes)[0][0]
            m2l_lidxs[i] = m2l_idx*ncheck_points
            m2l_ridxs[i] = (m2l_idx+1)*ncheck_points

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
        ncheck_points
     ):
    """
    L2L operator. Translate the local expansion of a parent node, to each of
        it's children.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    local_expansions : {np.int64: np.array(shape=(ncheck_points), dtype=np.float32)}
        Dictionary containing local expansions, indexed by Morton key of
        target nodes.
    l2l : np.array(shape=(8, n_check, n_check), dtype=np.float32)
        Unscaled pre-computed L2L operators for all children. Implicitly
            indexed by order of Morton encoding from
            adaptoctree.morton.find_children.
    """
    parent_idx = key_to_index[key]
    parent_lidx = parent_idx*ncheck_points
    parent_ridx = (parent_idx+1)*ncheck_points
    parent_equivalent_density = local_expansions[parent_lidx:parent_ridx]

    children = morton.find_children(key)

    for child in children:

        if child in key_to_index:

            # Compute operator index
            operator_idx = child == children

            # Compute contribution to local expansion of child from parent
            child_equivalent_density = l2l[operator_idx] @ parent_equivalent_density
            local_expansions[child] += child_equivalent_density


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
        ncheck_points,
        dc2e_inv,
        kernel,
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
    x_list : np.array(shape=(n_x_list, 1), dtype=np.int64)
        Morton keys of X list members.
    sources : np.array(shape=(nsources, 3), dtype=np.float32)
        Source coordinates.
    source_densities : np.array(shape=(nsources, 1), dtype=np.float32)
        Charge densities at source points.
    sources_to_keys : np.array(shape=(nsources, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) source lies.
    local_expansions : {np.int64: np.array(shape=(ncheck_points), dtype=np.float32)}
        Dictionary containing local expansions, indexed by Morton key of
        target nodes.
    x0 : np.array(shape=(1, 3), dtype=np.float32)
        Physical center of octree root node.
    r0 : np.float32
        Half side length of octree root node.
    alpha_inner: np.float32
        Relative size of inner surface
    check_surface : np.array(shape=(n_check, 3), dtype=np.float32)
        Discretised check surface.
    dc2e_inv : np.array(shape=(n_equivalent, n_check), dtype=np.float32)
    scale_function : function
        Function handle for kernel scaling.
    p2p_function : function
        Function handle for kernel P2P.
    """

    # Configure a kernel
    p2p_function = KERNELS[kernel]['p2p']
    scale_function = KERNELS[kernel]['scale']

    if len(x_list) > 0:
        level = np.int32(morton.find_level(key))
        scale = np.int32(scale_function(level))
        center = morton.find_physical_center_from_key(key, x0, r0).astype(np.float32)

        key_idx = key_to_index[key]
        key_lidx = key_idx*ncheck_points
        key_ridx = (key_idx+1)*ncheck_points

        downward_check_surface = surface.scale_surface(
            surf=check_surface,
            radius=r0,
            level=level,
            center=center,
            alpha=alpha_inner
        )

        for source in x_list:
            source_coodinates = []
            densities = []
            source_indices = sources_to_keys == source
            if np.any(source_indices == True):
                source_coodinates.extend(sources[source_indices])
                densities.extend(source_densities[source_indices])

                source_coodinates = np.array(source_coodinates).astype(np.float32)
                densities = np.array(densities).astype(np.float32)

                downward_check_potential = p2p_function(
                    sources=source_coodinates,
                    targets=downward_check_surface,
                    source_densities=densities
                )

                local_expansions[key_lidx:key_ridx] += (scale*(dc2e_inv @ (downward_check_potential)))


def m2t(
        key,
        w_list,
        targets,
        targets_to_keys,
        target_potentials,
        multipole_expansions,
        x0,
        r0,
        alpha_inner,
        equivalent_surface,
        kernel
    ):
    """
    M2T operator. M2L translations aren't applicable, as the source nodes in
        the W list are not outside of the downward equivalent surface of the
        target node.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    w_list : np.array(shape=(n_v_list, 1), dtype=np.int64)
        Morton keys of W list members.
    targets : np.array(shape=(ntargets, 3), dtype=np.float32)
        Target coordinates.
    targets_to_keys: np.array(shape=(ntargets, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) target lies.
    targets_potentials : np.array(shape=(ntargets,), dtype=np.float32)
        Potentials at all target points, due to all source points.
    multipole_expansions : {np.int64: np.array(shape=(nequivalent_points), dtype=np.float32)}
        Dictionary containing multipole expansions, indexed by Morton key of
        source nodes.
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
    # Configure a kernel
    p2p_function = KERNELS[kernel]['p2p']

    # Find target particles
    target_indices = targets_to_keys == key

    if (len(target_indices) > 0) and (len(w_list) > 0):
        target_coordinates = targets[target_indices]

        for source in w_list:

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
                source_densities=multipole_expansions[source]
            )


def l2t(
        key,
        targets,
        targets_to_keys,
        target_potentials,
        local_expansions,
        x0,
        r0,
        alpha_outer,
        equivalent_surface,
        kernel,
    ):
    """
    L2T operator. Evaluate the local expansion at the target points in a given
        target node.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    targets : np.array(shape=(ntargets, 3), dtype=np.float32)
        Target coordinates.
    targets_to_keys: np.array(shape=(ntargets, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) target lies.
    targets_potentials : np.array(shape=(ntargets,), dtype=np.float32)
        Potentials at all target points, due to all source points.
    local_expansions : {np.int64: np.array(shape=(ncheck_points), dtype=np.float32)}
        Dictionary containing local expansions, indexed by Morton key of
        target nodes.
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
    # Configure a kernel
    p2p_function = KERNELS[kernel]['p2p']

    target_idxs = key == targets_to_keys

    if len(target_idxs) > 0:

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
            source_densities=local_expansions[key]
        )


def near_field(
        key,
        u_list,
        targets,
        targets_to_keys,
        target_potentials,
        sources,
        source_densities,
        sources_to_keys,
        kernel
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

    # Configure a kernel
    p2p_function = KERNELS[kernel]['p2p']

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
