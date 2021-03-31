"""
Compute operators accelerated with Numba and OpenMP.
"""
import numpy as np
import numba

from adaptoctree import morton

import fmm.surface as surface


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

        # # Compute center of leaf box in cartesian coordinates
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


    # Prepare data using Numba
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

    # Launch OpenMP for matvec

    pass



def m2m():
    pass

def l2l():
    pass

def m2l():
    pass

def s2l():
    pass

def l2t():
    pass

def m2t():
    pass

def near_field():
    pass