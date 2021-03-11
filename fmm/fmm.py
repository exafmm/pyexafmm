"""
Implementation of the main FMM loop.
"""
import os
import pathlib

import h5py
import numpy as np
import numba

import adaptoctree.morton as morton
from adaptoctree.utils import deterministic_hash

import fmm.surface as surface
from fmm.kernel import KERNELS
from fmm.parameters import DIGEST_SIZE

import utils.data as data

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent
WORKING_DIR = pathlib.Path(os.getcwd())


def _particle_to_multipole(
        key,
        sources,
        source_densities,
        sources_to_keys,
        multipole_expansions,
        x0,
        r0,
        alpha_outer,
        check_surface,
        uc2e_inv,
        scale_function,
        p2p_function,
    ):
    """
    P2M operator. Form a multipole expansion from source points within a given
        source node.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    sources : np.array(shape=(nsources, 3), dtype=np.float32)
        Source coordinates.
    source_densities : np.array(shape=(nsources, 1), dtype=np.float32)
        Charge densities at source points.
    sources_to_keys : np.array(shape=(nsources, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) source lies.
    multipole_expansions : {np.int64: np.array(shape=(nequivalent_points)}
        Dictionary containing multipole expansions, indexed by Morton key of
        source nodes.
    x0 : np.array(shape=(1, 3), dtype=np.float32)
        Physical center of octree root node.
    r0 : np.float32
        Half side length of octree root node.
    alpha_outer : np.float32
        Relative size of outer surface
    check_surface : np.array(shape=(n_check, 3), dtype=np.float32)
        Discretised check surface.
    uc2e_inv : np.array(shape=(n_check, n_equivalent), dtype=np.float64)
    scale_function : function
        Function handle for kernel scaling.
    p2p_function : function
        Function handle for kernel P2P.
    """
    # Source indices in a given leaf
    source_indices = sources_to_keys == key

    # Find leaf sources, and leaf source densities
    leaf_sources = sources[source_indices]
    leaf_source_densities = source_densities[source_indices]

    # Compute center of leaf box in cartesian coordinates
    leaf_center = morton.find_physical_center_from_key(
        key=key, x0=x0, r0=r0
    )

    leaf_level = morton.find_level(key)

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

    upward_equivalent_density = (uc2e_inv @ check_potential)
    multipole_expansions[key] += (scale*upward_equivalent_density)


def _multipole_to_multipole(
        key,
        multipole_expansions,
        m2m,
    ):
    """
    M2M operator. Add the contribution of the multipole expansions of a given
        source node's children to it's own multipole expansion.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    multipole_expansions : {np.int64: np.array(shape=(nequivalent_points)}
        Dictionary containing multipole expansions, indexed by Morton key of
        source nodes.
    m2m : np.array(shape=(8, n_equivalent, n_equivalent), dtype=np.float32)
        Unscaled pre-computed M2M operators for all children. Implicitly
            indexed by order of Morton encoding from
            adaptoctree.morton.find_children.
    """
    children = morton.find_children(key)

    for child in children:

        if child in multipole_expansions:

            #  Compute operator index
            operator_idx = np.where(children == child)[0]

            # Get child equivalent density
            child_equivalent_density = multipole_expansions[child]

            # Compute parent equivalent density
            parent_equivalent_density = (
                m2m[operator_idx] @ child_equivalent_density
            )

            # Add to source data
            multipole_expansions[key] += np.ravel(parent_equivalent_density)


def _multipole_to_local(
        key,
        depth,
        v_list,
        multipole_expansions,
        local_expansions,
        dc2e_inv,
        ncheck_points,
        scale_function,
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
    multipole_expansions : {np.int64: np.array(shape=(nequivalent_points)}
        Dictionary containing multipole expansions, indexed by Morton key of
        source nodes.
    local_expansions : {np.int64: np.array(shape=(ncheck_points)}
        Dictionary containing local expansions, indexed by Morton key of
        target nodes.
    dc2e_inv : np.array(shape=(n_equivalent, n_check), dtype=np.float64)
    ncheck_points : np.int64
        Number of points discretising the check surface.
    m2l : h5py.Group
        HDF5 group, indexed by source node key, storing compressed M2L
        components.
    scale_function : function
        Function handle for kernel scaling.
    """
    level = morton.find_level(key)
    scale = np.float32(scale_function(level))

    transfer_vectors = morton.find_transfer_vectors(key, v_list, depth)
    hash_vectors = np.zeros(len(transfer_vectors), dtype=np.int64)
    m2l_lidxs = np.zeros(len(v_list), np.int32)
    m2l_ridxs = np.zeros(len(v_list), np.int32)

    for i in range(len(transfer_vectors)):
        # 940ns (64bit hashes)
        hash_vectors[i] = deterministic_hash(transfer_vectors[i], digest_size=DIGEST_SIZE)

        # 1.48mus
        m2l_idx = np.where(hash_vectors[i] == hashes)[0][0]

        # 178ns each
        m2l_lidxs[i] = m2l_idx*ncheck_points
        m2l_ridxs[i] = (m2l_idx+1)*ncheck_points

    for idx in range(len(v_list)):

        # Find source densities for v list of the key
        # 99ns
        source = v_list[idx]

        # Find compressed M2L operator for this transfer vector
        # 382 ns
        u_sub = u[m2l_lidxs[idx]:m2l_ridxs[idx]]

        # Compute contribution from source, to the local expansion
        # 5.88mus
        local_expansions[key] += scale*(dc2e_inv @ (u_sub @ (s @ (vt @ multipole_expansions[source]))))


def _local_to_local(
        key,
        local_expansions,
        l2l,
     ):
    """
    L2L operator. Translate the local expansion of a parent node, to each of
        it's children.

    Parameters:
    -----------
    key : np.int64
        Morton key of source node.
    local_expansions : {np.int64: np.array(shape=(ncheck_points)}
        Dictionary containing local expansions, indexed by Morton key of
        target nodes.
    l2l : np.array(shape=(8, n_check, n_check), dtype=np.float32)
        Unscaled pre-computed L2L operators for all children. Implicitly
            indexed by order of Morton encoding from
            adaptoctree.morton.find_children.
    """
    parent_equivalent_density = local_expansions[key]
    children = morton.find_children(key)

    for child in children:

        if child in local_expansions:

            # Compute operator index
            operator_idx = child == children

            # Compute contribution to local expansion of child from parent
            child_equivalent_density = l2l[operator_idx] @ parent_equivalent_density
            local_expansions[child] += np.ravel(child_equivalent_density)

            # print(type(parent_equivalent_density[0]), type(l2l[operator_idx][0]))

def _source_to_local(
        key,
        x_list,
        sources,
        source_densities,
        sources_to_keys,
        local_expansions,
        x0,
        r0,
        alpha_inner,
        check_surface,
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
    x_list : np.array(shape=(n_x_list, 1), dtype=np.int64)
        Morton keys of X list members.
    sources : np.array(shape=(nsources, 3), dtype=np.float32)
        Source coordinates.
    source_densities : np.array(shape=(nsources, 1), dtype=np.float32)
        Charge densities at source points.
    sources_to_keys : np.array(shape=(nsources, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) source lies.
    local_expansions : {np.int64: np.array(shape=(ncheck_points)}
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
    dc2e_inv : np.array(shape=(n_equivalent, n_check), dtype=np.float64)
    scale_function : function
        Function handle for kernel scaling.
    p2p_function : function
        Function handle for kernel P2P.
    """
    level = morton.find_level(key)
    scale = scale_function(level)
    center = morton.find_physical_center_from_key(key, x0, r0)
    source_coodinates = []
    densities = []

    for source in x_list:
        source_indices = sources_to_keys == source
        if np.any(source_indices == True):
            source_coodinates.extend(sources[source_indices])
            densities.extend(source_densities[source_indices])

    source_coodinates = np.array(source_coodinates)
    densities = np.array(densities)

    if len(densities) > 0:

        downward_check_surface = surface.scale_surface(
            surf=check_surface,
            radius=r0,
            level=level,
            center=center,
            alpha=alpha_inner
        )

        downward_check_potential = p2p_function(
            sources=source_coodinates,
            targets=downward_check_surface,
            source_densities=densities
        )

        downward_equivalent_density = (dc2e_inv @ downward_check_potential)

        local_expansions[key] += (scale*downward_equivalent_density)


def _multipole_to_target(
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
    w_list : np.array(shape=(n_v_list, 1), dtype=np.int64)
        Morton keys of W list members.
    targets : np.array(shape=(ntargets, 3), dtype=np.float32)
        Target coordinates.
    targets_to_keys: np.array(shape=(ntargets, 1), dtype=np.int64)
        (Leaf) Morton key where corresponding (via index) target lies.
    targets_potentials : np.array(shape=(ntargets,), dtype=np.float32)
        Potentials at all target points, due to all source points.
    multipole_expansions : {np.int64: np.array(shape=(nequivalent_points)}
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
    # Find target particles
    target_indices = targets_to_keys == key
    target_coordinates = targets[target_indices]

    for source in w_list:

        source_level = morton.find_level(source)
        source_center = morton.find_physical_center_from_key(source, x0, r0)

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


def _local_to_target(
        key,
        targets,
        targets_to_keys,
        target_potentials,
        local_expansions,
        x0,
        r0,
        alpha_outer,
        equivalent_surface,
        p2p_function
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
    local_expansions : {np.int64: np.array(shape=(ncheck_points)}
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
    level = morton.find_level(key)
    center = morton.find_physical_center_from_key(key, x0, r0)

    downward_equivalent_surface = surface.scale_surface(
        equivalent_surface,
        r0,
        level,
        center,
        alpha_outer
    )

    target_idxs = key == targets_to_keys
    target_coordinates = targets[target_idxs]

    target_potentials[target_idxs] += p2p_function(
        sources=downward_equivalent_surface,
        targets=target_coordinates,
        source_densities=local_expansions[key]
    )


def _near_field(
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


class Fmm:
    """
    FMM class. Configure with pre-computed operators and octree.

    Parameters:
    -----------
    config_filename : str
        Filename of configuration json file used to pre-compute operators.
    """

    def __init__(self, config_filename=None):

        # Load experimental database
        if config_filename is not None:
            config_filepath = WORKING_DIR / f"{config_filename}.json"
        else:
            config_filepath = WORKING_DIR / "config.json"

        self.config = data.load_json(config_filepath)

        db_filepath = WORKING_DIR / f"{self.config['experiment']}.hdf5"
        self.db = h5py.File(db_filepath, "r")

        # Load required data from disk
        ## Load surfaces, and inverse gram matrices
        self.check_surface = self.db["surface"]["check"][...]
        self.ncheck_points = len(self.check_surface)
        self.equivalent_surface = self.db["surface"]["equivalent"][...]
        self.nequivalent_points = len(self.equivalent_surface)
        self.uc2e_inv = self.db["uc2e_inv"][...]
        self.dc2e_inv = self.db["dc2e_inv"][...]
        self.alpha_outer = np.float32(self.config['alpha_outer'])
        self.alpha_inner = np.float32(self.config['alpha_inner'])

        ## Load linear, and complete octrees alongside their parameters
        self.x0 = self.db["octree"]["x0"][...].astype(np.float32)
        self.r0 = np.float32(self.db["octree"]["r0"][...][0])
        self.depth = self.db["octree"]["depth"][...][0]
        self.leaves = self.db["octree"]["keys"][...]
        self.nleaves = len(self.leaves)
        self.complete = self.db["octree"]["complete"][...]
        self.ncomplete = len(self.complete)
        self.complete_levels = morton.find_level(self.complete)

        ## Load source and target data
        self.sources = self.db["particle_data"]["sources"][...].astype(np.float32)
        self.nsources = len(self.sources)
        self.source_densities = self.db["particle_data"]["source_densities"][...].astype(np.float32)
        self.sources_to_keys = self.db["particle_data"]["sources_to_keys"][...]
        self.targets = self.db["particle_data"]["targets"][...].astype(np.float32)
        self.ntargets = len(self.targets)
        self.targets_to_keys = self.db["particle_data"]["targets_to_keys"][...]

        ## Load pre-computed operators
        self.m2m = self.db["m2m"][...]
        self.m2l = self.db["m2l"]
        self.l2l = self.db["l2l"][...]

        # Load interaction lists
        self.v_lists = self.db["interaction_lists"]["v"]
        self.x_lists = self.db["interaction_lists"]["x"]
        self.u_lists = self.db["interaction_lists"]["u"]
        self.w_lists = self.db["interaction_lists"]["w"]

        # Configure a kernel
        self.kernel = self.config["kernel"]
        self.eval = KERNELS[self.kernel]["eval"]
        self.p2p = KERNELS[self.kernel]["p2p"]
        self.scale = KERNELS[self.kernel]["scale"]

        #  Containers for results
        self.target_potentials = np.zeros(self.ntargets, dtype=np.float32)

        self.multipole_expansions = {
            key: np.zeros(self.nequivalent_points, dtype=np.float32) for key in self.complete
        }

        self.local_expansions = {
            key: np.zeros(self.nequivalent_points, dtype=np.float32) for key in self.complete
        }

    def upward_pass(self):
        """
        Post-order traversal of tree, compute multipole expansions for all
            nodes.
        """

        # Form multipole expansions for all leaf nodes
        for idx in range(self.nleaves):
            leaf = self.leaves[idx]
            self.particle_to_multipole(leaf)

        # Post-order traversal
        for level in range(self.depth-1, -1, -1):
            idxs = self.complete_levels == level
            for key in self.complete[idxs]:
                self.multipole_to_multipole(key)

    def downward_pass(self):
        """
        Pre-order traversal of tree. Compute local expansions for all nodes,
            and evaluate these at target points.
        """

        # Pre-order traversal of octree
        for level in range(2, self.depth + 1):

            idxs = self.complete_levels == level

            #  M2L operator stored in terms of its SVD components for each level
            str_level = str(level)
            u = self.m2l[str_level]["u"][...]
            s = np.diag(self.m2l[str_level]["s"][...])
            vt = self.m2l[str_level]["vt"][...]

            # Hashed transfer vectors for a given level, provide index for M2L operators
            hashes = self.m2l[str_level]["hashes"][...]

            for key in self.complete[idxs]:

                idx = np.where(self.complete== key)[0]

                v_list = self.v_lists[idx]
                v_list = v_list[v_list != -1]

                # V List interactions 947 mus
                # import IPython; IPython.embed()
                self.multipole_to_local(key, v_list, u, s, vt, hashes)
                # import sys; sys.exit()

                # X List interactions
                idx = np.where(self.complete == key)
                x_list = self.x_lists[idx]
                x_list = x_list[x_list != -1]
                if len(x_list) > 0:
                    self.source_to_local(key, x_list)

                # Translate local expansion to the node's children
                if level < self.depth:
                    self.local_to_local(key)

        # Leaf near-field computations
        for key in self.leaves:

            idx = np.where(self.complete == key)

            w_list = self.w_lists[idx]
            w_list = w_list[w_list != -1]

            u_list = self.u_lists[idx]
            u_list = u_list[u_list != -1]

            # import IPython; IPython.embed()
            # Evaluate local expansions at targets 291mus
            self.local_to_target(key)

            # W List interactions  6.2mus
            self.multipole_to_target(key, w_list)

            # U List interactions 887 mus
            self.near_field(key, u_list)
            # import sys; sys.exit()

    def run(self):
        """Run full algorithm"""
        self.upward_pass()
        self.downward_pass()

    def particle_to_multipole(self, key):
        """Compute multipole expansions from leaf particles."""
        _particle_to_multipole(
            key=key,
            sources=self.sources,
            source_densities=self.source_densities,
            sources_to_keys=self.sources_to_keys,
            multipole_expansions=self.multipole_expansions,
            x0=self.x0,
            r0=self.r0,
            alpha_outer=self.alpha_outer,
            check_surface=self.check_surface,
            uc2e_inv=self.uc2e_inv,
            scale_function=self.scale,
            p2p_function=self.p2p,
        )

    def multipole_to_multipole(self, key):
        """
        Combine multipole expansions of a node's children to approximate its
            own multipole expansion.
        """
        _multipole_to_multipole(
            key=key,
            multipole_expansions=self.multipole_expansions,
            m2m=self.m2m,
        )

    def multipole_to_local(self, key, v_list, u, s, vt, hashes):
        """
        V List interactions.
        """
        _multipole_to_local(
            key=key,
            depth=self.depth,
            v_list=v_list,
            multipole_expansions=self.multipole_expansions,
            local_expansions=self.local_expansions,
            dc2e_inv=self.dc2e_inv,
            ncheck_points=self.ncheck_points,
            scale_function=self.scale,
            u=u,
            s=s,
            vt=vt,
            hashes=hashes
        )

    def local_to_local(self, key):
        """
        Translate local expansion of a node to it's children.
        """
        _local_to_local(
            key=key,
            local_expansions=self.local_expansions,
            l2l=self.l2l,
        )

    def source_to_local(self, key, x_list):
        """
        X List interactions.
        """
        _source_to_local(
                key=key,
                x_list=x_list,
                sources=self.sources,
                source_densities=self.source_densities,
                sources_to_keys=self.sources_to_keys,
                local_expansions=self.local_expansions,
                x0=self.x0,
                r0=self.r0,
                alpha_inner=self.alpha_inner,
                check_surface=self.check_surface,
                dc2e_inv=self.dc2e_inv,
                scale_function=self.scale,
                p2p_function=self.p2p
        )

    def multipole_to_target(self, key, w_list):
        """
        W List interactions
        """
        _multipole_to_target(
            key=key,
            w_list=w_list,
            targets=self.targets,
            targets_to_keys=self.targets_to_keys,
            target_potentials=self.target_potentials,
            multipole_expansions=self.multipole_expansions,
            x0=self.x0,
            r0=self.r0,
            alpha_inner=self.alpha_inner,
            equivalent_surface=self.equivalent_surface,
            p2p_function=self.p2p
        )

    def local_to_target(self, key):
        """
        Evaluate local potentials at target points
        """
        _local_to_target(
            key=key,
            targets=self.targets,
            targets_to_keys=self.targets_to_keys,
            target_potentials=self.target_potentials,
            local_expansions=self.local_expansions,
            x0=self.x0,
            r0=self.r0,
            alpha_outer=self.alpha_outer,
            equivalent_surface=self.equivalent_surface,
            p2p_function=self.p2p
        )

    def near_field(self, key, u_list):
        """
        U List interactions
        """
        _near_field(
            key=key,
            u_list=u_list,
            targets=self.targets,
            targets_to_keys=self.targets_to_keys,
            target_potentials=self.target_potentials,
            sources=self.sources,
            source_densities=self.source_densities,
            sources_to_keys=self.sources_to_keys,
            p2p_function=self.p2p
        )
