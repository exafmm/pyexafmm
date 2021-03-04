"""
Implementation of the main FMM loop.
"""

import os
import pathlib

import h5py
import numpy as np

import adaptoctree.morton as morton

import fmm.surface as surface
from fmm.kernel import KERNELS

import utils.data as data

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent


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
        level=leaf_level,
        center=leaf_center,
        alpha=alpha_outer,
    )

    scale = scale_function(leaf_level)

    check_potential = p2p_function(
        targets=upward_check_surface,
        sources=leaf_sources,
        source_densities=leaf_source_densities,
    )

    upward_equivalent_density = scale * uc2e_inv @ check_potential
    multipole_expansions[key] += upward_equivalent_density


def _multipole_to_multipole(
        key,
        multipole_expansions,
        m2m,
    ):

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
        scale_function,
        complete_tree,
        v_list,
        dc2e_inv,
        multipole_expansions,
        local_expansions,
        m2l,
    ):

    level = morton.find_level(key)
    scale = scale_function(level)

    #  Find source densities for v list of the key

    source_equivalent_density = []
    for source in v_list:
        source_equivalent_density.extend(multipole_expansions[source])

    source_equivalent_density = np.array(source_equivalent_density)

    #  M2L operator stored in terms of its SVD components
    str_key = str(key)
    u = m2l[str_key]["U"][...]
    s = m2l[str_key]["S"][...]
    vt = m2l[str_key]["VT"][...]

    # Calculate target equivalent density, from assembled M2L matrix
    target_equivalent_density = (scale*dc2e_inv) @ (u @ np.diag(s) @ vt).T @ source_equivalent_density
    local_expansions[key] += target_equivalent_density


def _local_to_local(
        key,
        local_expansions,
        l2l,
     ):

    parent_equivalent_density = local_expansions[key]
    children = morton.find_children(key)

    for child in children:

        if child in local_expansions:

            #  Compute operator index
            operator_idx = child == children

            child_equivalent_density = l2l[operator_idx] @ parent_equivalent_density

            local_expansions[child] += np.ravel(child_equivalent_density)


def _source_to_local(
        key,
        x_list,
        sources,
        source_densities,
        sources_to_keys,
        check_surface,
        dc2e_inv,
        alpha_inner,
        x0,
        r0,
        local_expansions,
        p2p_function,
        scale_function
    ):

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

        downward_equivalent_density = (scale*dc2e_inv) @ downward_check_potential

        local_expansions[key] += downward_equivalent_density


def _multipole_to_target(
        key,
        x_list,
        multipole_expansions,
        targets,
        targets_to_keys,
        target_potentials,
        equivalent_surface,
        x0,
        r0,
        alpha_inner,
        p2p_function
    ):
    # Find target particles
    target_indices = targets_to_keys == key
    target_coordinates = targets[target_indices]

    for source in x_list:

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
        target_potentials,
        targets_to_keys,
        local_expansions,
        equivalent_surface,
        alpha_outer,
        x0,
        r0,
        p2p_function
    ):

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
        sources_to_keys,
        source_densities,
        p2p_function
    ):

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
            config_filepath = PARENT / f"{config_filename}.json"
        else:
            config_filepath = PARENT / "config.json"

        self.config = data.load_json(config_filepath)

        db_filepath = PARENT / f"{self.config['experiment']}.hdf5"
        self.db = h5py.File(db_filepath, "r")

        # Load required data from disk

        ## Load surfaces, and inverse gram matrices
        self.check_surface = self.db["surface"]["check"][...]
        self.equivalent_surface = self.db["surface"]["equivalent"][...]
        self.nequivalent_points = len(self.equivalent_surface)
        self.uc2e_inv = self.db["uc2e_inv"][...]
        self.dc2e_inv = self.db["dc2e_inv"][...]

        ## Load linear, and complete octrees alongside their parameters
        self.x0 = self.db["octree"]["x0"][...]
        self.r0 = self.db["octree"]["r0"][...]
        self.depth = self.db["octree"]["depth"][...][0]
        self.leaves = self.db["octree"]["keys"][...]
        self.nleaves = len(self.leaves)
        self.complete = self.db["octree"]["complete"][...]
        self.ncomplete = len(self.complete)
        self.complete_levels = morton.find_level(self.complete)

        ## Load source and target data
        self.sources = self.db["particle_data"]["sources"][...]
        self.nsources = len(self.sources)
        self.source_densities = self.db["particle_data"]["source_densities"][...]
        self.sources_to_keys = self.db["particle_data"]["sources_to_keys"][...]
        self.targets = self.db["particle_data"]["targets"][...]
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
        self.target_potentials = np.zeros(self.ntargets)

        self.multipole_expansions = {
            key: np.zeros(self.nequivalent_points) for key in self.complete
        }

        self.local_expansions = {
            key: np.zeros(self.nequivalent_points) for key in self.complete
        }

    def upward_pass(self):
        """
        Post order traversal of tree, compute multipole expansion from sources
            and transfer to their parents, until multipole expansions are
            obtained for all nodes.
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
        """Downward pass loop."""

        # Pre-order traversal of octree
        for level in range(2, self.depth + 1):

            idxs = self.complete_levels == level

            for key in self.complete[idxs]:

                idx = np.where(self.complete== key)[0]

                v_list = self.v_lists[idx]
                v_list = v_list[v_list != -1]

                # V List interactions
                self.multipole_to_local(key, v_list)

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

            # Evaluate local expansions at targets
            self.local_to_target(key)

            # W List interactions
            self.multipole_to_target(key, w_list)

            # U List interactions
            self.near_field(key, u_list)

    def run(self):
        """Run full algorithm"""
        self.upward_pass()
        self.downward_pass()

    def particle_to_multipole(self, key):
        """Compute multipole expansions from leaf particles."""
        _particle_to_multipole(
            key,
            self.sources,
            self.source_densities,
            self.sources_to_keys,
            self.multipole_expansions,
            self.x0,
            self.r0,
            self.config["alpha_outer"],
            self.check_surface,
            self.uc2e_inv,
            self.scale,
            self.p2p,
        )

    def multipole_to_multipole(self, key):
        """
        Combine multipole expansions of a node's children to approximate its
            own multipole expansion.
        """
        _multipole_to_multipole(
            key,
            self.multipole_expansions,
            self.m2m,
        )

    def multipole_to_local(self, key, v_list):
        """
        V List interactions.
        """
        _multipole_to_local(
            key,
            self.scale,
            self.complete,
            v_list,
            self.dc2e_inv,
            self.multipole_expansions,
            self.local_expansions,
            self.m2l,
        )

    def local_to_local(self, key):
        """
        Translate local expansion of a node to it's children.
        """
        _local_to_local(
            key,
            self.local_expansions,
            self.l2l,
        )

    def source_to_local(self, key, x_list):
        """
        X List interactions.
        """
        _source_to_local(
                key,
                x_list,
                self.sources,
                self.source_densities,
                self.sources_to_keys,
                self.check_surface,
                self.dc2e_inv,
                self.config['alpha_inner'],
                self.x0,
                self.r0,
                self.local_expansions,
                self.p2p,
                self.scale
        )

    def multipole_to_target(self, key, x_list):
        """
        W List interactions
        """
        _multipole_to_target(
            key,
            x_list,
            self.multipole_expansions,
            self.targets,
            self.targets_to_keys,
            self.target_potentials,
            self.equivalent_surface,
            self.x0,
            self.r0,
            self.config["alpha_inner"],
            self.p2p
        )

    def local_to_target(self, key):
        """
        Evaluate local potentials at target points
        """
        _local_to_target(
            key,
            self.targets,
            self.target_potentials,
            self.targets_to_keys,
            self.local_expansions,
            self.equivalent_surface,
            self.config["alpha_outer"],
            self.x0,
            self.r0,
            self.p2p
        )

    def near_field(self, key, u_list):
        """
        U List interactions
        """
        _near_field(
            key,
            u_list,
            self.targets,
            self.targets_to_keys,
            self.target_potentials,
            self.sources,
            self.sources_to_keys,
            self.source_densities,
            self.p2p
        )
