"""
Implementation of the main FMM loop.
"""

import os
import pathlib

import h5py
import numpy as np

import adaptoctree.morton as morton

import fmm.operator as operator
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

    upward_check_surface = operator.scale_surface(
        surface=check_surface,
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
        v_lists,
        dc2e_inv,
        multipole_expansions,
        local_expansions,
        m2l,
    ):

    level = morton.find_level(key)
    scale = scale_function(level)

    #  Find source densities for v list of the key
    idx = np.where(complete_tree == key)[0]

    v_list = v_lists[idx]
    v_list = v_list[v_list != -1]

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
    sources = []
    source_densities = []

    for source in x_list:
        source_indices = sources_to_keys == source

        # Find sources, and source densities
        sources.append(sources[source_indices])
        source_densities.append(source_densities[source_indices])

    sources = np.array(sources)
    source_densities = np.array(source_densities)

    downward_check_surface = operator.scale_surface(
        surface=check_surface,
        radius=r0,
        level=level,
        center=center,
        alpha=alpha_inner
    )

    downward_check_potential = p2p_function(
        sources=sources,
        targets=downward_check_surface,
        source_densities=source_densities
    )

    downward_equivalent_density = (scale*dc2e_inv) @ downward_check_potential
    local_expansions[key] += downward_equivalent_density



def _multipole_to_target(
        key,
        x_list,
        multipole_expansions,
        targets,
        target_potentials,
        equivalent_surface,
        x0,
        r0,
        alpha_inner,
        p2p_function
    ):

    # Find target particles
    target_indices = targets == key
    target_coordinates = targets[target_indices]

    for source in x_list:

        source_level = morton.find_level(source)
        source_center = morton.find_physical_center_from_key(source, x0, r0)

        upward_equivalent_surface = operator.scale_surface(
            surface=equivalent_surface,
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


def _near_field(key):
    pass


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
        for level in range(self.depth - 1, -1, -1):
            idxs = self.complete_levels == level
            for key in self.complete[idxs]:
                self.multipole_to_multipole(key)

    def downward_pass(self):
        """Downward pass loop."""

        # Pre-order traversal of octree
        for level in range(2, self.depth + 1):

            idxs = self.complete_levels == level

            for key in self.complete[idxs]:

                # V List interactions
                self.multipole_to_local(key)

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
            w_list = x_list[w_list != -1]

            u_list = self.u_lists[idx]
            u_list = x_list[u_list != -1]

            # W List interactions
            self.multipole_to_target(key, w_list)

            # U List interactions
            self.near_field(key, u_list)

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

    def multipole_to_local(self, key):
        """
        V List interactions.
        """
        _multipole_to_local(
            key,
            self.scale,
            self.complete,
            self.v_lists,
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
                self.config["alpha_inner"],
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
            self.target_potentials,
            self.equivalent_surface,
            self.x0,
            self.r0,
            self.config["alpha_inner"],
            self.p2p
        )

    def near_field(self, key):
        """
        U List interactions
        """
        _near_field(key)
