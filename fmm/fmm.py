"""Implementation of the main FMM loop."""
import os
import pathlib

import h5py
import numpy as np

import adaptoctree.morton as morton
from numpy.lib.utils import source

import fmm.operator as operator
from fmm.kernel import KERNELS

import utils.data as data

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent


class Fmm:
    """
    Main FMM loop.
    Initialising with an Octree, run the FMM loop with a specified kernel
    function, computing the expansion at a given precision specified by an
    expansion order.
    """

    def __init__(self, config_filename=None):

        # Load experimental database
        if config_filename is not None:
            config_filepath = PARENT / config_filename
        else:
            config_filepath = PARENT / "config.json"

        self.config = data.load_json(config_filepath)

        db_filepath = PARENT / f"{self.config['experiment']}.hdf5"
        self.db = h5py.File(db_filepath, "r")

        # Coefficients discretising surface of a node
        # self.ncoefficients = 6*(self.order-1)**2 + 2

        # Load required data from disk
        self.check_surface = self.db["surface"]["check"][...]
        self.equivalent_surface = self.db["surface"]["equivalent"][...]
        self.nequivalent_points = len(self.equivalent_surface)
        self.uc2e_inv = self.db["uc2e_inv"][...]
        self.dc2e_inv = self.db["dc2e_inv"][...]

        self.x0 = self.db["octree"]["x0"][...]
        self.r0 = self.db["octree"]["r0"][...]
        self.depth = self.db["octree"]["depth"][...][0]
        self.leaves = self.db["octree"]["keys"][...]
        self.nleaves = len(self.leaves)
        self.complete = self.db["octree"]["complete"][...]
        self.ncomplete = len(self.complete)
        self.complete_levels = morton.find_level(self.complete)

        self.sources = self.db["particle_data"]["sources"][...]
        self.nsources = len(self.sources)
        self.source_densities = self.db["particle_data"]["source_densities"][...]
        self.sources_to_keys = self.db["particle_data"]["sources_to_keys"][...]

        self.kernel = self.config["kernel"]
        self.eval = KERNELS[self.kernel]["eval"]
        self.p2p = KERNELS[self.kernel]["p2p"]
        self.scale = KERNELS[self.kernel]["scale"]

        self.m2m = self.db["m2m"][...]
        self.m2l = self.db["m2l"]
        self.l2l = self.db["l2l"][...]

        self.v_lists = self.db["interaction_lists"]["v"]
        self.x_lists = self.db["interaction_lists"]["x"]

        #  Containers for results
        self.result_data = []

        self.upward_equivalent_densities = {
            key: np.zeros(self.nequivalent_points) for key in self.complete
        }

        self.downward_equivalent_densities = {
            key: np.zeros(self.nequivalent_points) for key in self.complete
        }

    def upward_pass(self):
        """Upward pass loop."""

        # Form multipole expansions for all leaf nodes
        for idx in range(self.nleaves):
            leaf = self.leaves[idx]
            self.particle_to_multipole(leaf)

        # Post-order traversal of octree
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
                self.source_to_local(key)

                # Translate local expansion to the node's children
                if level < self.depth:
                    self.local_to_local(key)

        # Treat local expansion as charge density, and evaluate at each particle
        # at leaf node, compute near field.
        # for leaf_node_index in range(len(self.octree.target_index_ptr) - 1):
        #     self.local_to_particle(leaf_node_index)
        #     self.compute_near_field(leaf_node_index)

    def particle_to_multipole(self, leaf):
        """Compute multipole expansions from leaf particles."""

        # Source indices in a given leaf
        source_indices = self.sources_to_keys == leaf

        # Find leaf sources, and leaf source densities
        leaf_sources = self.sources[source_indices]
        leaf_source_densities = self.source_densities[source_indices]

        # Compute center of leaf box in cartesian coordinates
        leaf_center = morton.find_physical_center_from_key(
            key=leaf, x0=self.x0, r0=self.r0
        )

        leaf_level = morton.find_level(leaf)

        upward_check_surface = operator.scale_surface(
            surface=self.check_surface,
            radius=self.r0,
            level=leaf_level,
            center=leaf_center,
            alpha=self.config["alpha_outer"],
        )

        scale = self.scale(leaf_level)

        check_potential = self.p2p(
            targets=upward_check_surface,
            sources=leaf_sources,
            source_densities=leaf_source_densities,
        )

        upward_equivalent_density = scale * self.uc2e_inv @ check_potential
        self.upward_equivalent_densities[leaf] += upward_equivalent_density

    def multipole_to_multipole(self, key):
        """
        Combine multipole expansions of a node's children to approximate its
            own multipole expansion.
        """

        children = morton.find_children(key)

        for child in children:

            #  Compute operator index
            operator_idx = np.where(children == child)[0]

            # Get child equivalent density
            child_equivalent_density = self.upward_equivalent_densities[child]

            # Compute parent equivalent density
            parent_equivalent_density = (
                self.m2m[operator_idx] @ child_equivalent_density
            )

            # Add to source data
            self.upward_equivalent_densities[key] += np.ravel(parent_equivalent_density)

    def multipole_to_local(self, key):
        """
        V List interactions.
        """

        level = morton.find_level(key)
        scale = self.scale(level)

        #  Find source densities for v list of the key
        idx = np.where(self.complete == key)[0]

        v_list = self.v_lists[idx]
        v_list = v_list[v_list != -1]

        source_equivalent_density = []
        for source in v_list:
            source_equivalent_density.extend(self.upward_equivalent_densities[source])

        source_equivalent_density = np.array(source_equivalent_density)

        #  M2L operator stored in terms of its SVD components
        str_key = str(key)
        u = self.m2l[str_key]["U"][...]
        s = self.m2l[str_key]["S"][...]
        vt = self.m2l[str_key]["VT"][...]

        # Calculate target equivalent density, from assembled M2L matrix
        target_equivalent_density = (scale*self.dc2e_inv) @ (u @ np.diag(s) @ vt).T @ source_equivalent_density
        self.downward_equivalent_densities[key] += target_equivalent_density

    def local_to_local(self, key):
        """
        Translate local expansion of a node to it's children.
        """

        parent_equivalent_density = self.downward_equivalent_densities[key]
        children = morton.find_children(key)

        for child in children:

            #  Compute operator index
            operator_idx = child == children

            child_equivalent_density = (
                self.l2l[operator_idx] @ parent_equivalent_density
            )

            self.downward_equivalent_densities[child] += child_equivalent_density

    def source_to_local(self, key):
        """
        X List interactions.
        """

        idx = np.where(self.complete == key)[0]

        x_list = self.x_lists[idx]
        x_list = x_list[x_list != 0]

