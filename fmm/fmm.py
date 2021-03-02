"""Implementation of the main FMM loop."""
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
        self.db = h5py.File(db_filepath, 'r')

        # Coefficients discretising surface of a node
        # self.ncoefficients = 6*(self.order-1)**2 + 2

        # Load required data from disk
        self.check_surface = self.db['surface']['check'][...]
        self.equivalent_surface = self.db['surface']['equivalent'][...]
        self.nequivalent_points = len(self.equivalent_surface)
        self.uc2e_inv = self.db['uc2e_inv'][...]

        self.x0 = self.db['octree']['x0'][...]
        self.r0 = self.db['octree']['r0'][...]
        self.depth = self.db['octree']['depth'][...][0]
        self.leaves = self.db['octree']['keys'][...]
        self.nleaves = len(self.leaves)
        self.complete = self.db['octree']['complete'][...]
        self.ncomplete = len(self.complete)
        self.complete_levels = morton.find_level(self.complete)

        self.sources = self.db['particle_data']['sources'][...]
        self.nsources = len(self.sources)
        self.source_densities = self.db['particle_data']['source_densities'][...]
        self.sources_to_keys = self.db['particle_data']['sources_to_keys'][...]

        self.kernel = self.config['kernel']
        self.eval = KERNELS[self.kernel]['eval']
        self.p2p = KERNELS[self.kernel]['p2p']
        self.scale = KERNELS[self.kernel]['scale']

        self.m2m = self.db['m2m'][...]

        # Containers for results
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

        # Post-order traversal of octree, translating multipole expansions from
        # leaf nodes to root
        for level in range(self.depth-1, -1, -1):
            idxs = self.complete_levels == level
            for key in self.complete[idxs]:
                self.multipole_to_multipole(key)

    def downward_pass(self):
        """Downward pass loop."""

        # Pre-order traversal of octree
        for level in range(2, 1 + self.octree.maximum_level):

            for target in self.octree.non_empty_target_nodes_by_level[level]:
                # Translate multipole expansions from non-empty source nodes
                # in this target's interaction list to a local expansion about
                # the target.
                self.multipole_to_local(target)

                # Translate local expansion to the node's children
                if level < self.octree.maximum_level:
                    self.local_to_local(target)

        # Treat local expansion as charge density, and evaluate at each particle
        # at leaf node, compute near field.
        for leaf_node_index in range(len(self.octree.target_index_ptr) - 1):
            self.local_to_particle(leaf_node_index)
            self.compute_near_field(leaf_node_index)

    def particle_to_multipole(self, leaf):
        """Compute multipole expansions from leaf particles."""

        # Source indices in a given leaf
        source_indices = (self.sources_to_keys == leaf)

        # Find leaf sources, and leaf source densities
        leaf_sources = self.sources[source_indices]
        leaf_source_densities = self.source_densities[source_indices]

        # Compute center of leaf box in cartesian coordinates
        leaf_center = morton.find_physical_center_from_key(
            key=leaf,
            x0=self.x0,
            r0=self.r0
        )

        leaf_level = morton.find_level(leaf)

        upward_check_surface = operator.scale_surface(
            surface=self.check_surface,
            radius=self.r0,
            level=leaf_level,
            center=leaf_center,
            alpha=self.config['alpha_outer']
        )

        scale = self.scale(leaf_level)

        check_potential = self.p2p(
            targets=upward_check_surface,
            sources=leaf_sources,
            source_densities=leaf_source_densities
        )

        upward_equivalent_density = scale*self.uc2e_inv @ check_potential
        self.upward_equivalent_densities[leaf] += upward_equivalent_density

    def multipole_to_multipole(self, key):
        """
        Combine multipole expansions of a node's children to approximate its
            own multipole expansion.
        """

        children = morton.find_children(key)

        for child in children:

            # Compute operator index
            operator_idx = np.where(children==child)[0]

            # Get child equivalent density
            child_equivalent_density = self.upward_equivalent_densities[child]

            # Compute parent equivalent density
            parent_equivalent_density = self.m2m[operator_idx] @ child_equivalent_density

            # Add to source data
            self.upward_equivalent_densities[key] += np.ravel(parent_equivalent_density)


    def multipole_to_local(self, target_key):
        """
        Translate all multipole expansions of source nodes in target node's
            interaction list into a local expansion about the target node.
        """

        # Lookup all non-empty source nodes in target's interaction list, and
        # vstack their equivalent densities

        target_index = self.octree.target_node_to_index[target_key]

        source_equivalent_density = None
        for neighbor_list in self.octree.interaction_list[target_index]:

            for source_key in neighbor_list:
                if source_key != -1:

                    if source_equivalent_density is None:
                        source_equivalent_density = self.source_data[source_key].expansion
                    else:

                        source_equivalent_density = np.vstack(
                                (
                                    source_equivalent_density,
                                    self.source_data[source_key].expansion
                                )
                            )
        source_equivalent_density = np.ravel(source_equivalent_density)

        # Lookup (compressed) m2l operator
        m2l_operator = self.m2l[target_key]

        # M2L operator stored in terms of its SVD components
        u, s, vt = m2l_operator

        # Calculate target equivalent density
        target_equivalent_density = np.matmul(vt, source_equivalent_density)
        target_equivalent_density = np.matmul(np.diag(s), target_equivalent_density)
        target_equivalent_density = np.matmul(u, target_equivalent_density)

        self.target_data[target_key].expansion += target_equivalent_density

    def local_to_local(self, key):
        """Translate local expansion of a node to it's children."""

        parent_equivalent_density = self.target_data[key].expansion

        for child in hilbert.get_children(key):
            if self.octree.target_node_to_index[child] != -1:

                # Compute operator index
                operator_idx = (child % 8) - 1

                # Updating indices
                self.target_data[child].indices.update(
                    self.target_data[key].indices
                )

                child_equivalent_density = np.matmul(
                    self.l2l[operator_idx], parent_equivalent_density
                )

                self.target_data[child].expansion = child_equivalent_density

    def local_to_particle(self, leaf_index):
        """
        Directly evaluate potential at particles in a leaf node, treating the
            local expansion points as sources.
        """

        target_indices = self.octree.targets_by_leafs[
            self.octree.target_index_ptr[leaf_index]
            : self.octree.target_index_ptr[leaf_index + 1]
        ]

        leaf_key = self.octree.target_leaf_nodes[leaf_index]
        leaf_center = hilbert.get_center_from_key(
            leaf_key, self.octree.center, self.octree.radius)

        leaf_density = self.target_data[leaf_key].expansion

        leaf_surface = operator.scale_surface(
            surface=self.surface,
            radius=self.octree.radius,
            level=self.octree.maximum_level,
            center=leaf_center,
            alpha=self.config['alpha_outer']
        )

        for target_index in target_indices:

            # Updating indices
            self.result_data[target_index].indices.update(
                self.target_data[leaf_key].indices
            )

            target = self.octree.targets[target_index].reshape(1, 3)

            result = operator.p2p(
                kernel_function=self.kernel_function,
                targets=target,
                sources=leaf_surface,
                source_densities=leaf_density
            )

            self.result_data[target_index].density = result.density

    def compute_near_field(self, leaf_node_index):
        """
        Compute near field influence from neighbouring box's local expansions.
        """

        target_indices = self.octree.targets_by_leafs[
            self.octree.target_index_ptr[leaf_node_index]
            : self.octree.target_index_ptr[leaf_node_index + 1]
        ]

        leaf_node_key = self.octree.target_leaf_nodes[leaf_node_index]

        for neighbor_key in self.octree.target_neighbors[
                self.octree.target_node_to_index[leaf_node_key]
                ]:
            if neighbor_key != -1:

                neighbor_index = self.octree.source_node_to_index[neighbor_key]

                neighbor_source_indices = self.octree.sources_by_leafs[
                    self.octree.source_index_ptr[neighbor_index]:
                    self.octree.source_index_ptr[neighbor_index + 1]
                ]

                neighbor_sources = self.octree.sources[neighbor_source_indices]
                neighbor_source_densities = self.octree.source_densities[neighbor_source_indices]

                for target_index in target_indices:

                    # Updating indices
                    self.result_data[target_index].indices.update(
                        self.source_data[neighbor_key].indices
                    )

                    target = self.octree.targets[target_index].reshape(1, 3)

                    result = operator.p2p(
                        kernel_function=self.kernel_function,
                        targets=target,
                        sources=neighbor_sources,
                        source_densities=neighbor_source_densities
                    )

                    self.result_data[target_index].density += result.density

        if self.octree.source_node_to_index[leaf_node_key] != -1:

            leaf_index = self.octree.source_node_to_index[leaf_node_key]

            leaf_source_indices = self.octree.sources_by_leafs[
                self.octree.source_index_ptr[leaf_index]:
                self.octree.source_index_ptr[leaf_index + 1]
            ]

            leaf_sources = self.octree.sources[leaf_source_indices]
            leaf_source_densities = self.octree.source_densities[leaf_source_indices]

            for target_index in target_indices:
                target = self.octree.targets[target_index].reshape(1, 3)

                result = operator.p2p(
                    kernel_function=self.kernel_function,
                    targets=target,
                    sources=leaf_sources,
                    source_densities=leaf_source_densities
                )

                self.result_data[target_index].density += result.density

                # Updating indices
                self.result_data[target_index].indices.update(
                    self.source_data[leaf_node_key].indices)
