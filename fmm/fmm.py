"""Implementation of the main FMM loop."""
import os
import pathlib

import numpy as np

import fmm.hilbert as hilbert
from fmm.density import Potential
from fmm.node import Node
from fmm.operator import p2p, scale_surface, compute_m2l_operator_index
from fmm.kernel import KERNELS
from fmm.octree import Octree

from utils.data import load_json, load_hdf5_to_array

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

        if config_filename is not None:
            config_filepath = PARENT / config_filename
        else:
            config_filepath = PARENT / "config.json"


        self.config = load_json(config_filepath)

        data_dirpath = PARENT / self.config["data_dirname"]
        operator_dirpath = PARENT/ self.config["operator_dirname"]
        source_filename = self.config['source_filename']
        target_filename = self.config['target_filename']
        source_densities_filename = self.config['source_densities_filename']

        # Load sources, targets and source densities
        self.surface = load_hdf5_to_array('surface', 'surface', operator_dirpath)
        self.sources = load_hdf5_to_array(source_filename, source_filename, data_dirpath)
        self.targets = load_hdf5_to_array(target_filename, target_filename, data_dirpath)
        self.source_densities = load_hdf5_to_array(
            'source_densities', source_densities_filename, data_dirpath)

        # Load precomputed operators
        self.uc2e_u = load_hdf5_to_array('uc2e_u', 'uc2e_u', operator_dirpath)
        self.uc2e_v = load_hdf5_to_array('uc2e_v', 'uc2e_v', operator_dirpath)
        self.m2m = load_hdf5_to_array('m2m', 'm2m', operator_dirpath)
        self.l2l = load_hdf5_to_array('l2l', 'l2l', operator_dirpath)
        self.m2l = load_hdf5_to_array('m2l', 'm2l', operator_dirpath)
        self.sources_relative_to_targets = load_hdf5_to_array(
            'sources_relative_to_targets', 'sources_relative_to_targets',
            operator_dirpath
        )

        # Load configuration properties
        self.maximum_level = self.config['octree_max_level']
        self.kernel_function = KERNELS[self.config['kernel']]()
        self.order = self.config['order']
        self.octree = Octree(
            self.sources, self.targets, self.maximum_level, self.source_densities)

        # Coefficients discretising surface of a node
        self.ncoefficients = 6*(self.order-1)**2 + 2

        # Containers for results
        self.result_data = [
            Potential(target, np.zeros(1, dtype='float64'))
            for target in self.octree.targets
            ]

        self.source_data = {
            key: Node(key, self.ncoefficients)
            for key in self.octree.non_empty_source_nodes
        }

        self.target_data = {
            key: Node(key, self.ncoefficients)
            for key in self.octree.non_empty_target_nodes
        }

    def upward_pass(self):
        """Upward pass loop."""
        nleaves = len(self.octree.source_index_ptr) - 1

        # Form multipole expansions for all leaf nodes
        for index in range(nleaves):
            self.particle_to_multipole(index)

        # Post-order traversal of octree, translating multipole expansions from
        # leaf nodes to root
        for level in range(self.octree.maximum_level-1, -1, -1):
            for key in self.octree.non_empty_source_nodes_by_level[level]:
                self.multipole_to_multipole(key)

    def downward_pass(self):
        """Downward pass loop."""

        # Pre-order traversal of octree
        for level in range(2, 1 + self.octree.maximum_level):

            for key in self.octree.non_empty_target_nodes_by_level[level]:
                index = self.octree.target_node_to_index[key]

                # Translating mutlipole expansion in far field to a local
                # expansion at currently examined node.
                for neighbor_list in self.octree.interaction_list[index]:
                    for child in neighbor_list:
                        if child != -1:
                            self.multipole_to_local(child, key)

                # Translate local expansion to the node's children
                if level < self.octree.maximum_level:
                    self.local_to_local(key)

        # Treat local expansion as charge density, and evaluate at each particle
        # at leaf node, compute near field.
        for leaf_node_index in range(len(self.octree.target_index_ptr) - 1):
            self.local_to_particle(leaf_node_index)
            self.compute_near_field(leaf_node_index)

    def particle_to_multipole(self, leaf_node_index):
        """Compute particle to multipole interactions in leaf."""

        # Source indices in a given leaf
        source_indices = self.octree.sources_by_leafs[
            self.octree.source_index_ptr[leaf_node_index]
            : self.octree.source_index_ptr[leaf_node_index + 1]
        ]

        # Find leaf sources, and leaf source densities
        leaf_sources = self.octree.sources[source_indices]
        leaf_source_densities = self.octree.source_densities[source_indices]

        # Just adding index from argsort (sources by leafs)
        self.source_data[
            self.octree.source_leaf_nodes[leaf_node_index]
            ].indices.update(source_indices)

        # Compute key corresponding to this leaf index
        leaf_key = self.octree.non_empty_source_nodes[leaf_node_index]

        # Compute center of leaf box in cartesian coordinates
        leaf_center = hilbert.get_center_from_key(
            leaf_key, self.octree.center, self.octree.radius
        )

        upward_check_surface = scale_surface(
            surface=self.surface,
            radius=self.octree.radius,
            level=self.octree.maximum_level,
            center=leaf_center,
            alpha=self.config['alpha_outer']
        )

        scale = (1/2)**self.octree.maximum_level

        check_potential = p2p(
            kernel_function=self.kernel_function,
            targets=upward_check_surface,
            sources=leaf_sources,
            source_densities=leaf_source_densities
            ).density

        tmp = np.matmul(scale*self.uc2e_u, check_potential)
        upward_equivalent_density = np.matmul(self.uc2e_v, tmp)

        self.source_data[leaf_key].expansion = upward_equivalent_density

    def multipole_to_multipole(self, key):
        """Combine children expansions of node into node expansion."""

        for child in hilbert.get_children(key):
            # Only going through non-empty child nodes
            if self.octree.source_node_to_index[child] != -1:

                # Compute operator index
                operator_idx = (child % 8) - 1

                # Updating indices
                self.source_data[key].indices.update(
                    self.source_data[child].indices
                    )

                # Get child equivalent density
                child_equivalent_density = self.source_data[child].expansion

                # Compute parent equivalent density
                parent_equivalent_density = np.matmul(
                    self.m2m[operator_idx], child_equivalent_density)

                # Add to source data
                self.source_data[key].expansion += parent_equivalent_density

    def multipole_to_local(self, source_key, target_key):
        """
        Translate multipole expansions in target's interaction list to local
        expansions about the target.
        """

        # Updating indices
        self.target_data[target_key].indices.update(
            self.source_data[source_key].indices
        )

        source_equivalent_density = self.source_data[source_key].expansion

        # Compute 4D indice in order to lookup right (relative) m2l operator
        source_4d_idx = hilbert.get_4d_index_from_key(source_key)
        target_4d_idx = hilbert.get_4d_index_from_key(target_key)

        operator_idx = compute_m2l_operator_index(
            self.sources_relative_to_targets, source_4d_idx, target_4d_idx
            )

        operator = self.m2l[operator_idx]

        target_equivalent_density = np.matmul(operator, source_equivalent_density)
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

        leaf_surface = scale_surface(
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

            result = p2p(
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

                    result = p2p(
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

                result = p2p(
                    kernel_function=self.kernel_function,
                    targets=target,
                    sources=leaf_sources,
                    source_densities=leaf_source_densities
                )

                self.result_data[target_index].density += result.density

                # Updating indices
                self.result_data[target_index].indices.update(
                    self.source_data[leaf_node_key].indices)

