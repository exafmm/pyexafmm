"""Implementation of the main FMM loop."""
import os
import pathlib

import numpy as np

import fmm.hilbert as hilbert
from fmm.density import Potential
from fmm.node import Node
from fmm.operator import p2p, p2m
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

        data_dirpath = PARENT / "data"

        self.config = load_json(config_filepath)

        self.operator_dirpath = PARENT/ self.config["operator_dirname"]
        source_filename = self.config['source_filename']
        target_filename = self.config['target_filename']
        source_densities_filename = self.config['source_densities_filename']

        sources = load_hdf5_to_array('sources', source_filename, data_dirpath)
        targets = load_hdf5_to_array('targets', target_filename, data_dirpath)

        source_densities = load_hdf5_to_array(
            'source_densities', source_densities_filename, data_dirpath)

        # Extract config properties
        self.kernel_function = KERNELS[self.config['kernel']]()
        self.order = self.config['order']
        self.maximum_level = self.config['octree_max_level']
        self.octree = Octree(sources, targets, self.maximum_level)

        # Coefficients discretising surface of a node
        self.ncoefficients = 6*(self.order-1)**2 + 2

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
        for level in range(self.octree.maximum_level - 1, -1, -1):
            for key in self.octree.non_empty_source_nodes_by_level[level]:
                self.multipole_to_multipole(key)

    # def downward_pass(self):
    #     """Downward pass loop."""

    #     # Pre-order traversal of octree
    #     for level in range(1, 1 + self.octree.maximum_level):

    #         for key in self.octree.non_empty_target_nodes_by_level[level]:
    #             index = self.octree.target_node_to_index[key]

    #             # Translating mutlipole expansion in far field to a local
    #             # expansion at currently examined node.
    #             for neighbor_list in self.octree.interaction_list[index]:
    #                 for child in neighbor_list:
    #                     if child != -1:
    #                         self.multipole_to_local(child, key)

    #             # Translate local expansion to the node's children
    #             if level < self.octree.maximum_level:
    #                 self.local_to_local(key)

        # Treat local expansion as charge density, and evaluate at each particle
        # at leaf node, compute near field.
        # for leaf_node_index in range(len(self.octree.target_index_ptr) - 1):
        #     self.local_to_particle(leaf_node_index)
        #     self.compute_near_field(leaf_node_index)

    def particle_to_multipole(self, leaf_node_index):
        """Compute particle to multipole interactions in leaf."""

        # Source indices in a given leaf
        source_indices = self.octree.sources_by_leafs[
            self.octree.source_index_ptr[leaf_node_index]
            : self.octree.source_index_ptr[leaf_node_index + 1]
        ]

        # Find leaf sources
        leaf_sources = self.octree.sources[source_indices]

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

        # Compute expansion, and add to source data
        result = p2m(
            operator_dirpath=self.operator_dirpath,
            kernel_function=self.kernel_function,
            leaf_sources=leaf_sources,
            leaf_source_densities=np.ones(len(leaf_sources)),
            center=leaf_center,
            radius=self.octree.radius,
            level=self.octree.maximum_level
        )

        self.source_data[leaf_key].expansion = result.density

    def multipole_to_multipole(self, key):
        """Combine children expansions of node into node expansion."""

        # Compute center of parent boxes

        parent_center = hilbert.get_center_from_key(
            key, self.octree.center, self.octree.radius
        )
        parent_level = hilbert.get_level(key)

        for child in hilbert.get_children(key):
            # Only going through non-empty child nodes
            if self.octree.source_node_to_index[child] != -1:

                # Compute center of child box in cartesian coordinates
                child_center = hilbert.get_center_from_key(
                    child, self.octree.center, self.octree.radius
                    )

                child_level = hilbert.get_level(child)

                # Updating indices
                self.source_data[key].indices.update(
                    self.source_data[child].indices
                    )

                # Get child equivalent density
                child_equivalent_density = self.source_data[child].expansion

                # Compute expansion, and store
                # result = m2m(
                #     kernel_function=self.kernel_function,
                #     parent_center=parent_center,
                #     child_center=child_center,
                #     child_level=child_level,
                #     parent_level=parent_level,
                #     radius=self.octree.radius,
                #     order=self.order,
                #     child_equivalent_density=child_equivalent_density
                # )

                # self.source_data[key].expansion += result.density

    def multipole_to_local(self, source_key, target_key):
        """
        Translate multipole expansions in target's interaction list to local
        expansions about the target.
        """

        x0 = self.octree.center
        r0 = self.octree.radius

        # Updating indices
        self.target_data[target_key].indices.update(
            self.source_data[source_key].indices
        )

        source_level = hilbert.get_level(source_key)
        source_center = hilbert.get_center_from_key(source_key, x0, r0)

        target_level = hilbert.get_level(target_key)
        target_center = hilbert.get_center_from_key(target_key, x0, r0)

        source_equivalent_density = self.source_data[source_key].expansion

        # result = m2l(
        #     kernel_function=self.kernel_function,
        #     order=self.order,
        #     radius=self.octree.radius,
        #     source_center=source_center,
        #     source_level=source_level,
        #     target_center=target_center,
        #     target_level=target_level,
        #     source_equivalent_density=source_equivalent_density
        # )

        # self.target_data[target_key].expansion += result.density

    def local_to_local(self, key):
        """Translate local expansion of a node to it's children."""

        x0 = self.octree.center
        r0 = self.octree.radius

        parent_center = hilbert.get_center_from_key(key, x0, r0)
        parent_level = hilbert.get_level(key)
        parent_equivalent_density = self.target_data[key].expansion

        for child in hilbert.get_children(key):
            if self.octree.target_node_to_index[child] != -1:

                child_center = hilbert.get_center_from_key(child, x0, r0)
                child_level = hilbert.get_level(child)

                # Updating indices
                self.target_data[child].indices.update(
                    self.target_data[key].indices
                )

                # result = l2l(
                #     kernel_function=self.kernel_function,
                #     order=self.order,
                #     radius=r0,
                #     parent_center=parent_center,
                #     parent_level=parent_level,
                #     child_center=child_center,
                #     child_level=child_level,
                #     parent_equivalent_density=parent_equivalent_density
                # )

                # self.target_data[child].expansion = result.density

    # def local_to_particle(self, leaf_node_index):
    #     """
    #     Directly evaluate potential at particles in a leaf node, treating the
    #     local expansion points as sources.
    #     """
    #     x0 = self.octree.center
    #     r0 = self.octree.radius

    #     target_indices = self.octree.targets_by_leafs[
    #         self.octree.target_index_ptr[leaf_node_index]
    #         : self.octree.target_index_ptr[leaf_node_index + 1]
    #     ]

    #     leaf_node_key = self.octree.target_leaf_nodes[leaf_node_index]
    #     leaf_node_level = hilbert.get_level(leaf_node_key)
    #     leaf_node_center = hilbert.get_center_from_key(leaf_node_key, x0, r0)

    #     leaf_node_density = self.target_data[leaf_node_key].expansion

    #     leaf_node_surface = surface(
    #         order=self.order,
    #         radius=self.octree.radius,
    #         level=leaf_node_level,
    #         center=leaf_node_center,
    #         alpha=2.95
    #     )

    #     for target_index in target_indices:

    #         # Updating indices
    #         self.result_data[target_index].indices.update(
    #             self.target_data[leaf_node_key].indices
    #         )

    #         target = self.octree.targets[target_index].reshape(1, 3)

    #         result = p2p(
    #             kernel_function=self.kernel_function,
    #             targets=target,
    #             sources=leaf_node_surface,
    #             source_densities=leaf_node_density
    #         )

    #         self.result_data[target_index].density = result.density

    # def compute_near_field(self, leaf_node_index):
    #     """
    #     Compute near field influence from neighbouring box's local expansions.
    #     """

    #     target_indices = self.octree.targets_by_leafs[
    #         self.octree.target_index_ptr[leaf_node_index]
    #         : self.octree.target_index_ptr[leaf_node_index + 1]
    #     ]

    #     leaf_node_key = self.octree.target_leaf_nodes[leaf_node_index]

    #     for neighbor_key in self.octree.target_neighbors[
    #             self.octree.target_node_to_index[leaf_node_key]
    #             ]:
    #         if neighbor_key != -1:

    #             neighbor_index = self.octree.source_node_to_index[neighbor_key]

    #             neighbor_source_indices = self.octree.sources_by_leafs[
    #                 self.octree.source_index_ptr[neighbor_index]:
    #                 self.octree.source_index_ptr[neighbor_index + 1]
    #             ]

    #             neighbor_sources = self.octree.sources[neighbor_source_indices]

    #             for target_index in target_indices:

    #                 # Updating indices
    #                 self.result_data[target_index].indices.update(
    #                     self.source_data[neighbor_key].indices
    #                 )

    #                 target = self.octree.targets[target_index].reshape(1, 3)

    #                 result = p2p(
    #                     kernel_function=self.kernel_function,
    #                     targets=target,
    #                     sources=neighbor_sources,
    #                     source_densities=np.ones(len(neighbor_sources))
    #                 )

    #                 self.result_data[target_index].density += result.density

    #     if self.octree.source_node_to_index[leaf_node_key] != -1:

    #         leaf_index = self.octree.source_node_to_index[leaf_node_key]

    #         leaf_source_indices = self.octree.sources_by_leafs[
    #             self.octree.source_index_ptr[leaf_index]:
    #             self.octree.source_index_ptr[leaf_index + 1]
    #         ]

    #         leaf_sources = self.octree.sources[leaf_source_indices]

    #         for target_index in target_indices:
    #             target = self.octree.targets[target_index].reshape(1, 3)

    #             result = p2p(
    #                 kernel_function=self.kernel_function,
    #                 targets=target,
    #                 sources=leaf_sources,
    #                 source_densities=np.ones(len(leaf_sources))
    #             )

    #             self.result_data[target_index].density += result.density

    #             # Updating indices
    #             self.result_data[target_index].indices.update(
    #                 self.source_data[leaf_node_key].indices)


