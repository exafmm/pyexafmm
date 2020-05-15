"""Implementation of the main FMM loop."""

import numpy as np
import fmm.hilbert as hilbert
import collections as collections

NodeData = collections.namedtuple("NodeData", "key expansion indices")

class Fmm(object):
    """Main Fmm class."""

    def __init__(self, octree, order):
        """Initialize the Fmm."""

        self._octree = octree

        self._source_data = {}
        self._target_data = {}
        self._result_data = [set() for _ in range(octree.targets.shape[0])]

        for key in self.octree.non_empty_source_nodes:
            self._source_data[key] = NodeData(
                    key, np.zeros(order, dtype='float64'), set()
                    )
        for key in self.octree.non_empty_target_nodes:
            self._target_data[key] = NodeData(
                    key, np.zeros(order, dtype='float64'), set()
                    )

    @property
    def octree(self):
        """Return octree."""
        return self._octree

    def upward_pass(self):
        """Upward pass."""
        number_of_leafs = len(self.octree.source_index_ptr) - 1
        for index in range(number_of_leafs):
            self.particle_to_multipole(index)

        for level in range(self.octree.maximum_level - 1, -1, -1):
            for key in self.octree.non_empty_source_nodes_by_level[level]:
                self.multipole_to_multipole(key)

    def downward_pass(self):
        """Downward pass."""
        for level in range(2, 1 + self.octree.maximum_level):
            for key in self.octree.non_empty_target_nodes_by_level[level]:
                index = self.octree.target_node_to_index[key]
                for neighbor_list in self.octree.interaction_list[index]:
                    for child in neighbor_list:
                        if child != -1:
                            self.multipole_to_local(child, key)
                if level < self.octree.maximum_level:
                        self.local_to_local(key)

        for leaf_node_index in range(len(self.octree.target_index_ptr) - 1):
            self.local_to_particle(leaf_node_index)
            self.compute_near_field(leaf_node_index)


    def set_source_values(self, values):
        """Set source values."""
        pass

    def get_target_values(self):
        """Get target values."""
        pass


    def particle_to_multipole(self, leaf_node_index):
        """Compute particle to multipole interactions in leaf."""
        source_indices = self.octree.sources_by_leafs[
                self.octree.source_index_ptr[leaf_node_index] : self.octree.source_index_ptr[leaf_node_index + 1]
                ]
        self._source_data[self.octree.source_leaf_nodes[leaf_node_index]].indices.update(source_indices)

                
    def multipole_to_multipole(self, node):
        """Combine children expansions of node into node expansion."""
        for child in hilbert.get_children(node):
            if self.octree.source_node_to_index[child] != -1:
                self._source_data[node].indices.update(self._source_data[child].indices)


    def multipole_to_local(self, source_node, target_node):
        """Compute multipole to local."""
        self._target_data[target_node].indices.update(
                self._source_data[source_node].indices
                )

    def local_to_local(self, node):
        """Compute local to local."""
        for child in hilbert.get_children(node):
            if self.octree.target_node_to_index[child] != -1:
                self._target_data[child].indices.update(
                        self._target_data[node].indices
                        )
        

    def local_to_particle(self, leaf_node_index):
        """Compute local to particle."""
        target_indices = self.octree.targets_by_leafs[
                self.octree.target_index_ptr[leaf_node_index] : self.octree.target_index_ptr[leaf_node_index + 1]
                ]
        leaf_node_key = self.octree.target_leaf_nodes[leaf_node_index]
        for target_index in target_indices:
            self._result_data[target_index].update(
                    self._target_data[leaf_node_key].indices
                    )

    def compute_near_field(self, leaf_node_index):
        """Compute near field."""
        target_indices = self.octree.targets_by_leafs[
                self.octree.target_index_ptr[leaf_node_index] : self.octree.target_index_ptr[leaf_node_index + 1]
                ]
        leaf_node_key = self.octree.target_leaf_nodes[leaf_node_index]
        for neighbor in self.octree.target_neighbors[
                self.octree.target_node_to_index[leaf_node_key]
                ]:
            if neighbor != -1:
                for target_index in target_indices:
                    self._result_data[target_index].update(
                            self._source_data[neighbor].indices
                            )
        if self.octree.source_node_to_index[leaf_node_key] != -1:
            for target_index in target_indices:
                self._result_data[target_index].update(self._source_data[leaf_node_key].indices)

# Functions to try out, before integrating with class

import numpy as np
from fmm.octree import Octree


def surface(p, r, level, c, alpha):
    """
    Compute vectors to correspond to a surface of a box.

    Parameters:
    -----------
    p : int
        Order of the expansion.
    r : float
        Half side length of the bounding box
    level : int
        Level of box
    c : coordinate
        Coordinates of the centre of a box.
    alpha : float
        Ratio between side length of surface box and original box.

    Returns:
    --------
    vector
        Vector of coordinates of surface points.
    """
    n = 6*(p-1)**2 + 2
    res = np.zeros(shape=(n, 3))

    res[0] = np.array([-1, -1, -1])
    count = 1

    # Hold x fixed
    for i in range(p-1):
        for j in range(p-1):
            res[count][0] = -1
            res[count][1] = (2*(i+1)-(p-1))/(p-1)
            res[count][2] = (2*j-(p-1))/(p-1)
            count += 1

    # Hold y fixed
    for i in range(p-1):
        for j in range(p-1):
            res[count][0] = (2*j-(p-1))/(p-1)
            res[count][1] = -1
            res[count][2] = (2*(i+1)-(p-1))/(p-1)
            count += 1

    # Hold z fixed
    for i in range(p-1):
        for j in range(p-1):
            res[count][0] = (2*(i+1)-(p-1))/(p-1)
            res[count][1] = (2*j-(p-1))/(p-1)
            res[count][2] = -1
            count += 1

    # Reflect about origin, for remaining faces
    for i in range(n//2):
        res[count+i] = -res[i]

    # Translate box to specified centre, and scale
    r *= (0.5)**level
    b = alpha*r

    for i in range(n):
        res[i] = res[i]*b + c

    return res


def laplace(x, y):
    """
    3D single-layer Laplace kernel between two points. Alternatively called
        particle to particle (P2P) operator.
    """
    r = np.linalg.norm(x-y)

    return 1/(4*np.pi*r)


def gram_matrix(kernel, sources, targets):
    """
    Compute Gram matrix of given kernel function. Elements are the pairwise
        interactions of sources/targets under the action of the kernel function.

    Parameters:
    -----------
    kernel : kernelf
        Kernel function
    sources : vector
        The source locations on a surface.
    targets : vector
        The target locations on a surface.

    Returns:
    --------
    matrix
        The Gram matrix.
    """

    matrix = np.zeros(shape=(len(sources), len(targets)))

    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            matrix[i][j] = kernel(source, target)

    return matrix


def check_to_equivalent(gram_matrix):
    """ Compute the operator between the check and the equivalent surface.
    """

    # Based on Tingyu's knowledge from literature, equivalent to least squares
    # formulation, remember to get a reference

    return np.linalg.pinv(gram_matrix)


def potential_p2p(kernel, targets, sources):
    """Directly calculate potential at targets from sources
    """
    
    # Potential at target locations
    target_densities = np.zeros(shape=(len(targets), 1))

    for i, target in enumerate(targets):
        potential = 0
        for source in sources:
            # for now assume source densities are all 1
            source_density = 1
            potential += kernel(target, source)*source_density
        target_densities[i] = potential

    return target_densities


def m2m():
    """ Compute multipole to multipole (M2M) operator
    """
    pass


def test():
    p = 2
    r = 2
    c = np.array([0, 0, 0])
    level = 1
    
    upward_equivalent_surface = surface(p, r, level, c, 1.95)
    
    # Some sources placed inside a leaf node for testing.
    leaf_sources = np.array([
        [0,0,0],
        [1,0,0],
        [0,0,1]
    ])
    check_potential = potential_p2p(laplace, upward_equivalent_surface, leaf_sources)

    return check_potential
