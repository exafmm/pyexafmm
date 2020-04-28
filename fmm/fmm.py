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


import scipy.special as sp
import numpy as np

def cartesian_to_spherical(cart):
    """
    (Non-Optimised) Conversion from 3D Cartesian to spherical polar coordinates.
    """
    sph = np.zeros_like(cart)
    for i in range(len(cart)):
        sph[i][0] = np.sqrt(cart[i][0]**2+cart[i][1]**2+cart[i][2]**2) # Radial coordinate
        sph[i][1] = np.arctan2(cart[i][1], cart[i][0]) # azimuthal angle
        sph[i][2] = np.arccos(cart[i][2]/sph[i][0]) # zenith angle
    return sph

def sph_harm(m, n, theta, phi):
    """
    Un-normalise the spherical harmonic, to correpond to common formulation in
        the literature.
    """
    return np.sqrt(4*np.pi/(2*n+1))*sp.sph_harm(m, n, theta, phi)

def multipole_coefficient(m, n, sources):
    """
    Calculate the coefficient of the multipole expansion.
    """
    res = 0

    # Number of sources
    k = len(sources)

    for i in range(k):
        qi = 1 # ith charge (setting as unit charge for now)
        rhoi = sources[i][0] # ith source radial coord
        alphai = sources[i][1] # ith azimuthal angle
        betai = sources[i][2] # ith zenith angle
        res += qi*(rhoi**n)*sph_harm(-m, n, alphai, betai)

    return res

def J(m, n):
    if m*n < 0:
        return (-1)**min(m, n)
    return 1

def A(m, n):
    return (-1)**n/np.sqrt(sp.factorial(n-m) * sp.factorial(n+m))

def shifted_multipole_coefficient(k, j, sources):

    res = 0
    rho = 1
    alpha = 1
    beta = 1

    for n in range(j):
        for m in range(-k, k+1):
            res += multipole_coefficient(k-m, j-n, sources)\
                *J(k-m, m)*A(m, n)*A(k-m, j-n)\
                *(rho**n)*sph_harm(-m, n, alpha, beta)

    return res

def multipole_expansion(sources, target, degree):
    """
    Compute multipole expansion of sources at target to specified degree
    """
    
    res = 0

    r = target[0] # target radial coord
    theta = target[1] # target azimuth coord
    phi = target[2] # target zenith coord

    for n in range(degree):
        for m in range(-n, n+1):
            coeff = multipole_coefficient(m, n, sources)/(r**n+1) 
            res += (multipole_coefficient(m, n, sources)/(r**n+1))\
                *sph_harm(m, n, theta, phi)

    return res

def particle_to_multipole():
    pass

def multipole_to_multipole(sources, target, degree):
    """
    Compute translation of multipole expansion, to new expansion centre
    """

    res = 0
    r = 1
    theta = 1
    phi = 1

    for j in range(degree):
        for k in range(-j, j+1):
            res += shifted_multipole_coefficient(k, j, sources)/r**(j+1)\
                *sph_harm(k, j, theta, phi)

    return res

def multipole_to_local():
    pass

def local_to_local():
    pass