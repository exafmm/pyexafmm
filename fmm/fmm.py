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

def spherical_to_cartesian(sph):
    """
    Conversion from 3D Spherical Polars to Cartesian
    """
    cart = np.zeros_like(sph)
    for i in range(len(cart)):
        r = sph[i][0]; theta = sph[i][1]; phi = sph[i][2]
        cart[i][0] = r*np.sin(theta)*np.cos(phi)
        cart[i][1] = r*np.sin(theta)*np.sin(phi)
        cart[i][2] = r*np.cos(theta)

    return cart

def Y(m, n, theta, phi):
    """
    Un-normalise the spherical harmonic, to correpond to common formulation in
        the literature.
    """
    return np.sqrt(4*np.pi/(2*n+1))*sp.sph_harm(m, n, theta, phi)

def O(m, n, sources):
    """Multipole coefficient"""
    res = 0
    
    k = len(sources)
    for i in range(k):
        source = sources[i]
        q = 1 # unit charge for now
        rho = source[0]; alpha = source[1]; beta = source[2]
        res += q*(rho**n)*Y(-m, n, alpha, beta)

    return res

def J(m, n):
    if m*n < 0:
        return (-1)**min(m, n)
    return 1

def A(m, n):
    return (-1)**n/np.sqrt(sp.factorial(n-m) * sp.factorial(n+m))

def p2m(sources, target, degree):
    """
    Compute potential at target, from sources, using multipole expansion.
    """

    res = 0

    r = target[0] # target radial coord
    theta = target[1] # target azimuth coord
    phi = target[2] # target zenith coord

    for n in range(degree):
        for m in range(-n, n+1):
            res += (O(m, n, sources)/(r**(n+1)))\
                *Y(m, n, theta, phi)

    return res

def M(k, j, sources):
    """Coefficients of shifted Multipole expansion"""
    res = 0

    # Find centre of expansion, convert to carteisan and back
    converted = spherical_to_cartesian(sources)
    centre_cart = np.mean(converted, axis=0)
    centre_sph = cartesian_to_spherical(centre_cart)

    rho = centre_sph[0]; alpha = centre_sph[1]; beta = centre_sph[2]

    for n in range(j):
        for m in range(-n, n+1):
            res += O(k-m, j-n, sources)\
                *J(k-m, m)*A(m, n)*A(k-m, j-n)\
                *(rho**n)*Y(-m, n, alpha, beta)

    return res

def m2m(sources, target, degree):
    """
    Compute translation of multipole expansion, to new expansion centre
    """

    res = 0

    r = target[0]; theta = target[1]; phi = target[2]

    for j in range(degree):
        for k in range(-j, j+1):
            res += (M(k, j, sources)/(r**(j+1)))\
                *Y(k, j, theta, phi)

    return res

def L(k, j, sources, order):
    """Coefficients of Local expansion"""

    res = 0
    
    # Find centre of expansion, convert to carteisan and back
    converted = spherical_to_cartesian(sources)
    centre_cart = np.mean(converted, axis=0)
    centre_sph = cartesian_to_spherical(centre_cart)

    rho = centre_sph[0]; alpha = centre_sph[1]; beta = centre_sph[2]

    for n in range(order+1):
        for m in range(-n, n+1):
            res += O(m, n, sources)*J(m, k)*A(m, n)*A(k, j)*Y(m-k, j+n, alpha, beta)\
                /(A(m-k, j+n)*rho**(j+n+1))

    return res

def m2l(target, sources, order):
    """convert a multipole to a local expansion"""

    res = 0

    r = target[0]; theta = target[1]; phi = target[2]

    for j in range(order+1):
        for k in range(-j, j+1):
            res += L(k, j, sources, order)*Y(k, j, theta, phi)*(r**j)

    return res

def l2l():
    """Translate a local expansion"""
    pass

def direct_calculation(sources, target):
    """For checking"""
    res = 0

    for i in range(len(sources)):
        source = sources[i]
        dist = source - target # cartesian
        potential = 1/np.linalg.norm(dist)
        res += potential

    return res