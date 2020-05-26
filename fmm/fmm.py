"""Implementation of the main FMM loop."""
import numpy as np
import scipy.sparse.linalg

import fmm.hilbert as hilbert

class Potential:
    """
    Return object for computed potential, bundle equivalent density and its
        corresponding equivalent surface.
    """
    def __init__(self, equivalent_surface, equivalent_density):
        self.equivalent_surface = equivalent_surface
        self.equivalent_density = equivalent_density

class Node:
    """Holds expansion and source/target indices for each tree node"""
    def __init__(self, key, expansion, indices):
        """
        Parameters:
        ----------
        key : int
            Hilbert key for a node.
        expansion : np.array(shape=(ncoefficients))
            The expansion coefficients for this node's equivalent density.
        indices : set
            Set of indices.
        """
        self.key = key
        self.expansion = expansion
        self.indices = indices

    def __repr__(self):
        """Stringified repr of a node"""
        return str((self.key, self.expansion, self.indices))


class Fmm:
    """Main Fmm class."""

    def __init__(self, octree, order, kernel):
        """Initialize the Fmm."""

        self.kernel = kernel
        self.order = order
        self._octree = octree

        # One for each point on surface of a cube based discretisation
        self.ncoeffiecients = 6*(order-1)**2 + 2

        # Source and Target data indexed by Hilbert key
        self._source_data = {}
        self._target_data = {}

        self._result_data = [set() for _ in range(octree.targets.shape[0])]

        for key in self.octree.non_empty_source_nodes:
            self._source_data[key] = Node(
                    key, np.zeros(self.ncoeffiecients, dtype='float64'), set()
                    )
        for key in self.octree.non_empty_target_nodes:
            self._target_data[key] = Node(
                    key, np.zeros(self.ncoeffiecients, dtype='float64'), set()
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

        # Source indices in a given leaf
        source_indices = self.octree.sources_by_leafs[
                self.octree.source_index_ptr[leaf_node_index]
                :self.octree.source_index_ptr[leaf_node_index + 1]
                ]

        # 0.1 Find leaf sources
        leaf_sources = self.octree.sources[source_indices]

        # Just adding index from argsort (sources by leafs)
        self._source_data[
            self.octree.source_leaf_nodes[leaf_node_index]
            ].indices.update(source_indices)

        # 0.2 Compute key corresponding to this leaf index, and its parent
        child_key = self.octree.source_leaf_nodes[leaf_node_index]
        parent_key = hilbert.get_parent(child_key)

        # 0.3 Compute center of parent box in cartesian coordinates
        parent_center = hilbert.get_center_from_key(
            parent_key, self.octree.center, self.octree.radius)

        # 1. Compute expansion, and add to source data
        result = p2m(
            kernel_function=self.kernel,
            leaf_sources=leaf_sources,
            order=self.order,
            center=parent_center,
            radius=self.octree.radius,
            maximum_level=self.octree.maximum_lev
        )

        self._source_data[child_key].expansion = result.equivalent_density

    def multipole_to_multipole(self, key):
        """Combine children expansions of node into node expansion."""

        # Compute center of parent boxes
        parent_center = hilbert.get_center_from_key(
            key, self.octree.center, self.octree.radius)
        parent_level = hilbert.get_level(key)

        for child in hilbert.get_children(key):
            if self.octree.source_node_to_index[child] != -1:

                # Compute center of child box in cartesian coordinates
                child_center = hilbert.get_center_from_key(
                    child, self.octree.center, self.octree.radius
                    )
                child_level = hilbert.get_level(child)

                # Updating indices
                self._source_data[key].indices.update(
                    self._source_data[child].indices
                    )

                # Get child equivalent density
                child_equivalent_density = self._source_data[child].expansion

                # Compute expansion, and store
                self._source_data[key].expansion += m2m(
                    kernel=self.kernel,
                    parent_center=parent_center,
                    child_center=child_center,
                    child_level=child_level,
                    parent_level=parent_level,
                    radius=self.octree.radius,
                    order=self.order,
                    child_equivalent_density=child_equivalent_density
                    )

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


def surface(order, radius, level, center, alpha):
    """
    Compute vectors to correspond to quadrature points on surface of a specified
        cube.

    Parameters:
    -----------
    order : int
        Order of the expansion.
    radius : float
        Half side length of the octree's root node.
    level : int
        (Octree) level of cube.
    center : coordinate
        Coordinates of the centre of the cube.
    alpha : float
        Ratio between side length of surface cube and original cube.

    Returns:
    --------
    vector
        Vector of coordinates of surface points.
    """
    n_coeffs = 6*(order-1)**2 + 2
    surf = np.zeros(shape=(n_coeffs, 3))

    surf[0] = np.array([-1, -1, -1])
    count = 1

    # Hold x fixed
    for i in range(order-1):
        for j in range(order-1):
            surf[count][0] = -1
            surf[count][1] = (2*(i+1)-(order-1))/(order-1)
            surf[count][2] = (2*j-(order-1))/(order-1)
            count += 1

    # Hold y fixed
    for i in range(order-1):
        for j in range(order-1):
            surf[count][0] = (2*j-(order-1))/(order-1)
            surf[count][1] = -1
            surf[count][2] = (2*(i+1)-(order-1))/(order-1)
            count += 1

    # Hold z fixed
    for i in range(order-1):
        for j in range(order-1):
            surf[count][0] = (2*(i+1)-(order-1))/(order-1)
            surf[count][1] = (2*j-(order-1))/(order-1)
            surf[count][2] = -1
            count += 1

    # Reflect about origin, for remaining faces
    for i in range(n_coeffs//2):
        surf[count+i] = -surf[i]

    # Translate box to specified centre, and scale
    r = (0.5)**level * radius
    b = alpha*r

    for i in range(n_coeffs):
        surf[i] = surf[i]*b + center

    return surf


def laplace(x, y):
    """
    3D single-layer Laplace kernel between two points. Alternatively called
        particle to particle (P2P) operator.

    Parameters:
    -----------
    x : np.array(shape=(n))
        An n-dimensional vector corresponding to a point in n-dimensional space.
    y : np.array(shape=(n))
        Different n-dimensional vector corresponding to a point in n-dimensional
        space.

    Returns:
    --------
    float
        Operator value (scaled by 4pi) between points x and y.
    """
    r = np.linalg.norm(x-y)

    return 1/(4*np.pi*r)


def gram_matrix(kernel, sources, targets):
    """
    Compute Gram matrix of given kernel function. Elements are the pairwise
        interactions of sources/targets under the action of the kernel function.

    Parameters:
    -----------
    kernel : function
        Kernel function
    sources : np.array(shape=(n))
        The n source locations on a surface.
    targets : np.array(shape=(m))
        The m target locations on a surface.

    Returns:
    --------
    np.array(shape=(n, m))
        The Gram matrix.
    """

    matrix = np.zeros(shape=(len(sources), len(targets)))

    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            matrix[i][j] = kernel(source, target)

    return matrix


def pseudo_inverse(gram_matrix):
    """
    Compute the operator between the check and the equivalent surface.
    """

    return np.linalg.pinv(gram_matrix)


def potential_p2p(kernel_function, targets, sources, source_densities):
    """
    Directly calculate potential at m targets from n sources.

    Parameters:
    -----------
    kernel_function : function
    targets : np.array(shape=(m, 3))
    sources : np.array(shape=(n, 3))
    source_densities : np.array(shape=(n))

    Returns:
    --------
    np.array(shape=(m, 1))
        Potential from all sources at each target point.
    """

    # Potential at target locations
    target_densities = np.zeros(shape=(len(targets), 1))

    for i, target in enumerate(targets):
        potential = 0
        for j, source in enumerate(sources):
            source_density = source_densities[j]
            potential += kernel_function(target, source)*source_density
        target_densities[i] = potential

    return target_densities


def p2m(kernel_function,
        order,
        center,
        radius,
        maximum_level,
        leaf_sources):
    """
    Compute multipole expansion from sources at the leaf level supported at
        discrete points on the upward equivalent surface.

    Parameters:
    -----------
    kernel_function : function
    order : int
    center : np.array(shape=(3))
        The center of expansion.
    radius : float
        Half-side length of root node.
    maximum_level : int
        The maximium level of the octree.
    leaf_sources : np.array(shape=(n, 3))
        Sources in a given leaf node, at which multipole expansion is being
        computed.

    Returns:
    --------
    Potential
        Potential object, containing equivalent surface and equivalent density.
    """

    # 0.1 Compute relevant surfaces
    upward_check_surface = surface(
        order=order,
        radius=radius,
        level=maximum_level,
        center=center,
        alpha=2.95
        )

    upward_equivalent_surface = surface(
        order=order,
        radius=radius,
        level=maximum_level,
        center=center,
        alpha=1.05
    )

    # 0.2 Compute Gram Matrix
    kernel_matrix = gram_matrix(
        kernel_function, upward_equivalent_surface, upward_check_surface)

    # 0.3 Set unit densities at leaves for now
    leaf_source_densities = np.ones(shape=(len(leaf_sources)))

    # 1.0 Compute check potential directly using leaves
    check_potential = potential_p2p(
            kernel_function=kernel_function,
            targets=upward_check_surface,
            sources=leaf_sources,
            source_densities=leaf_source_densities
            )

    # 2.0 Compute backward-stable pseudo-inverse of kernel matrix
    # 2.1 SVD decomposition of kernel matrix
    u, s, v_transpose = np.linalg.svd(kernel_matrix)

    # 2.2 Invert S
    tol = 1e-1
    for i, val in enumerate(s):
        if  abs(val) < tol:
            s[i] = 0
        else:
            s[i] = 1/val

    tmp = np.matmul(v_transpose.T, np.diag(s))
    kernel_matrix_inv = np.matmul(tmp, u.T)

    # 3.0 Compute upward equivalent density
    upward_equivalent_density = np.matmul(kernel_matrix_inv, check_potential)

    return Potential(upward_equivalent_surface, upward_equivalent_density)


def m2m(kernel,
        order,
        parent_center,
        child_center,
        radius,
        child_level,
        parent_level,
        child_equivalent_density):
    """
    Compute multipole expansion at parent level, from child level.

    Parameters:
    -----------
    kernel : function
    order : int
    parent_center : np.array(shape=(3))
    child_center : np.array(shape=(3))
    radius : float
        Half-side length of root node.
    child_level : int
    parent_level : int
    child_equivalent_density : np.array(shape=(n))
        The equivalent densities calculated in the previous step at the
        `n` quadrature points at the child level.

    Returns:
    --------
    np.array(shape=(m))
        The equivalent densities calculated at the `m` quadrature points of the
        parent level.
    """
    # 0. Calculate surfaces
    child_equivalent_surface = surface(
        order=order,
        radius=radius,
        level=child_level,
        center=child_center,
        alpha=1.05
    )

    parent_equivalent_surface = surface(
        order=order,
        radius=radius,
        level=parent_level,
        center=parent_center,
        alpha=1.95
    )

    parent_check_surface = surface(
        order=order,
        radius=radius,
        level=parent_level,
        center=parent_center,
        alpha=2.95
    )

    # 1. Calculate check potential from child equivelent density
    check_potential = np.zeros(shape=(len(parent_check_surface)))

    for i, target in enumerate(parent_check_surface):
        potential = 0
        for j, source in enumerate(child_equivalent_surface):
            source_density = child_equivalent_density[j]
            potential += kernel(target, source)*source_density
        check_potential[i] = potential

    # 2. Calculate equivalent density on parent equivalent surface
    # 2.1 Form gram matrix between parent check surface and equivalent surface
    kernel_matrix = gram_matrix(
        kernel, parent_equivalent_surface, parent_check_surface)

    # 2.2 Invert gram matrix, and find equivalent density
    u, s, v_transpose = np.linalg.svd(kernel_matrix)

    # 2.3 Invert S
    tol = 1e-1
    for i, val in enumerate(s):
        if  abs(val) < tol:
            s[i] = 0
        else:
            s[i] = 1/val

    tmp = np.matmul(v_transpose.T, np.diag(s))
    kernel_matrix_inv = np.matmul(tmp, u.T)

    parent_equivalent_density = np.matmul(kernel_matrix_inv, check_potential)

    return Potential(parent_equivalent_surface, parent_equivalent_density)


def m2l():
    pass


def l2l():
    pass
