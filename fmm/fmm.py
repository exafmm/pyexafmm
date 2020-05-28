"""Implementation of the main FMM loop."""
import numpy as np
import scipy.sparse.linalg

import fmm.hilbert as hilbert

class Potential:
    """
    Return object for computed potential, bundle equivalent density and its
        corresponding equivalent surface.
    """
    def __init__(self, surface, density):
        """
        Parameters:
        -----------
        surface : np.array(shape=(n, 3))
            `n` quadrature points discretising equivalent surface.
        density : np.array(shape=(n))
            `n` potential densities, corresponding to each quadrature point.
        """
        self.surface = surface
        self.density = density

    def __repr__(self):
        return str((self.surface, self.density))


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
            maximum_level=self.octree.maximum_level
        )

        self._source_data[child_key].expansion = result.density

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
                result = m2m(
                    kernel_function=self.kernel,
                    parent_center=parent_center,
                    child_center=child_center,
                    child_level=child_level,
                    parent_level=parent_level,
                    radius=self.octree.radius,
                    order=self.order,
                    child_equivalent_density=child_equivalent_density
                    )

                self._source_data[key].expansion += result.density

    def multipole_to_local(self, source_key, target_key):
        """Compute multipole to local."""

        # Compute centers, and levels
        x0 = self.octree.center; r0 = self.octree.radius

        source_center = hilbert.get_center_from_key(source_key, x0, r0)
        source_level = hilbert.get_level(source_key)
        target_center = hilbert.get_center_from_key(target_key, x0, r0)
        target_level = hilbert.get_level(target_key)

        # Get source equivalent density
        source_equivalent_density = self._source_data[source_key].expansion

        result = m2l(
            kernel_function=self.kernel,
            order=self.order,
            radius=self.octree.radius,
            source_center=source_center,
            source_level=source_level,
            target_center=target_center,
            target_level=target_level,
            source_equivalent_density=source_equivalent_density
        )

        # Add to target expansions
        self._target_data[target_key].expansion += result.density

        self._target_data[target_key].indices.update(
                self._source_data[source_key].indices
                )

    def local_to_local(self, key):
        """Compute local to local."""

        # Compute center of parent boxes
        parent_center = hilbert.get_center_from_key(
            key, self.octree.center, self.octree.radius)
        parent_level = hilbert.get_level(key)

        for child in hilbert.get_children(key):
            if self.octree.target_node_to_index[child] != -1:

                # Updating indices
                self._target_data[child].indices.update(
                        self._target_data[key].indices
                        )

                # Compute center of child box in cartesian coordinates
                child_center = hilbert.get_center_from_key(
                    child, self.octree.center, self.octree.radius
                    )
                child_level = hilbert.get_level(child)

                # Get parent equivalent density
                parent_equivalent_density = self._target_data[key].expansion

                # Compute expansion, and store
                result = l2l(
                    kernel_function=self.kernel,
                    order=self.order,
                    radius=self.octree.radius,
                    parent_center=parent_center,
                    parent_level=parent_level,
                    child_center=child_center,
                    child_level=child_level,
                    parent_equivalent_density=parent_equivalent_density
                )

                self._target_data[key].expansion += result.density

    def local_to_particle(self, leaf_node_index):
        """Compute local to particle."""
        target_indices = self.octree.targets_by_leafs[
                self.octree.target_index_ptr[leaf_node_index]
                    :self.octree.target_index_ptr[leaf_node_index + 1]
                ]
        leaf_node_key = self.octree.target_leaf_nodes[leaf_node_index]
        for target_index in target_indices:
            self._result_data[target_index].update(
                    self._target_data[leaf_node_key].indices
                    )

    def compute_near_field(self, leaf_node_index):
        """Compute near field."""
        target_indices = self.octree.targets_by_leafs[
                self.octree.target_index_ptr[leaf_node_index]
                    :self.octree.target_index_ptr[leaf_node_index + 1]
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
                self._result_data[target_index].update(
                    self._source_data[leaf_node_key].indices)


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
    np.array(shape=(n_coeffs, 3))
        Vector of coordinates of surface points. `n_coeffs` is the number of
        points that discretise the surface of a cube.
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

    if np.isclose(r, 0, rtol=1e-1):
        return 1e10
    return 1/(4*np.pi*r)


def gram_matrix(kernel, sources, targets):
    """
    Compute Gram matrix of given kernel function. Elements are the pairwise
        interactions of sources/targets under the action of the kernel function.

    Parameters:
    -----------
    kernel : function
        Kernel function
    sources : np.array(shape=(n, 3))
        The n source locations on a surface.
    targets : np.array(shape=(m, 3))
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


def pseudo_inverse(matrix):
    """
    Compute a backward stable pseudo-inverse of a (n x m) matrix using an SVD
        decomposition. For inverting the  singular diagonal S matrix, if a value
        is less than a specified tolerance we set it to 0, otherwise we use the
        value 1/e where e is a diagonal element of S.

    Parameters:
    -----------
    matrix : np.array(shape=(n, m))

    Returns:
    --------
    np.array(shape=(m, n))
    """
    u, s, v_transpose = np.linalg.svd(matrix)

    tol = 1e-1
    for i, val in enumerate(s):
        if  abs(val) < tol:
            s[i] = 0
        else:
            s[i] = 1/val

    tmp = np.matmul(v_transpose.T, np.diag(s))

    return np.matmul(tmp, u.T)


def p2p(kernel_function, targets, sources, source_densities):
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
    Potential
        Potential denities at all target points from from all sources.
    """

    # Potential at target locations
    target_densities = np.zeros(shape=(len(targets), 1))

    for i, target in enumerate(targets):
        potential = 0
        for j, source in enumerate(sources):
            source_density = source_densities[j]
            potential += kernel_function(target, source)*source_density
        target_densities[i] = potential

    return Potential(targets, target_densities)


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
        Potential densities calculated at the discrete points on the equivalent
        surface.
    """

    # Compute relevant surfaces
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

    # Compute Gram Matrix
    kernel_matrix = gram_matrix(
        kernel_function, upward_equivalent_surface, upward_check_surface)

    # Set unit densities at leaves for now
    leaf_source_densities = np.ones(shape=(len(leaf_sources)))

    # Compute check potential directly using leaves
    check_potential = p2p(
            kernel_function=kernel_function,
            targets=upward_check_surface,
            sources=leaf_sources,
            source_densities=leaf_source_densities
            ).density

    # Compute backward-stable pseudo-inverse of kernel matrix
    kernel_matrix_inv = pseudo_inverse(kernel_matrix)

    # Compute upward equivalent density
    upward_equivalent_density = np.matmul(kernel_matrix_inv, check_potential)

    return Potential(upward_equivalent_surface, upward_equivalent_density)


def m2m(kernel_function,
        order,
        radius,
        parent_center,
        child_center,
        parent_level,
        child_level,
        child_equivalent_density):
    """
    Translate a multipole expansion at parent level, from child level.

    Parameters:
    -----------
    kernel_function : function
    order : int
    radius : float
        Half-side length of root node.
    parent_center : np.array(shape=(3))
    child_center : np.array(shape=(3))
    parent_level : int
    child_level : int
    child_equivalent_density : np.array(shape=(n))
        The equivalent densities calculated in the previous step at the `n`
        quadrature points at the child level.

    Returns:
    --------
    Potential
        Potential densities calculated at the `m` quadrature points of the
        parent level.
    """
    # Calculate surfaces
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
        alpha=1.05
    )

    parent_check_surface = surface(
        order=order,
        radius=radius,
        level=parent_level,
        center=parent_center,
        alpha=2.95
    )

    # Calculate check potential from child equivelent density
    check_potential = p2p(
        kernel_function=kernel_function,
        targets=parent_check_surface,
        sources=child_equivalent_surface,
        source_densities=child_equivalent_density).density

    # Calculate equivalent density on parent equivalent surface
    # Form gram matrix between parent check surface and equivalent surface
    kernel_matrix = gram_matrix(
        kernel_function, parent_equivalent_surface, parent_check_surface)

    # Compute backward-stable pseudo-inverse of kernel matrix
    kernel_matrix_inv = pseudo_inverse(kernel_matrix)

    parent_equivalent_density = np.matmul(kernel_matrix_inv, check_potential)

    return Potential(parent_equivalent_surface, parent_equivalent_density)


def m2l(kernel_function,
        order,
        radius,
        source_center,
        source_level,
        target_center,
        target_level,
        source_equivalent_density):
    """
    Translate a local expansion, from a multipole expansion in a box's
        interaction list to a local expansion for the box.

    Parameters:
    -----------
    kernel_function : function
    order : int
    radius : float
        Half-side length of root node.
    source_center : np.array(shape=(3))
    source_level : int
    target_center: np.array(shape=(3))
    target_level : int
    source_equivalent_density = np.array(shape=(n))
        The equivalent densities calculated for the source box during the
        upward pass.

    Returns:
    --------
    Potential
        Potential densities for the local expansion around the target box.
    """

    # Compute surfaces
    src_upward_equivalent_surface = surface(
        order=order,
        radius=radius,
        level=source_level,
        center=source_center,
        alpha=1.05
    )

    tgt_downward_equivalent_surface = surface(
        order=order,
        radius=radius,
        level=target_level,
        center=target_center,
        alpha=2.95
    )

    tgt_check_surface = surface(
        order=order,
        radius=radius,
        level=target_level,
        center=target_center,
        alpha=1.05
    )

    # Calculate check potential from source equivalent density
    check_potential = np.zeros(shape=(len(tgt_check_surface)))

    check_potential = p2p(
        kernel_function=kernel_function,
        targets=tgt_check_surface,
        sources=src_upward_equivalent_surface,
        source_densities=source_equivalent_density).density

    # Calculate downward equivalent density
    # Form gram-matrix between target and source box
    kernel_matrix = gram_matrix(kernel_function,
                                src_upward_equivalent_surface,
                                tgt_downward_equivalent_surface)

    # Invert gram matrix with SVD
    kernel_matrix_inv = pseudo_inverse(kernel_matrix)

    tgt_equivalent_density = np.matmul(kernel_matrix_inv, check_potential)

    return Potential(tgt_downward_equivalent_surface, tgt_equivalent_density)


def l2l(kernel_function,
        order,
        radius,
        parent_center,
        child_center,
        parent_level,
        child_level,
        parent_equivalent_density):
    """
    Translate a local expansion at parent level, to the child level.

    Parameters:
    -----------
    kernel_function : function
    order : int
    radius : float
        Half-side length of root node.
    parent_center : np.array(shape=(3))
    child_center : np.array(shape=(3))
    child_level : int
    parent_level : int
    child_equivalent_density : np.array(shape=(n))
        The equivalent densities calculated in the previous step at the `n`
        quadrature points at the child level.

    Returns:
    --------
    Potential
        Potential densities calculated at the `m` quadrature points of the
        parent level.
    """

    # Compute surfaces
    parent_equivalent_surface = surface(
        order=order,
        radius=radius,
        level=parent_level,
        center=parent_center,
        alpha=2.95
    )

    child_equivalent_surface = surface(
        order=order,
        radius=radius,
        level=child_level,
        center=child_center,
        alpha=2.95
    )

    child_check_surface = surface(
        order=order,
        radius=radius,
        level=child_level,
        center=child_center,
        alpha=1.05
    )

    # Calculate check potential from parent equivalent density
    check_potential = p2p(
        kernel_function=kernel_function,
        targets=child_check_surface,
        sources=parent_equivalent_surface,
        source_densities=parent_equivalent_density
    ).density

    # Calculate child downward equivalent density
    kernel_matrix = gram_matrix(
        kernel_function, parent_equivalent_surface, child_equivalent_surface)

    kernel_matrix_inv = pseudo_inverse(kernel_matrix)

    child_equivalent_density = np.matmul(kernel_matrix_inv, check_potential)

    return Potential(child_equivalent_surface, child_equivalent_density)
