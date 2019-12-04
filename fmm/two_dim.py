"""
Code for simulating FMM in 2D.
"""

from dask.delayed import delayed
import numpy as np

from fmm.quadtree import (
    find_parent, find_children, curve_to_point, point_to_curve
)


def m2m(child_potentials):
    """
    M2M operator. Sums together potentials from child sources.

    Parameters:
    -----------
    child_potentials : list[float/Delayed]
        Potentials of child nodes.

    Returns:
    --------
    float/Delayed
        Parent potential.
    """
    return sum(child_potentials)


def l2l(parent_potential):
    """
    L2L operator. Distributes potential of parent target across children.
    """

    return np.divide(parent_potential, 4)


def m2l(source_potential, target_potential):
    """
    Dummy M2M operator.

    Parameters:
    -----------
    source : int
    target : int
    m2m_result : int/Delayed

    Returns:
    --------
    int
    """

    return sum([source_potential, target_potential])


def direct_summation(tree, level, key):
    """
    Dummy direct summation at leaf level

    Parameters:
    -----------
    source : int
    target : int

    Returns:
    --------
    int
    """
    return 1


def upward_pass(tree):
    """
    Calculate task graph from upward pass over quadtree.

    Parameters:
    -----------
    tree : Quadtree
        Quadtree containing sources/targets.

    Returns:
    --------
    {int:{int:Delayed}}
        Dask task graph where the keys in the outer dictionary correspond to the
        level of the Quadtree, and the keys in the inner dictionary correspond
        to the Z-order keys of the parent-box for each box at this level. The
        Delayed values are the evaluated value of the M2M operator run at the
        leaf level.
    """
    leaves = tree.leaf_node_potentials

    current_level = tree.n_levels
    results = {i:dict() for i in range(1, current_level+1)}

    # Add leaf values to results for help with downward pass
    results[current_level+1] = {
        idx: val for idx, val in enumerate(tree.leaf_node_potentials)
        }

    while len(leaves) > 1:

        parent_leaves = []

        for i in range(0, len(leaves), 4):

            parent = find_parent(i)

            f_b = []
            for f_bi in leaves[i:i+4]:
                f_b.append(f_bi)

            m2m_result = delayed(m2m)(f_b)

            results[current_level][parent] = m2m_result
            parent_leaves.append(m2m_result)

        current_level -= 1
        leaves = parent_leaves

    return results


def downward_pass(tree, m2m_results):
    """
    Calculate downward pass of FMM algorithm.

    Parameters:
    -----------
    tree : Quadtree
        Quadtree containing sources/targets.

    m2m_results : {int: {int: Delayed}}
        Dask task graph where the keys in the outer dictionary correspond to the
        level of the Quadtree, and the keys in the inner dictionary correspond
        to the Z-order keys of the parent-box for each box at this level. The
        Delayed values are the evaluated value of the M2M operator run at the
        leaf level.

    Returns:
    --------
    m2l_results : {int: {int: Delayed}}
        Dictionary where the outer keys correspond to the level of the Quadtree,
        and the inner keys the Z-order keys of the nodes at this level. The
        Delayed values are the evalued M2L operator run at this level.
    """

    leaf_level = tree.n_levels
    root_level = 1

    results = {
        level: {key: 0 for key in range(4**level)}
        for level in range(1, leaf_level+1)
    }

    for level in range(1, leaf_level):

        precision = 2**(level-1)
        if root_level < level < leaf_level+1:

            sources = m2m_results[level+1].keys()
            for source in sources:

                interaction_list = calculate_interaction_list(source, precision)
                source_potential = m2m_results[level+1][source]
                for target in interaction_list:

                    target_potential = results[level][target]

                    # M2L operation
                    m2l_result = delayed(m2l)(
                        source_potential, target_potential
                    )

                    # L2L operation
                    parent = find_parent(target)
                    l2l_result = delayed(l2l)(results[level-1][parent])
                    results[level][target] = delayed(sum)(
                        [m2l_result, l2l_result]
                    )

    # At leaf level just do direct summation, as well as L2L
    for leaf_key in range(tree.n_leaves):
        # L2L operation
        parent = find_parent(leaf_key)
        l2l_result = delayed(l2l)(results[leaf_level-1][parent])

        # Direct sum operation
        direct_sum_result = delayed(direct_summation)(tree, level, leaf_key)

        results[leaf_level][leaf_key] = l2l_result + direct_sum_result

    return results


def find_neighbours(key, precision):
    """
    Find nearest neighbour nodes.

    Parameters:
    -----------
    key : int
        The distance of a node along a Z-order curve.
    precision : int
        The precision of the tree being considered.

    Returns:
    --------
    {int}
        Set of keys that the nearest neighbours.
    """
    y, x = curve_to_point(key, precision)

    n = y+1, x
    s = y-1, x
    w = y, x-1
    e = y, x+1

    nw = y+1, x-1
    ne = y+1, x+1
    sw = y-1, x-1
    se = y-1, x+1

    neighbours = [n, ne, e, se, s, sw, w, nw]

    z_neighbours = []

    for neighbour in neighbours:
        try:
            dist = point_to_curve(*neighbour, precision)
            z_neighbours.append(dist)
        except ValueError:
            pass

    return set(z_neighbours)


def calculate_interaction_list(key, precision):
    """
    Caculate the interaction list of a given node, at a given precision.

    Parameters:
    -----------
    key : int
        The distance of a node along a Z-order curve.
    precision : int
        The precision of the tree being considered.

    Returns:
    --------
    {int}
        Set of keys in the interaction list.
    """

    parent_key = find_parent(key)
    parent_precision = precision // 2
    parent_nodes = {i for i in range(0, (2*parent_precision)**2)}
    child_nodes = {i for i in range(0, (2*precision)**2)}

    child_near_field = find_neighbours(key, precision)
    child_near_field.add(key)
    parent_near_field = find_neighbours(parent_key, parent_precision)
    parent_near_field.add(parent_key)

    parent_far_field = parent_nodes.symmetric_difference(parent_near_field)

    parent_far_field_children = set()

    for parent in parent_far_field:
        for child in find_children(parent):
            parent_far_field_children.add(child)

    interaction_list = child_nodes.symmetric_difference(
        parent_far_field_children
    )
    interaction_list = interaction_list.symmetric_difference(child_near_field)

    return interaction_list
