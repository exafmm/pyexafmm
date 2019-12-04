"""
Code for simulating FMM in 2D.
"""

from dask.delayed import delayed

from fmm.quadtree import (
    find_parent, find_children, curve_to_point, point_to_curve
)


def m2m(child_potentials):
    """
    M2M operator. Sums together potentials from child nodes and returns a Dask
    Delayed object or a float corresponding to the parent potential.

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


def m2l(source, target, m2m_result):
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
    dummy = [0.5, 0.5]
    return delayed(sum)(dummy)


def direct_summation(source, target):
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

    m2l_results = {i: dict() for i in range(2, leaf_level+1)}

    for level in range(1, leaf_level):

        precision = 2**(level-1)

        if root_level < level < leaf_level+1:

            sources = m2m_results[level+1].keys()

            for source in sources:

                interaction_list = calculate_interaction_list(source, precision)

                for target in interaction_list:
                    m2m_result = m2m_results[level+1][source]
                    m2l_result = m2l(source, target, m2m_result)

                    m2l_results[level][target] = m2l_result

    # At leaf level just do direct summation, dummy for now
    m2l_results[leaf_level] = {i: 1 for i in range(tree.n_leaves)}

    return m2l_results


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
