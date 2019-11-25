"""
FMM in 2D, operates on Quadtree structure.
"""

from fmm.quadtree import find_parent

class Calculation:

    def __init__(self, key, name):
        self.key = key
        self.name = name

    def __repr__(self):
        return str(self.name)


def P2M(key):
    return Calculation(key, 'P2M')


def M2M(key):
    return Calculation(key, 'M2M')


def M2L(key):
    return Calculation(key, 'M2L')


def L2M(key):
    return Calculation(key, 'L2M')


def upward_pass(tree):

    result = 0
    level = tree.precision
    # Go up the tree from leaf level
    for leaf in tree.leaf_nodes:
        parent = find_parent(leaf)
        
        



