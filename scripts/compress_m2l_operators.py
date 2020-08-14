"""
Compress a given set of m2l operators.
"""
import os
import pathlib
import sys

import numpy as np

import fmm.hilbert as hilbert
from fmm.octree import Octree
import fmm.operator as operator
import utils.data as data
import utils.svd as svd


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent


def main(**config):
    data_dirpath = PARENT / f"{config['data_dirname']}/"
    m2l_dirpath = PARENT / f"{config['operator_dirname']}/"

    # Step 0: Construct Octree and load Python config objs
    print("source filename", data_dirpath)

    sources = data.load_hdf5_to_array(
        config['source_filename'],
        config['source_filename'],
        data_dirpath
        )

    targets = data.load_hdf5_to_array(
        config['target_filename'],
        config['target_filename'],
        data_dirpath
        )

    source_densities = data.load_hdf5_to_array(
        config['source_densities_filename'],
        config['source_densities_filename'],
        data_dirpath
        )

    octree = Octree(
        sources, targets, config['octree_max_level'], source_densities
        )

    m2l_compressed = {k: None for k in octree.non_empty_target_nodes}

    m2l_obj = operator.M2L(config_filepath)

    for level in range(2, 1 + octree.maximum_level):
        for target in octree.non_empty_target_nodes_by_level[level]:
            index = octree.target_node_to_index[target]

            m2l = []
            for neighbor_list in octree.interaction_list[index]:

                for source in neighbor_list:
                    if source != -1:

                        target_index = hilbert.remove_level_offset(target)

                        index_to_key = m2l_obj.index_to_key[level][target_index]
                        source_index = np.where(source == index_to_key)[0][0]

                        m2l_matrix = m2l_obj.operators[level][target_index][source_index]

                        m2l.append(m2l_matrix)

            # Run Compression
            m2l = np.bmat(m2l)
            m2l_compressed[target] = svd.compress(m2l)

    # Save results
    print("Saving Compressed M2L Operators")
    # Filter out keys without associated m2l operators

    m2l_compressed = {
        k: v
        for k, v in m2l_compressed.items()
        if v is not None
    }

    data.save_pickle(m2l_compressed, 'm2l_compressed', m2l_dirpath)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError(
            f'Must Specify Config Filepath!\
                e.g. `python compress_m2l_operators.py /path/to/config.json`')
    else:
        config_filepath = sys.argv[1]
        config = data.load_json(config_filepath)
        main(**config)
