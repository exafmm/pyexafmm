"""
Compress a given set of m2l operators.
"""
import os
import pathlib
import re
import sys

import numpy as np

import fmm.hilbert as hilbert
from fmm.octree import Octree
import utils.data as data
import utils.svd as svd


HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent


def get_level(filename):
    """Get level from the m2l operator's filename"""
    pattern = '(?<=level_)(.*)(?=.pkl)'
    prog = re.compile(pattern)
    level = int(prog.search(filename).group())
    return level


def main(**config):
    """
    Main script, configure using config.json file in module root.
    """

    data_dirpath = PARENT / f"{config['data_dirname']}/"
    m2l_dirpath = PARENT / f"{config['operator_dirname']}/"

    # Check if compression already computed
    if data.file_in_directory('m2l_compressed', m2l_dirpath, ext='pkl'):
        print("Already Compressed M2L Operators for this configuration.")

    else:

        # Load all M2L operators calculated, and their lookup tables into containers
        operator_files = m2l_dirpath.glob('m2l_level*')
        index_to_key_files = m2l_dirpath.glob('index*')

        m2l_operators = {
            level: None for level in range(2, config['octree_max_level']+1)
        }

        m2l_index_to_key = {
            level: None for level in range(2, config['octree_max_level']+1)
        }

        for filename in operator_files:
            level = get_level(str(filename))
            m2l_operators[level] = data.load_pickle(
                f'm2l_level_{level}', m2l_dirpath
            )

        for filename in index_to_key_files:
            level = get_level(str(filename))
            m2l_index_to_key[level] = data.load_pickle(
                f'index_to_key_level_{level}', m2l_dirpath
            )

        # Instantiate Octree, for ease of Hilbert method access
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


        # Container to store compressed operators
        m2l_compressed = {k: None for k in octree.non_empty_target_nodes}

        # Pre-order traversal of tree, collating a list of m2l operators for
        # each target.
        for level in range(2, 1 + octree.maximum_level):
            for target in octree.non_empty_target_nodes_by_level[level]:
                index = octree.target_node_to_index[target]

                # Container for collated m2l operators
                m2l = []
                for neighbor_list in octree.interaction_list[index]:

                    for source in neighbor_list:
                        if source != -1:

                            # Lookup required operators from dense calculated
                            # m2l operators.
                            target_index = hilbert.remove_level_offset(target)

                            index_to_key = m2l_index_to_key[level][target_index]
                            source_index = np.where(source == index_to_key)[0][0]

                            operator = m2l_operators[level][target_index][source_index]

                            m2l.append(operator)

                # Run Compression, with configured tolerance.
                m2l = np.bmat(m2l)

                m2l_compressed[target] = svd.compress(m2l, target_rank=config['target_rank'])

            print(f"Compressed M2L Operators for target nodes at level {level}")


        # Filter out keys without associated m2l operators, and save
        m2l_compressed = {
            k: v
            for k, v in m2l_compressed.items()
            if v is not None
        }

        print("Saving All Compressed M2L Operators")
        data.save_pickle(m2l_compressed, config['m2l_compressed_filename'], m2l_dirpath)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError(
            f'Must Specify Config Filepath!\
                e.g. `python compress_m2l_operators.py /path/to/config.json`')
    else:
        config_filepath = sys.argv[1]
        config = data.load_json(config_filepath)
        main(**config)
