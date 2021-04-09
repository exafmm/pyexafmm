"""
The Fmm object is responsible for data handling, calling optimised operator
functions implemented in distinct backend submodules, as well as implementing
the logic of the FMM loop.
"""
import time
import os
import pathlib

import h5py
import numba
import numpy as np

import adaptoctree.morton as morton

from fmm.backend import BACKEND
from fmm.kernel import KERNELS

import utils.data as data

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent
WORKING_DIR = pathlib.Path(os.getcwd())


class Fmm:
    """
    FMM class. Configure with pre-computed operators and octree.

    Example Usage:
    --------------
    >>> exp = Fmm('test_config')
    >>> exp.upward_pass()
    >>> exp.downward_pass()
    >>> # exp.run() # Run upward & downward passes together

    Parameters:
    -----------
    config_filename : str
        Filename of configuration json file used to pre-compute operators.
    """

    def __init__(self, config_filename=None):

        # Load experimental database
        if config_filename is not None:
            config_filepath = WORKING_DIR / f"{config_filename}.json"
        else:
            config_filepath = WORKING_DIR / "config.json"

        self.config = data.load_json(config_filepath)

        db_filepath = WORKING_DIR / f"{self.config['experiment']}.hdf5"
        self.db = h5py.File(db_filepath, "r")

        # Load required data from disk
        ## Load surfaces, and inverse gram matrices
        self.check_surface = self.db["surface"]["check"][...].astype(np.float32)
        self.ncheck_points = len(self.check_surface)
        self.equivalent_surface = self.db["surface"]["equivalent"][...].astype(np.float32)
        self.nequivalent_points = len(self.equivalent_surface)
        self.uc2e_inv = self.db["uc2e_inv"][...]
        self.dc2e_inv = self.db["dc2e_inv"][...]
        self.alpha_outer = np.float32(self.config['alpha_outer'])
        self.alpha_inner = np.float32(self.config['alpha_inner'])

        ## Load linear, and complete octrees alongside their parameters
        self.x0 = self.db["octree"]["x0"][...].astype(np.float32)
        self.r0 = np.float32(self.db["octree"]["r0"][...][0])
        self.depth = self.db["octree"]["depth"][...][0]
        self.leaves = self.db["octree"]["keys"][...]
        self.leaf_indices = np.zeros_like(self.leaves)
        self.nleaves = len(self.leaves)
        self.complete = self.db["octree"]["complete"][...]
        self.ncomplete = len(self.complete)
        self.complete_levels = morton.find_level(self.complete)

        ## Load source and target data
        self.sources = self.db["particle_data"]["sources"][...].astype(np.float32)
        self.nsources = len(self.sources)
        self.source_densities = self.db["particle_data"]["source_densities"][...].astype(np.float32)
        self.sources_to_keys = self.db["particle_data"]["sources_to_keys"][...]
        self.targets = self.db["particle_data"]["targets"][...].astype(np.float32)
        self.ntargets = len(self.targets)
        self.targets_to_keys = self.db["particle_data"]["targets_to_keys"][...]

        ## Load pre-computed operators
        self.m2m = self.db["m2m"][...]
        self.m2l = self.db["m2l"]
        self.l2l = self.db["l2l"][...]

        # Load interaction lists
        self.v_lists = self.db["interaction_lists"]["v"][...]
        self.x_lists = self.db["interaction_lists"]["x"]
        self.u_lists = self.db["interaction_lists"]["u"][...]
        self.w_lists = self.db["interaction_lists"]["w"]

        # Configure a compute backend and kernel functions
        self.kernel = self.config["kernel"]
        self.p2p_function = KERNELS[self.kernel]["p2p"]
        self.p2p_parallel_function = KERNELS[self.kernel]["p2p_parallel"]
        self.scale_function = KERNELS[self.kernel]["scale"]
        self.backend = BACKEND[self.config["backend"]]

        # Containers for results
        # self.target_potentials = np.zeros(self.ntargets, dtype=np.float32)
        self.multipole_expansions = np.zeros(self.nequivalent_points*self.ncomplete, dtype=np.float32)
        self.local_expansions = np.zeros(self.nequivalent_points*self.ncomplete, dtype=np.float32)

        # Map keys to their index in the complete tree, for looking up expansions
        self.key_to_index = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.int64
        )

        for i, k in enumerate(self.complete):
            self.key_to_index[k] = i

        # Create dictionaries for O(1) access to points in a given node
        self.key_to_sources = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.float64[:,:]
        )

        self.key_to_targets = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.float64[:,:]
        )

        # O(1) lookup for source densities in a given node
        self.key_to_source_densities = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.float64[:]
        )

        self.target_potentials = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.float64[:]
        )

        for i, k in enumerate(self.leaves):
            self.key_to_targets[k] = self.db['key_to_targets'][str(k)][...]
            self.key_to_sources[k] = self.db['key_to_sources'][str(k)][...]
            self.key_to_source_densities[k] = self.db['key_to_source_densities'][str(k)][...]
            ntargets = len(self.key_to_targets[k])
            self.target_potentials[k] = np.zeros(ntargets)

    def upward_pass(self):
        """
        Post-order traversal of tree, compute multipole expansions for all
            nodes.
        """

        start = time.time()
        # Form multipole expansions for all leaf nodes
        self.backend['p2m'](
                leaves=self.leaves,
                nleaves=self.nleaves,
                key_to_index=self.key_to_index,
                key_to_sources=self.key_to_sources,
                key_to_source_densities=self.key_to_source_densities,
                multipole_expansions=self.multipole_expansions,
                nequivalent_points=self.nequivalent_points,
                x0=self.x0,
                r0=self.r0,
                alpha_outer=self.alpha_outer,
                check_surface=self.check_surface,
                ncheck_points=self.ncheck_points,
                uc2e_inv=self.uc2e_inv,
                p2p_function=self.p2p_function,
                scale_function=self.scale_function
            )

        print('p2m time: ', time.time()-start)

        start = time.time()
        # Post-order traversal
        for level in range(self.depth, 0, -1):
            keys = self.complete[self.complete_levels == level]

            self.backend['m2m'](
                keys=keys,
                multipole_expansions=self.multipole_expansions,
                nequivalent_points=self.nequivalent_points,
                m2m=self.m2m,
                key_to_index=self.key_to_index
            )

        print('m2m time ', time.time()-start)


    def downward_pass(self):
        """
        Pre-order traversal of tree. Compute local expansions for all nodes,
            and evaluate these at target points.
        """

        local_start = time.time()

        # Pre-order traversal
        for level in range(2, self.depth + 1):

            # Keys at this level
            keys = self.complete[self.complete_levels == level]

            # M2L operator stored in terms of its SVD components for each level
            str_level = str(level)
            u = self.m2l[str_level]["u"][...]
            s = np.diag(self.m2l[str_level]["s"][...])
            vt = self.m2l[str_level]["vt"][...]

            # Hashed transfer vectors for a given level, provide index for M2L operators
            hashes = self.m2l[str_level]["hashes"][...]

            print('level', level)
            # V List interactions
            start = time.time()

            self.backend['m2l'](
                    keys=keys,
                    key_to_index=self.key_to_index,
                    v_lists=self.v_lists,
                    multipole_expansions=self.multipole_expansions,
                    local_expansions=self.local_expansions,
                    dc2e_inv=self.dc2e_inv,
                    nequivalent_points=self.nequivalent_points,
                    ncheck_points=self.ncheck_points,
                    u=u,
                    s=s,
                    vt=vt,
                    hashes=hashes,
                    scale_function=self.scale_function,
                    depth=self.depth,
                )
            print('m2l time', time.time()-start)

            start = time.time()

            for key in keys:

                idx = self.key_to_index[key]

                # X List interactions
                x_list = self.x_lists[idx]
                x_list = x_list[x_list != -1]

                if len(x_list) > 0:
                    self.backend['s2l'](
                            key=key,
                            key_to_index=self.key_to_index,
                            x_list=x_list,
                            key_to_sources=self.key_to_sources,
                            key_to_source_densities=self.key_to_source_densities,
                            local_expansions=self.local_expansions,
                            x0=self.x0,
                            r0=self.r0,
                            alpha_inner=self.alpha_inner,
                            check_surface=self.check_surface,
                            nequivalent_points=self.nequivalent_points,
                            dc2e_inv=self.dc2e_inv,
                            scale_function=self.scale_function,
                            p2p_function=self.p2p_function
                        )

                # Translate local expansion to the node's children
                if level < self.depth:
                    self.backend['l2l'](
                        key=key,
                        local_expansions=self.local_expansions,
                        l2l=self.l2l,
                        key_to_index=self.key_to_index,
                        nequivalent_points=self.nequivalent_points
                    )

            print('l2l and s2l time', time.time()-start)

        print('total local transfer time ', time.time()-local_start)

        start = time.time()
        # Leaf near-field computations
        print('starting near field computations')
        for key in self.leaves:

            idx = self.key_to_index[key]

            target_coordinates = self.key_to_targets[key]
            ntargets = len(target_coordinates)

            u_list = self.u_lists[idx]
            u_list = u_list[u_list != -1]

            w_list = self.w_lists[idx]
            w_list = w_list[w_list != -1]

            if ntargets > 0:

                # W List interactions
                self.backend['m2t'](
                    target_key=key,
                    key_to_index=self.key_to_index,
                    w_list=w_list,
                    target_coordinates=target_coordinates,
                    target_potentials=self.target_potentials,
                    multipole_expansions=self.multipole_expansions,
                    x0=self.x0,
                    r0=self.r0,
                    alpha_inner=self.alpha_inner,
                    equivalent_surface=self.equivalent_surface,
                    nequivalent_points=self.nequivalent_points,
                    p2p_function=self.p2p_function
                )

                # # Evaluate local expansions at targets
                self.backend['l2t'](
                    key=key,
                    key_to_index=self.key_to_index,
                    target_coordinates=target_coordinates,
                    target_potentials=self.target_potentials,
                    local_expansions=self.local_expansions,
                    x0=self.x0,
                    r0=self.r0,
                    alpha_outer=self.alpha_outer,
                    equivalent_surface=self.equivalent_surface,
                    nequivalent_points=self.nequivalent_points,
                    p2p_function=self.p2p_function
                )

        # U List interactions
        self.backend['near_field_u_list'](
            u_lists=self.u_lists,
            leaves=self.leaves,
            key_to_sources=self.key_to_sources,
            key_to_targets=self.key_to_targets,
            key_to_source_densities=self.key_to_source_densities,
            key_to_index=self.key_to_index,
            max_points=self.config['max_points'],
            p2p_parallel_function=self.p2p_parallel_function
        )

        print('near field time ', time.time()-start)

    def run(self):
        """Run full algorithm"""
        start = time.time()
        self.upward_pass()
        self.downward_pass()
        print()
        print('total time', time.time()-start)