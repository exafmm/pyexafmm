"""
The Fmm object acts as the API for interacting the PyExaFMM.
"""
import os
import pathlib
from time import time

import h5py
import numba
import numpy as np

import adaptoctree.morton as morton

from fmm.backend import BACKEND
from fmm.dtype import NUMBA, NUMPY
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
    >>> e = Fmm('config') # Initialise experiment
    >>> e.run() # Run upward & downward passes together
    >>> e.clear() # Clear containers, to re-run experiment
    """

    def __init__(self, config_filename=None, verbose=False):
        """
        The Fmm object is instantiated with a config JSON. This specifies the
        HDF5 database used in operator and tree precomputations.

        Notes:
        ------
        The configuration file and the HDF5 database must be located in the same
        directory to be picked up by PyExaFMM.

        Parameters:
        -----------
        config_filename : str
        verbose : bool
            Optionally print operator runtimes to stdout.
        """
        # Load experiment database
        if config_filename is not None:
            config_filepath = WORKING_DIR / f"{config_filename}.json"
        else:
            config_filepath = WORKING_DIR / "config.json"

        self.config = data.load_json(config_filepath)

        db_filepath = WORKING_DIR / f"{self.config['experiment']}.hdf5"
        self.db = h5py.File(db_filepath, "r")

        ## Operator runtimes
        operator_names = [
            'P2M', 'M2M', 'L2L', 'M2L', 'L2T', 'M2T', 'S2L', 'P2P'
        ]
        self.verbose = verbose
        self.times = {name: None for name in operator_names}

        ## Configure experiment precision
        self.numpy_dtype = NUMPY[self.config["precision"]]
        self.numba_dtype = NUMBA[self.config["precision"]]

        ## Load surfaces, and inverse gram matrices
        self.check_surface = self.db["surface"]["check"][...]
        self.ncheck_points = len(self.check_surface)
        self.equivalent_surface = self.db["surface"]["equivalent"][...]
        self.nequivalent_points = len(self.equivalent_surface)
        self.uc2e_inv_a = self.db["uc2e_inv_a"][...]
        self.uc2e_inv_b = self.db["uc2e_inv_b"][...]
        self.dc2e_inv_a = self.db["dc2e_inv_a"][...]
        self.dc2e_inv_b = self.db["dc2e_inv_b"][...]
        self.alpha_outer = self.config['alpha_outer']
        self.alpha_inner = self.config['alpha_inner']

        ## Load linear, and complete octrees alongside their parameters
        self.max_points = self.config["max_points"]
        self.x0 = self.db["octree"]["x0"][...]
        self.r0 = self.db["octree"]["r0"][...][0]
        self.depth = self.db["octree"]["depth"][...][0]
        self.leaves = self.db["octree"]["keys"][...]
        self.nleaves = len(self.leaves)
        self.complete = self.db["octree"]["complete"][...]
        self.ncomplete = len(self.complete)
        self.complete_levels = morton.find_level(self.complete)

        ## Load source and target data
        self.sources_to_keys = self.db["particle_data"]["sources_to_keys"][...]
        self.source_indices= self.db["particle_data"]["source_indices"][...]
        self.source_index_pointer = self.db["particle_data"]["source_index_pointer"][...]
        self.sources = self.db["particle_data"]["sources"][...][self.source_indices]
        self.source_densities = self.db["particle_data"]["source_densities"][...][self.source_indices]
        self.nsources = len(self.sources)

        self.targets_to_keys = self.db["particle_data"]["targets_to_keys"][...]
        self.target_indices = self.db["particle_data"]["target_indices"][...]
        self.target_index_pointer = self.db["particle_data"]["target_index_pointer"][...]
        self.targets = self.db["particle_data"]["targets"][...][self.target_indices]
        self.ntargets = len(self.targets)

        ## Load pre-computed operators
        self.m2m = self.db["m2m"][...]
        self.m2l = self.db["m2l"]
        self.l2l = self.db["l2l"][...]

        ## Load interaction lists
        self.v_lists = self.db["interaction_lists"]["v"][...]
        self.x_lists = self.db["interaction_lists"]["x"][...]
        self.u_lists = self.db["interaction_lists"]["u"][...]
        self.w_lists = self.db["interaction_lists"]["w"][...]

        ## Configure a compute backend and kernel functions
        kernel = self.config["kernel"]
        self.p2p_function = KERNELS[kernel]["p2p"]
        self.p2p_parallel_function = KERNELS[kernel]["p2p_parallel"]
        self.scale_function = KERNELS[kernel]["scale"]
        self.gradient_function = KERNELS[kernel]["gradient"]
        self.backend = BACKEND[self.config["backend"]]

        ## Containers for results
        self.multipole_expansions = np.zeros(
                self.nequivalent_points*self.ncomplete,
                dtype=self.numpy_dtype
            )

        self.local_expansions = np.zeros(
               self.nequivalent_points*self.ncomplete,
               dtype=self.numpy_dtype
            )

        self.target_potentials = np.zeros(
            (self.ntargets, 4),
            dtype=self.numpy_dtype
        )

        self.key_to_leaf_index = numba.typed.Dict.empty(key_type=numba.int64, value_type=numba.int64)
        self.key_to_index = numba.typed.Dict.empty(key_type=numba.int64, value_type=numba.int64)

        for k in self.leaves:
            self.key_to_leaf_index[k] = np.argwhere(self.leaves == k)[0][0]

        for k in self.complete:
            self.key_to_index[k] = np.argwhere(self.complete == k)[0][0]

    def upward_pass(self):
        """
        Post-order traversal of tree, compute multipole expansions for all
            nodes.
        """
        # Form multipole expansions for all leaf nodes
        p2m_start = time()
        self.backend['p2m'](
                leaves=self.leaves,
                nleaves=self.nleaves,
                key_to_index=self.key_to_index,
                key_to_leaf_index=self.key_to_leaf_index,
                sources=self.sources,
                source_densities=self.source_densities,
                source_index_pointer=self.source_index_pointer,
                multipole_expansions=self.multipole_expansions,
                nequivalent_points=self.nequivalent_points,
                x0=self.x0,
                r0=self.r0,
                alpha_outer=self.alpha_outer,
                check_surface=self.check_surface,
                ncheck_points=self.ncheck_points,
                uc2e_inv_a=self.uc2e_inv_a,
                uc2e_inv_b=self.uc2e_inv_b,
                p2p_function=self.p2p_function,
                scale_function=self.scale_function,
                dtype=self.numpy_dtype
            )
        self.times['P2M'] = time()-p2m_start

        m2m_start = time()
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
        self.times['M2M'] = time()-m2m_start

    def downward_pass(self):
        """
        Pre-order traversal of tree. Compute local expansions for all nodes,
            and evaluate these at target points.
        """

        # Pre-order traversal
        m2l_total = 0
        l2l_total = 0

        for level in range(2, self.depth + 1):

            m2l_start = time()

            # Keys at this level
            keys = self.complete[self.complete_levels == level]
            scale = self.numpy_dtype(self.scale_function(level))

            # V List interactions
            # M2L operator stored in terms of its SVD components for each level
            str_level = str(level)
            u = self.m2l[str_level]["u"][...]
            s = np.diag(self.m2l[str_level]["s"][...])
            vt = self.m2l[str_level]["vt"][...]

            # Hashed transfer vectors for a given level, provide index for M2L operators
            hashes =self.m2l[str_level]["hashes"][...]

            hash_to_index = numba.typed.Dict.empty(
                key_type=numba.types.int64,
                value_type=numba.types.int64
            )

            for i, hash in enumerate(hashes):
                hash_to_index[hash] = i

            self.backend['m2l'](
                    keys=keys,
                    v_lists=self.v_lists,
                    u=u,
                    s=s,
                    vt=vt,
                    dc2e_inv_a=self.dc2e_inv_a,
                    dc2e_inv_b=self.dc2e_inv_b,
                    multipole_expansions=self.multipole_expansions,
                    local_expansions=self.local_expansions,
                    nequivalent_points=self.nequivalent_points,
                    key_to_index=self.key_to_index,
                    hash_to_index=hash_to_index,
                    scale=scale
                )

            m2l_total += time()-m2l_start

            l2l_start = time()
            for key in keys:
                # Translate local expansion from the node's parent
                self.backend['l2l'](
                    key=key,
                    local_expansions=self.local_expansions,
                    l2l=self.l2l,
                    key_to_index=self.key_to_index,
                    nequivalent_points=self.nequivalent_points
                )
            l2l_total += time()-l2l_start

        self.times['M2L'] = m2l_total
        self.times['L2L'] = l2l_total

        # Leaf near-field computations

        s2l_start = time()
        # X List interactions
        self.backend['s2l'](
            leaves=self.leaves,
            nleaves=self.nleaves,
            sources=self.sources,
            source_densities=self.source_densities,
            source_index_pointer=self.source_index_pointer,
            key_to_index=self.key_to_index,
            key_to_leaf_index=self.key_to_leaf_index,
            x_lists=self.x_lists,
            local_expansions=self.local_expansions,
            x0=self.x0,
            r0=self.r0,
            alpha_inner=self.alpha_inner,
            check_surface=self.check_surface,
            nequivalent_points=self.nequivalent_points,
            dc2e_inv_a=self.dc2e_inv_a,
            dc2e_inv_b=self.dc2e_inv_b,
            scale_function=self.scale_function,
            p2p_function=self.p2p_function,
            dtype=self.numpy_dtype
        )
        self.times['S2L'] = time()-s2l_start

        # W List interactions
        m2t_start = time()
        self.backend['m2t'](
            leaves=self.leaves,
            nleaves=self.nleaves,
            w_lists=self.w_lists,
            targets=self.targets,
            target_index_pointer=self.target_index_pointer,
            key_to_index=self.key_to_index,
            key_to_leaf_index=self.key_to_leaf_index,
            target_potentials=self.target_potentials,
            multipole_expansions=self.multipole_expansions,
            x0=self.x0,
            r0=self.r0,
            alpha_inner=self.alpha_inner,
            equivalent_surface=self.equivalent_surface,
            nequivalent_points=self.nequivalent_points,
            p2p_function=self.p2p_function,
            gradient_function=self.gradient_function
        )
        self.times['M2T'] = time()-m2t_start

        # Evaluate local expansions at targets
        l2t_start = time()
        self.backend['l2t'](
            leaves=self.leaves,
            nleaves=self.nleaves,
            key_to_index=self.key_to_index,
            key_to_leaf_index=self.key_to_leaf_index,
            targets=self.targets,
            target_potentials=self.target_potentials,
            target_index_pointer=self.target_index_pointer,
            local_expansions=self.local_expansions,
            x0=self.x0,
            r0=self.r0,
            alpha_outer=self.alpha_outer,
            equivalent_surface=self.equivalent_surface,
            nequivalent_points=self.nequivalent_points,
            p2p_parallel_function=self.p2p_parallel_function,
            dtype=self.numpy_dtype
        )
        self.times['L2T'] = time()-l2t_start

        p2p_start = time()

        self.backend['near_field'](
            self.leaves,
            self.nleaves,
            self.key_to_leaf_index,
            self.key_to_index,
            self.targets,
            self.u_lists,
            self.target_index_pointer,
            self.sources,
            self.source_densities,
            self.source_index_pointer,
            self.target_potentials,
            self.p2p_function,
            self.gradient_function,
            )
        self.times['P2P'] = time()-p2p_start

    def run(self):
        """Run full algorithm"""
        self.upward_pass()
        self.downward_pass()

        if self.verbose:
            for k, v in self.times.items():
                print(f"{k}: {v}")

    def clear(self):
        """Clear containers"""
        self.target_potentials = np.zeros_like(self.target_potentials)
        self.multipole_expansions = np.zeros_like(self.multipole_expansions)
        self.local_expansions = np.zeros_like(self.local_expansions)
