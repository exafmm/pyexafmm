"""
Implementation of the main FMM loop.

Matrix-Vector products are computed in single precision, with appropriate
casting. However, as AdaptOctree's implementation depends on 64 bit Morton
keys, key-handling is generally left in double precision.
"""
from fmm.kernel import KERNELS
import os
import pathlib

import h5py
import numba
import numpy as np

import adaptoctree.morton as morton

from fmm.backend import BACKEND

import utils.data as data

HERE = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PARENT = HERE.parent
WORKING_DIR = pathlib.Path(os.getcwd())


class Fmm:
    """
    FMM class. Configure with pre-computed operators and octree.

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
        self.v_lists = self.db["interaction_lists"]["v"]
        self.x_lists = self.db["interaction_lists"]["x"]
        self.u_lists = self.db["interaction_lists"]["u"]
        self.w_lists = self.db["interaction_lists"]["w"]

        # Configure a compute backend and kernel
        self.kernel = self.config["kernel"]
        self.p2p_function = KERNELS[self.kernel]["p2p"]
        self.scale_function = KERNELS[self.kernel]["scale"]
        self.backend = BACKEND[self.config["backend"]]

        #  Containers for results
        self.target_potentials = np.zeros(self.ntargets, dtype=np.float32)
        self.multipole_expansions = np.zeros(self.nequivalent_points*self.ncomplete, dtype=np.float32)
        self.local_expansions = np.zeros(self.nequivalent_points*self.ncomplete, dtype=np.float32)

        # Map a key to it's index in the complete tree, for looking up expansions
        self.key_to_index = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.int64
        )

        for i, k in enumerate(self.complete):
            self.key_to_index[k] = i

    def upward_pass(self):
        """
        Post-order traversal of tree, compute multipole expansions for all
            nodes.
        """

        # Form multipole expansions for all leaf nodes
        self.backend['p2m'](
                leaves=self.leaves,
                nleaves=self.nleaves,
                key_to_index=self.key_to_index,
                sources=self.sources,
                source_densities=self.source_densities,
                sources_to_keys=self.sources_to_keys,
                multipole_expansions=self.multipole_expansions,
                nequivalent_points=self.nequivalent_points,
                x0=self.x0,
                r0=self.r0,
                alpha_outer=self.alpha_outer,
                check_surface=self.check_surface,
                uc2e_inv=self.uc2e_inv,
                p2p_function=self.p2p_function,
                scale_function=self.scale_function
            )

        # Post-order traversal
        for level in range(self.depth, 0, -1):
            keys = self.complete[self.complete_levels == level]
            for idx in range(len(keys)):
                key = keys[idx]
                self.backend['m2m'](
                        key=key,
                        multipole_expansions=self.multipole_expansions,
                        nequivalent_points=self.nequivalent_points,
                        m2m=self.m2m,
                        key_to_index=self.key_to_index,
                    )

    def downward_pass(self):
        """
        Pre-order traversal of tree. Compute local expansions for all nodes,
            and evaluate these at target points.
        """

        # Pre-order traversal of octree
        for level in range(2, self.depth + 1):

            idxs = self.complete_levels == level

            #  M2L operator stored in terms of its SVD components for each level
            str_level = str(level)
            u = self.m2l[str_level]["u"][...]
            s = np.diag(self.m2l[str_level]["s"][...])
            vt = self.m2l[str_level]["vt"][...]

            # Hashed transfer vectors for a given level, provide index for M2L operators
            hashes = self.m2l[str_level]["hashes"][...]

            for key in self.complete[idxs]:

                idx = self.key_to_index(key)

                # V List interactions
                v_list = self.v_lists[idx]
                v_list = v_list[v_list != -1]

                if len(v_list) > 0:
                    self.backend['m2l'](
                            key=key,
                            key_to_index=self.key_to_index,
                            depth=self.depth,
                            v_list=v_list,
                            multipole_expansions=self.multipole_expansions,
                            local_expansions=self.local_expansions,
                            dc2e_inv=self.dc2e_inv,
                            nequivalent_points=self.nequivalent_points,
                            ncheck_points=self.ncheck_points,
                            u=u,
                            s=s,
                            vt=vt,
                            hashes=hashes,
                            scale_function=self.scale_function
                        )

                # X List interactions
                x_list = self.x_lists[idx]
                x_list = x_list[x_list != -1]

                if len(x_list) > 0:
                    self.backend['s2l'](
                            key=key,
                            key_to_index=self.key_to_index,
                            x_list=x_list,
                            sources=self.sources,
                            source_densities=self.source_densities,
                            sources_to_keys=self.sources_to_keys,
                            local_expansions=self.local_expansions,
                            x0=self.x0,
                            r0=self.r0,
                            alpha_inner=self.alpha_inner,
                            check_surface=self.check_surface,
                            ncheck_points=self.ncheck_points,
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

        # Leaf near-field computations
        for key in self.leaves:

            idx = self.key_to_index(key)

            w_list = self.w_lists[idx]
            w_list = w_list[w_list != -1]

            u_list = self.u_lists[idx]
            u_list = u_list[u_list != -1]

            # Evaluate local expansions at targets
            self.backend['l2t'](
                key=key,
                key_to_index=self.key_to_index,
                targets=self.targets,
                targets_to_keys=self.targets_to_keys,
                target_potentials=self.target_potentials,
                local_expansions=self.local_expansions,
                x0=self.x0,
                r0=self.r0,
                alpha_outer=self.alpha_outer,
                equivalent_surface=self.equivalent_surface,
                nequivalent_points=self.nequivalent_points,
                p2p_function=self.p2p_function
            )

            # W List interactions
            if len(w_list) > 0:
                self.backend['m2t'](
                    key=key,
                    key_to_index=self.key_to_index,
                    w_list=w_list,
                    targets=self.targets,
                    targets_to_keys=self.targets_to_keys,
                    target_potentials=self.target_potentials,
                    multipole_expansions=self.multipole_expansions,
                    x0=self.x0,
                    r0=self.r0,
                    alpha_inner=self.alpha_inner,
                    equivalent_surface=self.equivalent_surface,
                    nequivalent_points=self.nequivalent_points,
                    p2p_function=self.p2p_function
                )

            # U List interactions
            self.backend['near_field'](
                key=key,
                u_list=u_list,
                targets=self.targets,
                targets_to_keys=self.targets_to_keys,
                target_potentials=self.target_potentials,
                sources=self.sources,
                source_densities=self.source_densities,
                sources_to_keys=self.sources_to_keys,
                p2p_function=self.p2p_function
            )

    def run(self):
        """Run full algorithm"""
        self.upward_pass()
        self.downward_pass()
