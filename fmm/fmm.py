"""
Implementation of the main FMM loop.

Matrix-Vector products are computed in single precision, with appropriate
casting. However, as AdaptOctree's implementation depends on 64 bit Morton
keys, key-handling is generally left in double precision.
"""
import os
import pathlib

import h5py
import numpy as np

import adaptoctree.morton as morton

from fmm.kernel import KERNELS
from fmm.backend import BACKENDS

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

        # Configure a kernel
        self.kernel = self.config["kernel"]
        self.eval = KERNELS[self.kernel]["eval"]
        self.p2p = KERNELS[self.kernel]["p2p"]
        self.scale = KERNELS[self.kernel]["scale"]

        # Configure a compute backend
        self.backend = BACKENDS[self.config["backend"]]

        #  Containers for results
        self.target_potentials = np.zeros(self.ntargets, dtype=np.float32)

        self.multipole_expansions = {
            key: np.zeros(self.nequivalent_points, dtype=np.float32) for key in self.complete
        }

        self.local_expansions = {
            key: np.zeros(self.nequivalent_points, dtype=np.float32) for key in self.complete
        }

    def upward_pass(self):
        """
        Post-order traversal of tree, compute multipole expansions for all
            nodes.
        """

        # Form multipole expansions for all leaf nodes
        for idx in range(self.nleaves):
            leaf = self.leaves[idx]
            self.particle_to_multipole(leaf)

        # Post-order traversal
        for level in range(self.depth-1, -1, -1):
            idxs = self.complete_levels == level
            for key in self.complete[idxs]:
                self.multipole_to_multipole(key)

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

                idx = np.where(self.complete== key)

                # V List interactions
                v_list = self.v_lists[idx]
                v_list = v_list[v_list != -1]
                if len(v_list) > 0:
                    self.multipole_to_local(key, v_list, u, s, vt, hashes)

                # X List interactions
                x_list = self.x_lists[idx]
                x_list = x_list[x_list != -1]
                if len(x_list) > 0:
                    self.source_to_local(key, x_list)

                # Translate local expansion to the node's children
                if level < self.depth:
                    self.local_to_local(key)

        # Leaf near-field computations
        for key in self.leaves:

            idx = np.where(self.complete == key)

            w_list = self.w_lists[idx]
            w_list = w_list[w_list != -1]

            u_list = self.u_lists[idx]
            u_list = u_list[u_list != -1]

            # Evaluate local expansions at targets
            self.local_to_target(key)

            # # W List interactions
            self.multipole_to_target(key, w_list)

            # U List interactions
            self.near_field(key, u_list)

    def run(self):
        """Run full algorithm"""
        self.upward_pass()
        self.downward_pass()

    def particle_to_multipole(self, key):
        """Compute multipole expansions from leaf particles."""
        self.backend['p2m'](
            key=key,
            sources=self.sources,
            source_densities=self.source_densities,
            sources_to_keys=self.sources_to_keys,
            multipole_expansions=self.multipole_expansions,
            x0=self.x0,
            r0=self.r0,
            alpha_outer=self.alpha_outer,
            check_surface=self.check_surface,
            uc2e_inv=self.uc2e_inv,
            scale_function=self.scale,
            p2p_function=self.p2p,
        )

    def multipole_to_multipole(self, key):
        """
        Combine multipole expansions of a node's children to approximate its
            own multipole expansion.
        """
        self.backend['m2m'](
            key=key,
            multipole_expansions=self.multipole_expansions,
            m2m=self.m2m,
        )

    def multipole_to_local(self, key, v_list, u, s, vt, hashes):
        """
        V List interactions.
        """
        self.backend['m2l'](
            key=key,
            depth=self.depth,
            v_list=v_list,
            multipole_expansions=self.multipole_expansions,
            local_expansions=self.local_expansions,
            dc2e_inv=self.dc2e_inv,
            ncheck_points=self.ncheck_points,
            scale_function=self.scale,
            u=u,
            s=s,
            vt=vt,
            hashes=hashes
        )

    def local_to_local(self, key):
        """
        Translate local expansion of a node to it's children.
        """
        self.backend['l2l'](
            key=key,
            local_expansions=self.local_expansions,
            l2l=self.l2l,
        )

    def source_to_local(self, key, x_list):
        """
        X List interactions.
        """
        self.backend['s2l'](
                key=key,
                x_list=x_list,
                sources=self.sources,
                source_densities=self.source_densities,
                sources_to_keys=self.sources_to_keys,
                local_expansions=self.local_expansions,
                x0=self.x0,
                r0=self.r0,
                alpha_inner=self.alpha_inner,
                check_surface=self.check_surface,
                dc2e_inv=self.dc2e_inv,
                scale_function=self.scale,
                p2p_function=self.p2p
        )

    def multipole_to_target(self, key, w_list):
        """
        W List interactions
        """
        self.backend['m2t'](
            key=key,
            w_list=w_list,
            targets=self.targets,
            targets_to_keys=self.targets_to_keys,
            target_potentials=self.target_potentials,
            multipole_expansions=self.multipole_expansions,
            x0=self.x0,
            r0=self.r0,
            alpha_inner=self.alpha_inner,
            equivalent_surface=self.equivalent_surface,
            p2p_function=self.p2p
        )

    def local_to_target(self, key):
        """
        Evaluate local potentials at target points
        """
        self.backend['l2t'](
            key=key,
            targets=self.targets,
            targets_to_keys=self.targets_to_keys,
            target_potentials=self.target_potentials,
            local_expansions=self.local_expansions,
            x0=self.x0,
            r0=self.r0,
            alpha_outer=self.alpha_outer,
            equivalent_surface=self.equivalent_surface,
            p2p_function=self.p2p,
        )

    def near_field(self, key, u_list):
        """
        U List interactions
        """
        self.backend['near_field'](
            key=key,
            u_list=u_list,
            targets=self.targets,
            targets_to_keys=self.targets_to_keys,
            target_potentials=self.target_potentials,
            sources=self.sources,
            source_densities=self.source_densities,
            sources_to_keys=self.sources_to_keys,
            p2p_function=self.p2p
        )
