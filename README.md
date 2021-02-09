<h1 align='center'>
PyExaFMM
</h1>

The goal of PyExaFMM is to develop a highly performant implementation of the
particle FMM that is written in Python. The utility of FMM algorithms are hindered
by their relatively complex implementation, especially for achieving high-performance.
PyExaFMM is a particle kernel-independent FMM based on [1], written in pure Python
with some extensions. Representing a compromise between portability, east of use,
and performance. Optimisations are currently implemented  using Numba, Numpy, and
Multiprocessing. However the vision of the project is to eventually provide
optimisations taking advantage distributed and heterogenous computing environments,
and to scale from desktops to HPC clusters.

## Install

Build from source, and install locally into a Conda/Miniconda environment

```bash
# Clone repository
git clone git@github.com:exafmm/pyexafmm.git
cd pyexafmm

# Build Conda package
conda build conda.recipe

# Install conda package
conda install --use-local pyexafmm

# (For developers) Install in editable mode
pip3 install -e .
```

## Configure

After installation, use provided scripts to precompute and cache FMM operators,

e.g.

```bash
exafmm compute-operators
```

Make sure to configure the FMM simulation using the `config.json` file.

```json
{
    "order": 3,
    "operator_dirname": "precomputed_operators_order_3_test",
    "surface_filename": "surface",
    "kernel": "laplace",
    "alpha_inner": 1.05,
    "alpha_outer": 2.95,
    "data_dirname": "data_1k_random_test",
    "source_filename": "sources",
    "target_filename": "targets",
    "source_densities_filename": "source_densities",
    "octree_max_level": 3,
    "target_rank": 3,
    "m2l_compressed_filename": "m2l_compressed"
}
```

|Parameter    | Description |
|---	    |---	 |
| `order`	| Order of local and multipole expansions. |
| `operator_dirname`	| Directory in which to store operator precomputations. |	|
| `surface_filename`	| Filename to use for cached surface. |
| `kernel` | Kernel function to use. |
| `alpha_inner`	| Relative size of inner surface's radius. |	|
| `alpha_outer`	| Relative size of outer surface's radius. |
| `data_dirname` | Directory in which to store particle data. |
| `source_filename` | Filename to use for source particles generated. |
| `target_filename` | Filename to use for target particles generated. |
| `source_densities_filename` | Filename to use for source densities generated. |
| `octree_max_level` | Depth of octree to use in simulations. |
| `target_rank` | Target rank in low-rank compression of M2L matrix. |
| `m2l_compressed_filename` | Filename to use for compressed M2L matrix. |


## CLI

```bash
exafmm [OPTIONS] COMMAND [ARGS]
```

|Command    | Action |
|---	    |---	 |
| `build`	| Build dev version in current python env |
| `test`	| Run test suite	|
| `lint`	| Run project linter 	|
| `compute-operators` | Run operator pre-computations |
| `generate-test-data [npoints] [dtype]` | Generate `npoints` random sources & targets `dtype = random or separated`|
| `recompute-operators` | Clear cache of computed operators, and recalculate with current the config |
| `compress-m2l` | Compress M2L Operators computed via `compute-operators` |
| `recompress-m2l` | Clear cache, and re-compress M2L operators |


## References

[1] Ying, L., Biros, G., & Zorin, D. (2004). A kernel-independent adaptive fast multipole algorithm in two and three dimensions. Journal of Computational Physics, 196(2), 591-626.
