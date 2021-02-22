<h1 align='center'>
PyExaFMM
</h1>

The goal of PyExaFMM is to develop a highly performant implementation of the
particle FMM that is written in Python.

The utility of FMM algorithms are hindered  by their relatively complex implementation, especially for achieving high-performance.

PyExaFMM is a particle kernel-independent FMM based on [1], written in pure Python
with some extensions. Representing a compromise between portability, east of use,
and performance. Optimisations are currently implemented  using Numba, Numpy, and
CUDA acceleration.

The vision of the project is to eventually provide optimisations fully taking advantage
distributed and heterogenous computing environments, and to scale from desktops to HPC clusters.
Most importantly however, Python will allow non-specialist scientists and engineers to solve
particle FMM problems, from a jupyter notebook!

## System Requirements

An NVidia GPU is required, as PyExaFMM is accellerated with CUDA.

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
python setup.py develop
```

## Configure

After installation, you must precompute and cache the FMM operators for your dataset. Most of these are calculated using the techniques in [1], notably M2M and L2L matrices can be computed for a single parent node and its children, and scaled in a simple fashion for the kernels implemented by PyExaFMM. For the M2L operators, we introduce a randomised SVD compression [2], to avoid storing and applying potentially very large dense matrices.

This is done via a `config.json` file,

```json
{
    "experiment": "fmm",
    "npoints": 10000,
    "data_type": "random",
    "order": 5,
    "kernel": "laplace",
    "alpha_inner": 1.05,
    "alpha_outer": 2.95,
    "max_level": 10,
    "max_points": 100,
    "target_rank": 1
}
```

|Parameter      | Description                                        |
|--------------	|-----------------------------------------------	 |
| `experiment`	| Order of local and multipole expansions.           |
| `npoints`     | Number of points to generate in test data.         |
| `data_type`   | Type of test data to generate.                     |
| `order`	    | Order of local and multipole expansions.           |
| `kernel`      | Kernel function to use.                            |
| `alpha_inner`	| Relative size of inner surface's radius.           |
| `alpha_outer`	| Relative size of outer surface's radius.           |
| `max_level`   | Depth of octree to use in simulations.             |
| `target_rank` | Target rank in low-rank compression of M2L matrix. |

PyExaFMM provides some simple test-data generation functions, which can be configured for. However, to use your own data, simply create a HDF5 file, with the same name as `experiment` in your configuration file, with the following group hierarchy,

```bash
particle_data/
    |_ sources/
    |_ source_densities/
    |_ targets/
```

where `sources` and `targets` are the coordinates of your source and target particles respectivley, of shape `(nsources/ntargets, 3)`, and source densities is of shape `(nsources, 1)`.

The CLI workflow is as follows,

```bash

# Generate test data (optional)
fmm generate-test-data

# Run operator pre-computations
fmm compute-operators
```

Once this is done, you are ready to start programming with PyExaFMM.

## CLI

```bash
fmm [OPTIONS] COMMAND [ARGS]
```

|Command               | Action |
|--------------        |------------------------------------	 |
| `build`	           | Build dev version in current python env |
| `test`	           | Run test suite	                         |
| `lint`	           | Run project linter 	                 |
| `compute-operators`  | Run operator pre-computations           |
| `generate-test-data` | Generate `npoints` sources & targets    |


## References

[1] Ying, L., Biros, G., & Zorin, D. (2004). A kernel-independent adaptive fast multipole algorithm in two and three dimensions. Journal of Computational Physics, 196(2), 591-626.

[2] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.