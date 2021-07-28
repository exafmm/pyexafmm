<h1 align='center'>
PyExaFMM
</h1>

[![Anaconda-Server Badge](https://img.shields.io/conda/v/skailasa/pyexafmm.svg)](https://anaconda.org/skailasa/pyexafmm) [![Anaconda-Server Badge](https://anaconda.org/skailasa/pyexafmm/badges/latest_release_date.svg)](https://anaconda.org/skailasa/pyexafmm) [![Anaconda-Server Badge](https://anaconda.org/skailasa/pyexafmm/badges/platforms.svg)](https://anaconda.org/skailasa/pyexafmm)

PyExaFMM is an adaptive particle kernel-independent FMM based on [1], written in pure Python with some extensions. Representing a compromise between portability, maintainability, and performance.

## Install

Download from Anaconda cloud into a conda/mini-conda environment:

```bash
conda install -c skailasa pyexafmm
```

Developers may want to build from source:

```bash
# Add required channels
conda config --env --add channels skailasa
conda config --env --add channels conda-forge
conda config --env --add channels anaconda
conda config --env --add channels nvidia

# Clone
git clone git@github.com:exafmm/pyexafmm.git
cd pyexafmm

# Build
conda build conda.recipe

# Install
conda install --use-local pyexafmm

# Editable mode for live development
python setup.py develop
```

## Configuration

### Experimental Data

After installation, you must pre-compute and cache the FMM operators and octree for your dataset. Most of these are calculated using the techniques in [1], notably M2M and L2L matrices can be computed for a single parent node and its children, and scaled in a simple fashion for the kernels implemented by PyExaFMM. We sparsify the M2L operators using a technique based on randomized SVD compression [2] and transfer vectors [3].

This is done via a `config.json` file, PyExaFMM will look for this in your **current working directory**, which allows you to configure experimental parameters, as well as choose a kernel, and computational backend, which optimise the operator methods of the FMM using different approaches.

```json
{
    "experiment": "test",
    "npoints": 1000000,
    "data_type": "random",
    "order_equivalent": 5,
    "order_check": 5,
    "kernel": "laplace",
    "backend": "numba",
    "alpha_inner": 1.05,
    "alpha_outer": 2.9,
    "max_level": 10,
    "max_points": 150,
    "target_rank": 20,
    "precision": "single"
}
```

|Parameter      | Description                                        |
|--------------	|-----------------------------------------------	 |
| `experiment`	| Experiment name, used to label HDF5 database       |
| `npoints`     | Number of points to generate in test data using demo functions.         |
| `data_type`   | Type of test data to generate.                     |
| `order_equivalent`| Order of multipole expansions, same as discretization of equivalent surface.  |
| `order_check`     | Order of local expansions, same as discretization of check surface.           |
| `kernel`      | Kernel function to use, currently only supports laplace.          |
| `backend`      | Compute backend to use, currently only supports Numba.           |
| `alpha_inner`	| Relative size of inner surface's radius.           |
| `alpha_outer`	| Relative size of outer surface's radius.           |
| `max_level`   | Depth of octree to use in simulations.             |
| `target_rank` | Target rank in low-rank compression of M2L matrix. |
| `precision`   | Experimental precision, 'single' or 'double'.             |

PyExaFMM provides some simple test-data generation functions, which can be configured for. However, to use your own data, simply create a HDF5 file, with the same name as `experiment` in your configuration file, with the following group hierarchy,

```bash
particle_data/
    |_ sources/
    |_ source_densities/
    |_ targets/
```

where `sources` and `targets` are the coordinates of your source and target particles resp., of shape `(nsources/ntargets, 3)`, and source densities is of shape `(nsources, 1)`.

The CLI workflow is as follows,

```bash
# Generate test data (optional)
fmm generate-test-data -c config.json

# Run operator and tree pre-computations
fmm compute-operators -c config.json
```

Once this is done, you'll be left with a `.hdf5` database of precomputed parametrization, with the same name as your specified `experiment` parameter from your `config.json`. If you've used your own data, then the operator pre-computations will be written into the same HDF5 file.

### Configure Threading Layer

We optimize PyExaFMM for an OpenMP backend as the Numba threading layer. To avoid oversubscription due to nested parallelism, created when calling Numpy/Scipy functions from within OpenMP threads, we restrict internal Numpy thread pools to contain at most a single thread.

Optimum parameters (for Intel CPUs) are provided in the `.env` file, and set with,

```bash
# Set threading environment variables
source .env
```

## Usage

The `Fmm` class acts as the API for PyExaFMM. Example usage patterns are provided below:

```python
from fmm import Fmm

# Instantiate an experiment through an FMM object, with default 'config.json'
e = Fmm()

# Optionally specify non-default config filename, e.g. 'test_config.json'
# e = Fmm('test_config')

# Run FMM algorithm
e.run()

# Access targets, sources, and computed potentials/potential gradients at
# a given leaf
leaf = e.leaves[42]

# Dictionary mapping leaf key to a pointer defining alignment
leaf_idx = e.key_to_leaf_index[leaf]

# Targets at leaf
leaf_targets = e.targets[
    e.target_index_pointer[leaf_idx]:e.target_index_pointer[leaf_idx+1]
]

# Sources at leaf
leaf_sources = e.sources[
    e.source_index_pointer[leaf_idx]:e.source_index_pointer[leaf_idx+1]
]

# Source densities at leaf
leaf_source_densities = e.sources[
    e.source_index_pointer[leaf_idx]:e.source_index_pointer[leaf_idx+1]
]

# 4-vectors of target potentials/potential gradients aligned with 'leaf_targets'
leaf_potentials = e.target_potentials[
    e.target_index_pointer[leaf_idx]:e.target_index_pointer[leaf_idx+1]
]

# Access multipole/local expansions at a given tree node
key = e.complete[42]

# Dictionary mapping a given key to a pointer defining alignment
key_idx = e.key_to_index[key]

# Multipole expansions defined by equivalent surface
multipole_index = key_index*e.nequivalent_points
multipole_expansion = e.multipole_expansions[
    multipole_index:multipole_index+e.nequivalent_points
]

# Local expansions defined by check surface
local_index = key_index*e.ncheck_points
local_expansion = e.local_expansions[
    local_index:local_index+e.ncheck_points
]

# Clear potentials, gradients, and expansions to re-run experiment
e.clear()
```


## CLI

```bash
fmm [OPTIONS] COMMAND [ARGS]
```

|Command               | Action                                  | Options                 |
|--------------        |------------------------------------	 |--------------           |
| `compute-operators`  | Run operator pre-computations           | `-c <config_filename>`  |
| `generate-test-data` | Generate `npoints` sources & targets    | `-c <config_filename>`  |

The option to specify a custom config filename `-c` overrides the PyExaFMM default to search for a file named `config.json` in your current working directory to parameterize pre-computations.


## References

[1] Ying, L., Biros, G., & Zorin, D. (2004). A kernel-independent adaptive fast multipole algorithm in two and three dimensions. Journal of Computational Physics, 196(2), 591-626.

[2] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.

[3] Fong, W., & Darve, E. (2009). The black-box fast multipole method. Journal of Computational Physics, 228(23), 8712-8725.