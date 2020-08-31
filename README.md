<h1 align='center'>
PyExaFMM
</h1>

The goal of PyExaFMM is to develop a version of ExaFMM that is written in Python as much as possible without sacrificing performance.

It is envisioned that the library mainly uses Numpy, Numba, Numexpr, and Multiprocessing to achieve high performance. For offloading onto heterogenous
devices PyOpenCL and PyCuda may also be added.

## Milestones

1) Availability of all basic tree data structures in Python
1) Basic non-optimised KI FMM on Python level
1) Analysis of potential performance bottlenecks and further optimisations using PyOpenCL
1) Support for distributed compute devices
1) Offloading on Nvidia GPUs with PyCuda

## Install

We use Anaconda for environment management

1) Create an environment:

```bash

# Create exafmm environement
conda env create -f environment.yml

# Activate environment
conda activate exafmm

# Install PyExaFMM module
python setup.py install

# (Optional) For installation of CI module for developers
conda develop .
```

## Configure

After installation, use provided scripts to precompute and cache FMM operators,

e.g.

```bash
ci compute-operators
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

The operators are calculated from an Octree that is data dependent.


## CLI

```bash
ci [OPTIONS] COMMAND [ARGS]
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
