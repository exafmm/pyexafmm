<h1 align='center'>
PyExaFMM
</h1>

The goal of PyExaFMM is to develop a version of ExaFMM that is written in Python as much as possible without sacrificing performance.

It is envisioned that the library mainly uses Numpy, Numba, Numexpr, and Multiprocessing to achieve high performance. For offloading onto heterogoeneous devices PyOpenCL and PyCuda may also be added.

## Milestones

1) Availability of all basic tree data structures in Python
1) Basic non-optimised KI FMM on Python level
1) Analysis of potential performance bottlenecks and urther optimisations using PyOpenCL
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

## CLI

```bash
ci [OPTIONS] COMMAND [ARGS]
```

|Command    | Action |
|---	    |---	 |
| `build`	| Build dev version in current python env |
| `test`	| Run test suite	|
| `lint`	| Run project linter 	|