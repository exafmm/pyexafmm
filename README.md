<h1 align='center'>
PyExaFMM
</h1>

The goal of PyExaFMM is to develop a version of ExaFMM that is written
in Python as much as possible without sacrificing performance.

It is envisioned that the library mainly uses Numpy, Numba, Numexpr, and
Multiprocessing to achieve high performance. For offloading onto heterogoeneous
devices PyOpenCL and PyCuda may also be added.

Planned development milestones:

1.) Availability of all basic tree data structures in Python
2.) Basic non-optimised KI FMM on Python level
3.) Analysis of potential performance bottlenecks and 
    further optimisations using PyOpenCL
4.) Support for distributed compute devices
5.) Offloading on Nvidia GPUs with PyCuda
