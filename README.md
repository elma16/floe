# floe (🌊🧊)

## Contents :

This is the repository for my master's thesis, titled "Sea Ice Dynamics Using Finite Elements". This is a flexible, extensible piece of software which replicates the results of ["Sea Ice Dynamics on Triangular Grids"](https://arxiv.org/abs/2006.00547) by [Carolin Mehlmann](https://mpimet.mpg.de/en/staff/carolin-mehlmann) and [Peter Korn](https://mpimet.mpg.de/en/staff/peter-korn/teaching-and-supervision) using a hierarchy of sea ice model classes to organise the different levels of complexity. This project also used an implicit timestepping scheme to attempt to achieve larger timesteps than explicit timestepping schemes. The spatial discretisation was implemented by using the finite element method. This was achieved by using [Firedrake](https://www.firedrakeproject.org).

## Installation :

A key dependency on this project is Firedrake. Here are [installation instructions](https://www.firedrakeproject.org/download.html) for this package.

In order to install the package to run the examples, run the following command in the directory outside floe.

    pip install -e floe
