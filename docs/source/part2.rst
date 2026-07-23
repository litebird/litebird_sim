Structure of the framework
==========================

This part describes the core building blocks of the framework: the central
``Simulation`` class, the objects it ties together (observations, detectors, the
instrument model), and the cross-cutting facilities shared by every module —
units, reports, reproducible random numbers, and MPI parallelism.

.. toctree::
   :maxdepth: 1

   interfaces.rst
   simulations.rst
   imo.rst
   detectors.rst
   plot_fp.rst
   observations.rst
   data_layout.rst
   maps_and_harmonics.rst
   units.rst
   profiling.rst
   reports.rst
   random_numbers.rst
   mpi.rst
   integrating.rst
