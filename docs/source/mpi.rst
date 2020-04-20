.. _using_mpi:

Using MPI
=========

The LiteBIRD Simulation Framework lists mpi4py as an *optional*
dependency. This means that simulation codes should be able to cope
with the lack of MPI.

The following table lists the variable that can be used to detect if
MPI is present or not:


+----------------------------+-------+---------------------------------------+
| Variable                   | Value | Meaning                               |
+----------------------------+-------+---------------------------------------+
| :data:`.HAVE_MPI4PY`       | Bool  | ``True`` if mpi4py was imported       |
+----------------------------+-------+---------------------------------------+
| :data:`.MPI_RANK`          | Int   | Rank of current MPI process, in       |
|                            |       | the range ``0 … MPI_SIZE - 1``        |
|                            |       | (0 if MPI is not available)           |
+----------------------------+-------+---------------------------------------+
| :data:`.MPI_SIZE`          | Int   | Number of available MPI processes     |
|                            |       | (1 if MPI is not available)           |
+----------------------------+-------+---------------------------------------+
| :data:`.MPI_CONFIGURATION` | Dict  | MPI configuration                     |
+----------------------------+-------+---------------------------------------+
| :data:`.MPI_COMM_WORLD`    |       | ``mpi4py.MPI.COMM_WORLD`` or ``None`` |
+----------------------------+-------+---------------------------------------+

The fact that :data:`.MPI_SIZE` and :data:`.MPI_RANK` are defined even
if ``mpi4py`` was not imported should make the creation of analysis
codes easier.


API reference
-------------

.. automodule:: litebird_sim.mpi
    :members:
    :undoc-members:
    :show-inheritance:
