.. _using_mpi:

Using MPI
=========

The LiteBIRD Simulation Framework lists mpi4py as an *optional*
dependency. This means that simulation codes should be able to cope
with the lack of MPI.

The framework can be forced to use MPI or not using the variable
``LITEBIRD_SIM_MPI``:

- Set the variable to 1 or an empty value to *force* importing `mpi4py`;
- Set the variable to 0 to avoid importing `mpi4py`;
- If the variable is not set, the code will try to import `mpi4py`,
  but in case of error it will not complain and will silently shift to
  serial execution.

The framework provides a global variable, :data:`.MPI_COMM_WORLD`,
which is the same as ``mpi4py.MPI.COMM_WORLD`` if MPI is being used.
Otherwise, it contains only the following members:

- `rank` (set to ``0``);
- `size` (set to ``1``).

Thus, the following code works regardless whether MPI is present or
not::

  import litebird_sim as lbs

  if lbs.MPI_COMM_WORLD.rank == 0:
      print("Hello, world!")

However, you can use :data:`.MPI_COMM_WORLD` to call MPI functions
only if MPI was actually enabled. You can check this using the Boolean
variable :data:`.MPI_ENABLED`::

  import litebird_sim as lbs

  comm = lbs.MPI_COMM_WORLD
  if lbs.MPI_ENABLED:
      comm.barrier()


To ensure that your code uses MPI in the proper way, you should always
use ``.MPI_COMM_WORLD`` instead of importing ``mpi4py`` directly.


Enabling/disabling MPI
----------------------

The user can control whether MPI must be used or not in a script,
through the environment variable ``LITEBIRD_SIM_PI`` (``ENABLE_MPI``
is accepted as well):

- If the variable is set to the empty string or to ``1``, ``true``,
  ``on``, ``yes``, then ``mpi4py`` is imported, and an exception is
  raised if this cannot be done (e.g., because it was not installed
  using the flag ``--extra=mpi`` when ``poetry install`` was called).

- If the variable is set to ``0``, ``false``, ``off`` or ``no``, then
  ``mpi4py`` is *not* imported, even if it is installed.

- If the variable is not set, then ``mpi4py`` will be imported, but
  any failure will be accepted and the framework will silently switch
  to serial mode.


API reference
-------------

.. automodule:: litebird_sim.mpi
    :members:
    :undoc-members:
    :show-inheritance:
