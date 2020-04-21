.. _using_mpi:

Using MPI
=========

The LiteBIRD Simulation Framework lists mpi4py as an *optional*
dependency. This means that simulation codes should be able to cope
with the lack of MPI.

The framework provides a global variable, ``MPI_COMM_WORLD``, which
behaves like ``mpi4py.MPI.COMM_WORLD`` but is defined even if MPI is
not present. It has the following features:

- The member variables ``rank`` and ``size`` are always defined, even
  in absence of ``mpi4py`` (in this case they are set to 0 and 1,
  respectively);
- A member variable ``have_mpi`` is set to ``True`` if ``mpi4py`` was
  imported;
- Importing ``mpi4py`` can be controlled through the environment
  variable ``LITEBIRD_SIM_MPI``.

Thus, the following code will always work, regardless of the fact that
``mpi4py`` was found and imported or not::

  import litebird_sim as lbs

  if lbs.MPI_COMM_WORLD.rank == 0:
      print("Hello, world!")

However, you can call MPI functions only if
``MPI.COMM_WORLD.have_mpi`` is ``True``:

  import litebird_sim as lbs

  comm = lbs.MPI_COMM_WORLD
  if lbs.MPI_COMM_WORLD.have_mpi:
      # The MPI call to "barrier" is broadcasted to mpi4py
      comm.barrier()



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
