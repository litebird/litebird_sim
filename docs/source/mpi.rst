.. _using_mpi:

Multithreading and MPI
======================

As typical operations on time-ordered data can be quite consuming,
the LiteBIRD Simulation Framework provides a number of tools to
exploit the presence of multiple CPU cores and even multiple
computing nodes. This section details how to take advantage
of these facilities and is split in two parts:

- We will first present the ability of the framework to use multiple
  CPU threads; in this context, the data samples are kept in a chunk
  of memory that is shared between several processes. The framework
  uses Numba, which can take advantage either of the `Intel
  Threading Building Blocks <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html>`_
  library or of `OpenMP <https://www.openmp.org/>`_.

- Then, we will discuss the possibility to run the code on
  multiple *computing nodes*, where the memory of each node is
  **not** shared with the others. The framework is able to
  use any MPI library, through the Python package
  `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_.


Multithreading
~~~~~~~~~~~~~~

Some parts of the LiteBIRD Simulation Framework are able to
exploit multiple cores because several of its modules rely on
the `Numba <https://numba.pydata.org/>`_ library.

If you are running your code on your multi-core laptop, you do not
have to do anything fancy in order to use all the CPUs on your machine:
in its default configuration, the Framework should be able to take
advantage all the available CPU cores.

However, if you want to tune the way the Framework uses the CPUs,
you can either set the environment variable ``OMP_NUM_THREADS``
to the number of CPUs to use, or use two parameters in
the constructor of the class :class:`.Simulation`:

- `numba_num_of_threads`: this is the number of CPUs that Numba will
  use for parallel calculations. The parameter defaults to ``None``,
  which means that Numba will check how many CPUs are available and will
  use all of them.

- `numba_threading_layer`: this parameter is a string that specifies
  which threading library should be used by Numba. The value depends
  both on the version of Numba you are running and on the availability
  of these libraries, as they are not installed together with the
  LiteBIRD Simulation Framework. Numba 0.53 provides the following choices:

  - ``tbb``: `Intel Threading Building Blocks
    <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html>`_.
    You should pick this if you are running your code on Intel machines and the
    Tbb library is available.

  - ``omp``: `OpenMP <https://www.openmp.org/>`_. If you pick this one,
    be sure that the OpenPM library is available.

  - ``workqueue``: this is an internal threading library provided by Numba.
    It's probably the least efficient of the three; its main advantage is
    that it is always available.

These parameters can be passed through a TOML parameter file (see
:ref:`parameter_files`) as well:

.. code-block:: toml

   # This is file "my_conf.toml"
   [simulation]
   random_seed = 12345
   numba_num_of_threads = 32
   numba_threading_layer = "tbb"

Both ``tbb`` and ``omp`` require that the relevant library be available on
your system, as the command ``pip install litebird_sim`` does **not** install
them. If you are running your code on a HPC cluster, it is probably a matter
of running a command like the following:

.. code-block:: sh

   # This might change depending on how the environment on your cluster
   # is configured; the following commands are just examples.
   $ module load tbb     # Intel Threading Building Blocks
   $ module load openmp  # OpenMP


MPI
~~~

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
Otherwise, if MPI is **not** being used, it still contains the
following members:

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
use :data:`.MPI_COMM_WORLD` instead of importing ``mpi4py`` directly.


Enabling/disabling MPI
----------------------

The user can control whether MPI must be used or not in a script,
through the environment variable ``LITEBIRD_SIM_MPI`` (``ENABLE_MPI``
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


Grasping how MPI is being used
------------------------------

You will typically use MPI to spread TODs among many MPI processes, so
that the simulation can span several detectors and a longer time scale.
Unfortunately, this means that it's often complicated to understand how
data is being kept in memory.

If you use the :class:`.Simulation` object (and you should, you *really*
should!), you can call :meth:`.Simulation.describe_mpi_distribution` after
you have allocated the TODs via :meth:`.Simulation.create_observations`; it
will return an instance of the class :class:`.MpiDistributionDescr`, which
can be inspected and printed to the terminal. See Section :ref:`simulations`
for more information about this.


API reference
-------------

.. automodule:: litebird_sim.mpi
    :members:
    :undoc-members:
    :show-inheritance:
