# -*- encoding: utf-8 -*-

try:
    import mpi4py
    from mpi4py import MPI

    #: Set to True if MPI is available through mpi4py
    HAVE_MPI4PY = True

    #: If MPI is available, this is the rank of the current process,
    #: otherwise it's zero
    MPI_RANK = MPI.COMM_WORLD.rank

    #: If MPI is available, this is the number of MPI processes
    #: currently running, otherwise it's 1.
    MPI_SIZE = MPI.COMM_WORLD.size

    #: Either a dictionary containing the mpi4py configuration, or
    #: ``None`` if mpi4py is not available
    MPI_CONFIGURATION = mpi4py.get_config()

    #: Either the constant ``mpi4py.MPI.COMM_WORLD``, or ``None``
    #: if mpi4py is not available
    MPI_COMM_WORLD = MPI.COMM_WORLD

except ModuleNotFoundError:
    #: Set to True if MPI is available through mpi4py
    HAVE_MPI4PY = False

    #: If MPI is available, this is the rank of the current process,
    #: otherwise it's zero
    MPI_RANK = 0

    #: If MPI is available, this is the number of MPI processes
    #: currently running, otherwise it's 1.
    MPI_SIZE = 1

    #: Either a dictionary containing the mpi4py configuration, or
    #: ``None`` if mpi4py is not available
    MPI_CONFIGURATION = None

    #: Either the constant ``mpi4py.MPI.COMM_WORLD``, or ``None``
    #: if mpi4py is not available
    MPI_COMM_WORLD = None
