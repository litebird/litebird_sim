# -*- encoding: utf-8 -*-

try:
    import mpi4py
    from mpi4py import MPI

    HAVE_MPI4PY = True
    MPI_RANK = MPI.COMM_WORLD.rank
    MPI_SIZE = MPI.COMM_WORLD.size
    MPI_CONFIGURATION = mpi4py.get_config()
    MPI_COMM_WORLD = MPI.COMM_WORLD

except ModuleNotFoundError:
    HAVE_MPI4PY = False
    MPI_RANK = 0
    MPI_SIZE = 1
    MPI_CONFIGURATION = None
    MPI_COMM_WORLD = None
