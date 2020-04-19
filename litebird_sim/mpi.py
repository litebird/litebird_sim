# -*- encoding: utf-8 -*-

try:
    from mpi4py import MPI

    HAVE_MPI4PY = True
    MPI_RANK = MPI.COMM_WORLD.rank
    MPI_SIZE = MPI.COMM_WORLD.size

except ModuleNotFoundError:
    HAVE_MPI4PY = False
    MPI_RANK = 0
    MPI_SIZE = 1
