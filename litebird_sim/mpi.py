# -*- encoding: utf-8 -*-

import os


def _check_if_enable_mpi():
    for keyword in ["LITEBIRD_SIM_MPI", "ENABLE_MPI"]:
        if keyword not in os.environ:
            continue

        value = os.environ[keyword]
        if value.lower() in ["1", "yes", "true", "on", ""]:
            return True
        if value.lower() in ["0", "no", "false", "off"]:
            return False

    return None


class _SerialMpiCommunicator:
    rank = 0
    size = 1


class _GridCommClass:
    """
    This class encapsulates the `COMM_OBS_GRID` and `COMM_NULL` communicators. It
    offers explicitly defined setter functions so that the communicators cannot be
    changed accidentally.

    Attributes:

        COMM_OBS_GRID (mpi4py.MPI.Intracomm): A subset of `MPI.COMM_WORLD` that
            contain all the processes associated with non-zero observations.

        COMM_NULL (mpi4py.MPI.Comm): A NULL communicator. When MPI is not enabled, it
            is set as `None`. If MPI is enabled, it is set as `MPI.COMM_NULL`

    """

    def __init__(self, comm_obs_grid=_SerialMpiCommunicator(), comm_null=None):
        self._MPI_COMM_OBS_GRID = comm_obs_grid
        self._MPI_COMM_NULL = comm_null

    @property
    def COMM_OBS_GRID(self):
        return self._MPI_COMM_OBS_GRID

    @property
    def COMM_NULL(self):
        return self._MPI_COMM_NULL

    def _set_comm_obs_grid(self, comm_obs_grid):
        self._MPI_COMM_OBS_GRID = comm_obs_grid

    def _set_null_comm(self, comm_null):
        self._MPI_COMM_NULL = comm_null


#: Global variable equal either to `mpi4py.MPI.COMM_WORLD` or a object
#: that defines the member variables `rank = 0` and `size = 1`.
MPI_COMM_WORLD = _SerialMpiCommunicator()


#: Global object with two attributes:
#:
#: - ``COMM_OBS_GRID``: It is a partition of ``MPI_COMM_WORLD`` that includes all the
#:   MPI processes with global rank less than ``n_blocks_time * n_blocks_det``. On MPI
#:   processes with higher ranks, it points to NULL MPI communicator
#:   ``mpi4py.MPI.COMM_NULL``.
#:
#: - ``COMM_NULL``: If :data:`.MPI_ENABLED` is ``True``, this object points to a NULL
#:   MPI communicator (``mpi4py.MPI.COMM_NULL``). Otherwise it is ``None``.
MPI_COMM_GRID = _GridCommClass()

#: `True` if MPI should be used by the application. The value of this
#: variable is set according to the following rules:
#:
#:
#: - If the environment variable :data:`.LITEBIRD_SIM_MPI` is set to
#:   `1`, use MPI and fail if `mpi4py` cannot be imported;
#:
#: - If the environment variable :data:`.LITEBIRD_SIM_MPI` is set to
#:   `0`, avoid using MPI even if `mpi4py` is present;
#:
#: - If the environment variable :data:`.LITEBIRD_SIM_MPI` is *not* set,
#:   try to use MPI and gracefully revert to a serial mode of execution
#:   if `mpi4py` cannot be imported.
MPI_ENABLED = False

#: If :data:`.MPI_ENABLED` is `True`, this is a dictionary containing
#: information about the MPI configuration. Otherwise, it is an empty
#: dictionary
MPI_CONFIGURATION = {}

_enable_mpi = _check_if_enable_mpi()
if _enable_mpi in [True, None]:
    try:
        import mpi4py
        from mpi4py import MPI

        MPI_COMM_WORLD = MPI.COMM_WORLD
        MPI_COMM_GRID._set_comm_obs_grid(comm_obs_grid=MPI.COMM_WORLD)
        MPI_COMM_GRID._set_null_comm(comm_null=MPI.COMM_NULL)
        MPI_ENABLED = True
        MPI_CONFIGURATION = mpi4py.get_config()
    except ImportError:
        if _enable_mpi:
            raise  # If MPI was explicitly requested, re-raise the exception
        else:
            pass  # Ignore the error
