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


#: Global variable equal either to `mpi4py.MPI.COMM_WORLD` or a object
#: that defines the member variables `rank = 0` and `size = 1`.
MPI_COMM_WORLD = _SerialMpiCommunicator()

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
        MPI_ENABLED = True
        MPI_CONFIGURATION = mpi4py.get_config()
    except ImportError:
        if _enable_mpi:
            raise  # If MPI was explicitly requested, re-raise the exception
        else:
            pass  # Ignore the error
