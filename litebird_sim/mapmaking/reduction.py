"""Reductions of the map-makers over the MPI processes.

A map-maker accumulates the contributions of its local observations and then
sums them over the processes that hold the other observations of the same
list. Which processes these are depends on how the observations were
distributed (see :func:`reduction_for_observations`); this module is the only
place that decides it, so that every map-maker sums over the same group.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from litebird_sim.observations import Observation


class SerialReduction:
    """Reduction for a serial run, or for observations local to one process.

    Every operation leaves its arguments unchanged."""

    comm = None
    rank = 0
    size = 1

    def sum_in_place(self, *arrays: np.ndarray) -> None:
        """Sum each array over the processes, in place"""

    def sum(self, value: Any) -> Any:
        """Return the sum of a scalar over the processes"""
        return value

    def max(self, value: Any) -> Any:
        """Return the maximum of a scalar over the processes"""
        return value


class MpiReduction:
    """Reduction over an MPI communicator."""

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size

    def sum_in_place(self, *arrays: np.ndarray) -> None:
        """Sum each array over the processes, in place"""
        from litebird_sim.mpi import MPI

        for array in arrays:
            self.comm.Allreduce(MPI.IN_PLACE, array, op=MPI.SUM)

    def sum(self, value: Any) -> Any:
        """Return the sum of a scalar over the processes"""
        from litebird_sim.mpi import MPI

        return self.comm.allreduce(value, op=MPI.SUM)

    def max(self, value: Any) -> Any:
        """Return the maximum of a scalar over the processes"""
        from litebird_sim.mpi import MPI

        return self.comm.allreduce(value, op=MPI.MAX)


Reduction = SerialReduction | MpiReduction


def reduction_for_observations(obs_list: list[Observation]) -> Reduction:
    """Return the reduction over which a map-maker must sum `obs_list`.

    Observations distributed over the processes (``obs.comm``) are summed over
    their communicator; whole observations spread over the processes by
    :meth:`.Simulation.create_observations` with
    ``split_list_over_processes=True`` (``obs.comm`` is None) are summed over
    the communicator of the list (``obs.list_comm``). Without MPI, or when no
    observation has a communicator, the reduction is serial.

    Raise ``NotImplementedError`` if the observations are distributed over
    different groups of processes.
    """
    from litebird_sim import mpi

    if not mpi.MPI_ENABLED:
        return SerialReduction()

    comms = [
        obs.list_comm if getattr(obs, "list_comm", None) is not None else obs.comm
        for obs in obs_list
    ]
    if all(c is None for c in comms):
        return SerialReduction()

    valid = [c for c in comms if c is not None]
    if len(valid) < len(comms) or not all(
        mpi.MPI.Comm.Compare(valid[i], valid[i + 1]) < 2 for i in range(len(valid) - 1)
    ):
        raise NotImplementedError(
            "All observations must be distributed over the same MPI groups"
        )
    return MpiReduction(valid[0])
