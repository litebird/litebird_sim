from types import SimpleNamespace

import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim import mpi
from litebird_sim.mapmaking.reduction import (
    MpiReduction,
    SerialReduction,
    reduction_for_observations,
)


def test_serial_reduction_is_identity():
    red = SerialReduction()
    a = np.arange(4.0)
    b = np.ones((2, 3))
    red.sum_in_place(a, b)
    np.testing.assert_array_equal(a, np.arange(4.0))
    np.testing.assert_array_equal(b, np.ones((2, 3)))
    assert red.sum(3.5) == 3.5
    assert red.max(-2) == -2
    assert (red.rank, red.size, red.comm) == (0, 1, None)


def test_reduction_without_communicators_is_serial():
    obs = [SimpleNamespace(comm=None, list_comm=None) for _ in range(3)]
    assert isinstance(reduction_for_observations(obs), SerialReduction)


def test_reduction_of_simulation_observations():
    sim = lbs.Simulation(start_time=0.0, duration_s=10.0, random_seed=1)
    sim.create_observations(
        detectors=[lbs.DetectorInfo(name="d", sampling_rate_hz=1.0)]
    )
    red = reduction_for_observations(sim.observations)

    assert isinstance(red, (SerialReduction, MpiReduction))
    assert red.size == mpi.MPI_COMM_WORLD.size
    assert red.sum(1) == mpi.MPI_COMM_WORLD.size


@pytest.mark.skipif(not mpi.MPI_ENABLED, reason="needs MPI")
def test_reduction_rejects_mixed_distributions():
    obs = [
        SimpleNamespace(comm=mpi.MPI_COMM_WORLD, list_comm=None),
        SimpleNamespace(comm=None, list_comm=None),
    ]
    with pytest.raises(NotImplementedError):
        reduction_for_observations(obs)
