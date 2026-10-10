import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim.scanning import SharedRotQuaternion

pytest.importorskip(
    modname="mpi4py",
    reason="`mpi4py` is required to run MPI shared memory tests",
)
from mpi4py import MPI  # noqa: E402


def _make_simulation(tmp_path, name):
    sim = lbs.Simulation(
        base_path=tmp_path / name,
        start_time=0.0,
        duration_s=100.0,
        random_seed=12345,
        mpi_comm=MPI.COMM_WORLD,
    )
    sim.set_instrument(
        lbs.InstrumentInfo(name="test_inst", spin_boresight_angle_rad=np.deg2rad(50.0))
    )
    return sim


def _set_scanning_strategy(sim, shared_memory):
    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            spin_rate_hz=1.0 / 60.0,
            precession_rate_hz=1.0 / 600.0,
        ),
        delta_time_s=1.0,
        append_to_report=False,
        shared_memory=shared_memory,
    )


def _check_read_only_on_non_root(sim, quats):
    manager = sim._shared_memory_manager
    if manager.node_rank != manager.node_root:
        assert not quats.flags.writeable


def test_shared_scanning_strategy(tmp_path):
    sim_std = _make_simulation(tmp_path, "simulation_std")
    sim_shared = _make_simulation(tmp_path, "simulation_shared")

    _set_scanning_strategy(sim_std, shared_memory=False)
    _set_scanning_strategy(sim_shared, shared_memory=True)

    assert isinstance(sim_shared.spin2ecliptic_quats, SharedRotQuaternion)
    np.testing.assert_array_equal(
        sim_std.spin2ecliptic_quats.quats, sim_shared.spin2ecliptic_quats.quats
    )
    _check_read_only_on_non_root(sim_shared, sim_shared.spin2ecliptic_quats.quats)


@pytest.mark.parametrize("split_list_over_processes", [True, False])
def test_shared_prepare_pointings(tmp_path, split_list_over_processes):
    comm_size = MPI.COMM_WORLD.size
    sims = []
    for name, shared_memory in [("std", False), ("shared", True)]:
        sim = _make_simulation(tmp_path, f"simulation_{name}")
        _set_scanning_strategy(sim, shared_memory=shared_memory)
        sim.create_observations(
            detectors=[
                lbs.DetectorInfo("det1", sampling_rate_hz=10.0),
                lbs.DetectorInfo("det2", sampling_rate_hz=10.0),
            ],
            # With `split_list_over_processes=True`, every process must get
            # at least one observation; otherwise, the observations are split
            # in time blocks among the processes
            num_of_obs_per_detector=2 * comm_size if split_list_over_processes else 2,
            n_blocks_time=1 if split_list_over_processes else comm_size,
            split_list_over_processes=split_list_over_processes,
        )
        sim.prepare_pointings(append_to_report=False, shared_memory=shared_memory)
        sims.append(sim)

    sim_std, sim_shared = sims

    # All the observations share the same copy of the boresight quaternions
    bore2ecl = sim_shared.observations[0].pointing_provider.bore2ecliptic_quats
    assert isinstance(bore2ecl, SharedRotQuaternion)
    for cur_obs in sim_shared.observations:
        assert cur_obs.pointing_provider.bore2ecliptic_quats is bore2ecl
    _check_read_only_on_non_root(sim_shared, bore2ecl.quats)

    # Shared memory must not change the pointings
    for obs_std, obs_shared in zip(sim_std.observations, sim_shared.observations):
        pointings_std, _ = obs_std.get_pointings()
        pointings_shared, _ = obs_shared.get_pointings()
        np.testing.assert_array_equal(pointings_std, pointings_shared)


def test_shared_memory_windows_freed_at_exit(tmp_path):
    # MPI windows must be freed before MPI_Finalize, otherwise some MPI
    # implementations (e.g., Intel MPI) hang when the interpreter exits
    sim = _make_simulation(tmp_path, "simulation_exit")
    _set_scanning_strategy(sim, shared_memory=True)

    manager = sim._shared_memory_manager
    assert manager.list_windows

    manager._free_at_exit()
    assert not manager.list_windows
