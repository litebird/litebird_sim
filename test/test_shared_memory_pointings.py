import numpy as np

import litebird_sim as lbs
from litebird_sim.scanning import RotQuaternion, SharedRotQuaternion
import pytest

pytest.importorskip(
    modname="mpi4py",
    reason="`mpi4py` is required to run MPI shared memory tests",
)
from mpi4py import MPI  # noqa: E402


def test_shared_scanning_strategy(tmp_path):

    # Create two simulations
    sim_std = lbs.Simulation(
        base_path=tmp_path / "simulation_std",
        start_time=0.0,
        duration_s=10.0,
        random_seed=12345,
        mpi_comm=MPI.COMM_WORLD,
    )

    sim_shared = lbs.Simulation(
        base_path=tmp_path / "simulation_shared",
        start_time=0.0,
        duration_s=10.0,
        random_seed=12345,
        mpi_comm=MPI.COMM_WORLD,
    )

    # dummy scanning strategy class
    class DummyScanningStrategy(lbs.ScanningStrategy):
        def generate_spin2ecl_quaternions(self, start_time, time_span_s, delta_time_s):
            n_samples = int(np.ceil(time_span_s / delta_time_s))
            quats = np.random.randn(n_samples, 4)
            quats /= np.linalg.norm(quats, axis=1)[:, np.newaxis]
            return RotQuaternion(
                quats, start_time=start_time, sampling_rate_hz=1.0 / delta_time_s
            )

    scanning_strategy = DummyScanningStrategy()

    # We set a random seed here so the two instances generate the same quaternions
    np.random.seed(123)
    sim_std.set_scanning_strategy(
        scanning_strategy=scanning_strategy, delta_time_s=1.0, append_to_report=False
    )

    np.random.seed(123)
    sim_shared.set_scanning_strategy_shmem(
        scanning_strategy=scanning_strategy, delta_time_s=1.0, append_to_report=False
    )

    q_std = sim_std.spin2ecliptic_quats.quats
    q_shared = sim_shared.spin2ecliptic_quats.quats

    np.testing.assert_allclose(q_std, q_shared)
    assert isinstance(sim_shared.spin2ecliptic_quats, SharedRotQuaternion)


def test_shmem_bore2ecliptic_quats(tmp_path):
    comm_size = MPI.COMM_WORLD.size

    # Create two simulations
    sim_std = lbs.Simulation(
        base_path=tmp_path / "simulation_std",
        start_time=0.0,
        duration_s=10.0,
        random_seed=12345,
        mpi_comm=MPI.COMM_WORLD,
    )

    sim_shared = lbs.Simulation(
        base_path=tmp_path / "simulation_shared",
        start_time=0.0,
        duration_s=10.0,
        random_seed=12345,
        mpi_comm=MPI.COMM_WORLD,
    )

    # Add an instrument
    instrument = lbs.InstrumentInfo(
        name="test_inst",
        spin_boresight_angle_rad=np.deg2rad(50.0),
    )
    sim_std.set_instrument(instrument)
    sim_shared.set_instrument(instrument)

    # Create observations (this handles comm_time_block and comm_det_block properly)
    det1 = lbs.DetectorInfo("det1", sampling_rate_hz=10.0)
    det2 = lbs.DetectorInfo("det2", sampling_rate_hz=10.0)

    # n_blocks_det * n_blocks_time must equal comm.size
    sim_std.create_observations(
        detectors=[det1, det2],
        n_blocks_time=comm_size // 2 if comm_size % 2 == 0 else comm_size,
        n_blocks_det=2 if comm_size % 2 == 0 else 1,
        split_list_over_processes=False,
    )
    sim_shared.create_observations(
        detectors=[det1, det2],
        n_blocks_time=comm_size // 2 if comm_size % 2 == 0 else comm_size,
        n_blocks_det=2 if comm_size % 2 == 0 else 1,
        split_list_over_processes=False,
    )

    # Fake spin2ecliptic quaternions
    n_samples = sim_std.observations[0].n_samples_global
    if sim_std.mpi_comm.rank == 0:
        quats = np.random.randn(n_samples, 4)
        quats /= np.linalg.norm(quats, axis=1)[:, np.newaxis]
    else:
        quats = np.empty((n_samples, 4), dtype=np.float64)
    sim_std.mpi_comm.Bcast(quats, root=0)
    spin2ecliptic_quats = RotQuaternion(quats, start_time=0.0, sampling_rate_hz=10.0)

    # Standard pointings preparation
    sim_std.observations[0].prepare_pointings(
        instrument=sim_std.instrument, spin2ecliptic_quats=spin2ecliptic_quats
    )

    # Shared memory pointings preparation
    sim_shared.observations[0].prepare_pointings_shmem(
        instrument=sim_shared.instrument, spin2ecliptic_quats=spin2ecliptic_quats
    )

    # Check if the outputs are identical
    q_std = sim_std.observations[0].pointing_provider.bore2ecliptic_quats.quats
    q_shared = sim_shared.observations[0].pointing_provider.bore2ecliptic_quats.quats

    np.testing.assert_allclose(q_std, q_shared)

    # Ensure that it is indeed a SharedRotQuaternion
    assert isinstance(
        sim_shared.observations[0].pointing_provider.bore2ecliptic_quats,
        SharedRotQuaternion,
    )
