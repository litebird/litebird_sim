# -*- encoding: utf-8 -*-

import litebird_sim as lbs
import numpy as np


def test_distribution():
    return #XXX
    comm_world = lbs.MPI_COMM_WORLD

    if comm_world.rank == 0:
        print(f"MPI configuration: {lbs.MPI_CONFIGURATION}")

    sim = lbs.Simulation(mpi_comm=comm_world)
    det1 = lbs.Detector(
        name="pol01",
        wafer="mywafer",
        pixel=1,
        pixtype="A",
        channel=30,
        sampling_rate_hz=5,
        fwhm_arcmin=30,
        ellipticity=1.0,
        net_ukrts=1.0,
        fknee_mhz=10,
        fmin_hz=1e-6,
        alpha=1.0,
        pol="Q",
        orient="A",
        quat=[0, 0, 0, 0],
    )
    det2 = lbs.Detector(
        name="pol02",
        wafer="mywafer",
        pixel=2,
        pixtype="B",
        channel=44,
        sampling_rate_hz=50,
        fwhm_arcmin=30,
        ellipticity=1.0,
        net_ukrts=1.0,
        fknee_mhz=10,
        fmin_hz=1e-6,
        alpha=1.0,
        pol="Q",
        orient="A",
        quat=[0, 0, 0, 0],
    )
    sim.create_observations(
        [det1, det2],
        num_of_obs_per_detector=10,
        start_time=0.0,
        duration_s=86400.0 * 10,
    )

    assert (
        sim.observations
    ), "No observations have been defined for process {lbs.MPI_RANK + 1}/{lbs.MPI_SIZE}"

    if comm_world.size > 1:
        allobs = comm_world.allreduce([x.detector.name for x in sim.observations])
    else:
        allobs = sim.observations

    assert len(allobs) == 20


def test_observation_tod_single_block():
    comm_world = lbs.MPI_COMM_WORLD
    obs = lbs.Observation(
        n_detectors = 3,
        n_samples=9,
        start_time=0.0,
        sampling_rate_hz=1.0,
        comm=comm_world,
    )

    if comm_world.rank == 0:
        assert obs.tod.shape == (3, 9)
        assert obs.tod.dtype == np.float32
    else:
        assert obs.tod.shape == (0, 0)


def test_observation_tod_two_block_time():
    comm_world = lbs.MPI_COMM_WORLD
    try:
        obs = lbs.Observation(
            n_detectors = 3,
            n_samples=9,
            start_time=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=2,
            comm=comm_world,
        )
    except ValueError:
        # Not enough processes to split the TOD, constuctor expected to rise
        if comm_world.size < 2:
            return

    if comm_world.rank == 0:
        assert obs.tod.shape == (3, 5)
    elif comm_world.rank == 1:
        assert obs.tod.shape == (3, 4)
    else:
        assert obs.tod.shape == (0, 0)


def test_observation_tod_two_block_det():
    comm_world = lbs.MPI_COMM_WORLD
    try:
        obs = lbs.Observation(
            n_detectors = 3,
            n_samples=9,
            start_time=0.0,
            sampling_rate_hz=1.0,
            n_blocks_det=2,
            comm=comm_world,
        )
    except ValueError:
        # Not enough processes to split the TOD, constuctor expected to rise
        if comm_world.size < 2:
            return

    if comm_world.rank == 0:
        assert obs.tod.shape == (2, 9)
    elif comm_world.rank == 1:
        assert obs.tod.shape == (1, 9)
    else:
        assert obs.tod.shape == (0, 0)


def test_observation_tod_set_blocks():
    comm_world = lbs.MPI_COMM_WORLD
    try:
        obs = lbs.Observation(
            n_detectors=3,
            n_samples=9,
            start_time=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=2,
            comm=comm_world,
        )
    except ValueError:
        # Not enough processes to split the TOD, constuctor expected to rise
        if comm_world.size < 2:
            return
    
    # Two time blocks
    ref_tod = np.arange(27, dtype=np.float32).reshape(3, 9)
    if comm_world.rank == 0:
        obs.tod[:] = ref_tod[:, :5]
    elif comm_world.rank == 1:
        obs.tod[:] = ref_tod[:, 5:]

    # Two detector blocks
    obs.set_n_blocks(n_blocks_time=1, n_blocks_det=2)
    if comm_world.rank == 0:
        assert np.all(obs.tod == ref_tod[:2])
    elif comm_world.rank == 1:
        assert np.all(obs.tod == ref_tod[2:])
    else:
        assert obs.tod.size == 0

    # One block
    obs.set_n_blocks(n_blocks_det=1, n_blocks_time=1)
    if comm_world.rank == 0:
        assert np.all(obs.tod == ref_tod)
    else:
        assert obs.tod.size == 0

    # Three time blocks
    if comm_world.size < 3:
        return
    obs.set_n_blocks(n_blocks_det=1, n_blocks_time=3)
    if comm_world.rank == 0:
        assert np.all(obs.tod == ref_tod[:, :3])
    elif comm_world.rank == 1:
        assert np.all(obs.tod == ref_tod[:, 3:6])
    elif comm_world.rank == 2:
        assert np.all(obs.tod == ref_tod[:, 6:])
    else:
        assert obs.tod.size == 0

    # Two detector blocks and two time blocks
    if comm_world.size < 4:
        return
    obs.set_n_blocks(n_blocks_time=2, n_blocks_det=2)
    if comm_world.rank == 0:
        assert np.all(obs.tod == ref_tod[:2, :5])
    elif comm_world.rank == 1:
        assert np.all(obs.tod == ref_tod[:2, 5:])
    elif comm_world.rank == 2:
        assert np.all(obs.tod == ref_tod[2:, :5])
    elif comm_world.rank == 3:
        assert np.all(obs.tod == ref_tod[2:, 5:])
    else:
        assert obs.tod.size == 0

    try:
        obs.set_n_blocks(n_blocks_det=4, n_blocks_time=1)
    except ValueError:
        pass
    else:
        raise Exception("ValueError expected")

    # Two detector blocks and three time blocks
    if comm_world.size < 6:
        return
    obs.set_n_blocks(n_blocks_det=2, n_blocks_time=3)
    if comm_world.rank == 0:
        assert np.all(obs.tod == ref_tod[:2, :3])
    elif comm_world.rank == 1:
        assert np.all(obs.tod == ref_tod[:2, 3:6])
    elif comm_world.rank == 2:
        assert np.all(obs.tod == ref_tod[:2, 6:])
    elif comm_world.rank == 3:
        assert np.all(obs.tod == ref_tod[2:, :3])
    elif comm_world.rank == 4:
        assert np.all(obs.tod == ref_tod[2:, 3:6])
    elif comm_world.rank == 5:
        assert np.all(obs.tod == ref_tod[2:, 6:])
    else:
        assert obs.tod.size == 0

    # Three detector blocks and three time blocks
    if comm_world.size < 9:
        return
    obs.set_n_blocks(n_blocks_det=3, n_blocks_time=3)
    if comm_world.rank == 0:
        assert np.all(obs.tod == ref_tod[:1, :3])
    elif comm_world.rank == 1:
        assert np.all(obs.tod == ref_tod[:1, 3:6])
    elif comm_world.rank == 2:
        assert np.all(obs.tod == ref_tod[:1, 6:])
    elif comm_world.rank == 3:
        assert np.all(obs.tod == ref_tod[1:2, :3])
    elif comm_world.rank == 4:
        assert np.all(obs.tod == ref_tod[1:2, 3:6])
    elif comm_world.rank == 5:
        assert np.all(obs.tod == ref_tod[1:2, 6:])
    elif comm_world.rank == 6:
        assert np.all(obs.tod == ref_tod[2:, :3])
    elif comm_world.rank == 7:
        assert np.all(obs.tod == ref_tod[2:, 3:6])
    elif comm_world.rank == 8:
        assert np.all(obs.tod == ref_tod[2:, 6:])
    else:
        assert obs.tod.size == 0


def main():
    #test_distribution()
    test_observation_tod_single_block()
    test_observation_tod_two_block_time()
    test_observation_tod_two_block_det()
    test_observation_tod_set_blocks()

main()
