# -*- encoding: utf-8 -*-
# NOTE: all the following tests should be valid also in a serial execution
from pathlib import Path
from tempfile import TemporaryDirectory

import astropy.time as astrotime
import pytest

import litebird_sim as lbs
import numpy as np
from litebird_sim import MPI_COMM_WORLD


@pytest.fixture
def mpi_tmp_path():
    if not lbs.MPI_ENABLED:
        with TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
        return

    # It's critical that all MPI processes use the same output directory
    comm = lbs.MPI_COMM_WORLD
    if comm.rank == 0:
        tmp_dir = TemporaryDirectory()
        tmp_path = tmp_dir.name
        comm.bcast(tmp_path, root=0)
    else:
        tmp_dir = None
        tmp_path = comm.bcast(None, root=0)

    # https://docs.pytest.org/en/stable/how-to/fixtures.html#yield-fixtures-recommended
    yield Path(tmp_path)

    comm.barrier()
    if tmp_dir:
        tmp_dir.cleanup()


def test_observation_time():
    comm_world = lbs.MPI_COMM_WORLD
    if comm_world.size > 2:
        pytest.skip("This test is only supported with 1 or 2 MPI processes")

    ref_time = astrotime.Time("2020-02-20", format="iso")

    obs_no_mjd = lbs.Observation(
        detectors=1,
        start_time_observation=0.0,
        sampling_rate_hz=5.0,
        n_samples_observation=5,
        n_blocks_time=comm_world.size,
        comm=comm_world,
    )
    obs_mjd_astropy = lbs.Observation(
        detectors=1,
        start_time_observation=ref_time,
        sampling_rate_hz=5.0,
        n_samples_observation=5,
        n_blocks_time=comm_world.size,
        comm=comm_world,
    )

    res_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    res_mjd = np.array(
        [0.0, 2.314_816_81e-06, 4.629_626_35e-06, 6.944_443_16e-06, 9.259_259_97e-06]
    )
    res_cxcsec = np.array(
        [6.985_440_69e8, 6.985_440_69e8, 6.985_440_70e8, 6.985_440_70e8, 6.985_440_70e8]
    )

    if not comm_world or comm_world.size == 1:
        assert np.allclose(obs_no_mjd.get_times(), res_times)
        assert np.allclose(
            (obs_mjd_astropy.get_times(astropy_times=True) - ref_time).jd, res_mjd
        )
        assert np.allclose(
            obs_mjd_astropy.get_times(normalize=False, astropy_times=False), res_cxcsec
        )
    elif comm_world.size == 2:
        if comm_world.rank == 0:
            assert np.allclose(obs_no_mjd.get_times(), res_times[:3])
            assert np.allclose(
                (obs_mjd_astropy.get_times(astropy_times=True) - ref_time).jd,
                res_mjd[:3],
            )
            assert np.allclose(
                obs_mjd_astropy.get_times(normalize=False, astropy_times=False),
                res_cxcsec[:3],
            )
        elif comm_world.rank == 1:
            assert np.allclose(obs_no_mjd.get_times(), res_times[3:])
            assert np.allclose(
                (obs_mjd_astropy.get_times(astropy_times=True) - ref_time).jd,
                res_mjd[3:],
            )
            assert np.allclose(
                obs_mjd_astropy.get_times(normalize=False, astropy_times=False),
                res_cxcsec[3:],
            )


def test_construction_from_detectors():
    comm_world = lbs.MPI_COMM_WORLD
    if comm_world.size % 2 != 0:
        pytest.skip("This test requires the number of MPI processes to be even")

    det1 = dict(
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
        quat=[0.0, 0.0, 0.0, 0.0],
    )
    det2 = dict(
        name="pol02",
        wafer="mywafer",
        pixel=2,
        # pixtype="B",
        channel=44,
        sampling_rate_hz=50,
        fwhm_arcmin=30,
        ellipticity=2.0,
        net_ukrts=1.0,
        fknee_mhz=10,
        fmin_hz=1e-6,
        # alpha=1.0,
        pol="Q",
        orient="A",
        quat=[1.0, 1.0, 1.0, 1.0],
    )

    obs = lbs.Observation(
        detectors=[det1, det2],
        n_samples_observation=100,
        start_time_observation=0.0,
        sampling_rate_hz=1.0,
        n_blocks_time=comm_world.size,
        comm=comm_world,
        root=0,
    )

    assert obs.name[0] == "pol01"
    assert obs.name[1] == "pol02"
    assert obs.wafer[0] == "mywafer"
    assert obs.wafer[1] == "mywafer"
    assert obs.pixel[0] == 1
    assert obs.pixel[1] == 2
    assert obs.pixtype[0] == "A"
    assert obs.pixtype[1] is None
    assert obs.alpha[0] == 1.0
    assert np.isnan(obs.alpha[1])
    assert obs.ellipticity[0] == 1.0
    assert obs.ellipticity[1] == 2.0
    assert np.all(obs.quat[0] == np.zeros(4))
    assert np.all(obs.quat[1] == np.ones(4))

    if comm_world.size % 2 == 0:
        obs.set_n_blocks(n_blocks_time=comm_world.size // 2, n_blocks_det=2)
        if comm_world.rank < comm_world.size // 2:
            assert obs.name[0] == "pol01"
            assert obs.wafer[0] == "mywafer"
            assert obs.pixel[0] == 1
            assert obs.pixtype[0] == "A"
            assert obs.ellipticity[0] == 1.0
            assert np.all(obs.quat[0] == np.zeros(4))
            assert obs.alpha[0] == 1.0
        elif comm_world.rank > comm_world.size // 2:
            assert obs.name[0] == "pol02"
            assert obs.wafer[0] == "mywafer"
            assert obs.pixel[0] == 2
            assert obs.pixtype[0] is None
            assert obs.ellipticity[0] == 2.0
            assert np.all(obs.quat[0] == np.ones(4))
            assert np.isnan(obs.alpha[0])


def test_observation_invalid_n_blocks():
    comm_world = lbs.MPI_COMM_WORLD

    # more blocks than the number of samples
    with pytest.raises(ValueError):
        obs = lbs.Observation(
            detectors=3,
            n_samples_observation=comm_world.size - 1,
            start_time_observation=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=comm_world.size,
            n_blocks_det=1,
            comm=comm_world,
        )

    # more blocks than the number of detectors
    with pytest.raises(ValueError):
        obs = lbs.Observation(
            detectors=comm_world.size - 1,
            n_samples_observation=9,
            start_time_observation=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=1,
            n_blocks_det=comm_world.size,
            comm=comm_world,
        )

    # more blocks than the number of MPI processes
    with pytest.raises(ValueError):
        obs = lbs.Observation(
            detectors=3,
            n_samples_observation=9,
            start_time_observation=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=comm_world.size,
            n_blocks_det=2,
            comm=comm_world,
        )

    # when set_n_blocks is called with more blocks than the number of MPI processes
    with pytest.raises(ValueError):
        obs = lbs.Observation(
            detectors=3,
            n_samples_observation=9,
            start_time_observation=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=comm_world.size,
            n_blocks_det=1,
            comm=comm_world,
        )
        obs.set_n_blocks(n_blocks_time=comm_world.size, n_blocks_det=2)


def test_observation_tod_time_blocks():
    comm_world = lbs.MPI_COMM_WORLD
    comm_size = comm_world.size
    comm_rank = comm_world.rank
    if comm_size > 4:
        pytest.skip("This test requires at most 9 MPI processes")

    nsamples_global = 9
    obs = lbs.Observation(
        detectors=3,
        n_samples_observation=nsamples_global,
        start_time_observation=0.0,
        sampling_rate_hz=1.0,
        n_blocks_time=comm_world.size,
        comm=comm_world,
    )

    remainder = nsamples_global % comm_size
    if comm_rank < remainder:
        assert obs.tod.shape == (3, nsamples_global // comm_size + 1)
    else:
        assert obs.tod.shape == (3, nsamples_global // comm_size)

    assert obs.tod.dtype == np.float32


def test_observation_tod_two_block_det():
    comm_world = lbs.MPI_COMM_WORLD
    try:
        obs = lbs.Observation(
            detectors=3,
            n_samples_observation=9,
            start_time_observation=0.0,
            sampling_rate_hz=1.0,
            n_blocks_det=2,
            comm=comm_world,
        )
    except ValueError:
        pytest.skip("This test requires exactly 2 MPI processes")

    if comm_world.rank == 0:
        assert obs.tod.shape == (2, 9)
    elif comm_world.rank == 1:
        assert obs.tod.shape == (1, 9)


def test_observation_tod_set_blocks():
    comm_world = lbs.MPI_COMM_WORLD
    if comm_world.size not in [2, 3, 4, 6]:
        pytest.skip("This test requires 2, 3, 4, or 6 MPI processes")

    def assert_det_info():
        assert np.all(
            obs.row_int == (obs.tod[:, 0] // obs._n_samples_observation).astype(int)
        )
        assert np.all(obs.row_int.astype(str) == obs.row_str)

    if comm_world.size == 2:
        obs = lbs.Observation(
            detectors=3,
            n_samples_observation=9,
            start_time_observation=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=2,
            comm=comm_world,
        )

        # Two time blocks
        ref_tod = np.arange(27, dtype=np.float32).reshape(3, 9)
        if comm_world.rank == 0:
            obs.tod[:] = ref_tod[:, :5]
        elif comm_world.rank == 1:
            obs.tod[:] = ref_tod[:, 5:]

        # Add detector info
        obs.setattr_det_global("row_int", np.arange(3))
        obs.setattr_det_global("row_str", np.array("0 1 2".split()))
        assert_det_info()

        # Two detector blocks
        obs.set_n_blocks(n_blocks_time=1, n_blocks_det=2)
        if comm_world.rank == 0:
            assert np.all(obs.tod == ref_tod[:2])
        elif comm_world.rank == 1:
            assert np.all(obs.tod == ref_tod[2:])
        assert_det_info()

    elif comm_world.size == 3:
        obs = lbs.Observation(
            detectors=3,
            n_samples_observation=9,
            start_time_observation=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=3,
            comm=comm_world,
        )

        # Three time blocks
        ref_tod = np.arange(27, dtype=np.float32).reshape(3, 9)
        if comm_world.rank == 0:
            obs.tod[:] = ref_tod[:, :3]
        elif comm_world.rank == 1:
            obs.tod[:] = ref_tod[:, 3:6]
        elif comm_world.rank == 2:
            obs.tod[:] = ref_tod[:, 6:]

        # Add detector info
        obs.setattr_det_global("row_int", np.arange(3))
        obs.setattr_det_global("row_str", np.array("0 1 2".split()))
        assert_det_info()

        # Three detector blocks
        obs.set_n_blocks(n_blocks_time=1, n_blocks_det=3)
        if comm_world.rank == 0:
            assert np.all(obs.tod == ref_tod[:1])
        elif comm_world.rank == 1:
            assert np.all(obs.tod == ref_tod[1:2])
        elif comm_world.rank == 2:
            assert np.all(obs.tod == ref_tod[2:])
        assert_det_info()

    elif comm_world.size == 4:
        obs = lbs.Observation(
            detectors=3,
            n_samples_observation=9,
            start_time_observation=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=4,
            comm=comm_world,
        )

        # Four time blocks
        ref_tod = np.arange(27, dtype=np.float32).reshape(3, 9)
        if comm_world.rank == 0:
            obs.tod[:] = ref_tod[:, :3]
        elif comm_world.rank == 1:
            obs.tod[:] = ref_tod[:, 3:5]
        elif comm_world.rank == 2:
            obs.tod[:] = ref_tod[:, 5:7]
        elif comm_world.rank == 3:
            obs.tod[:] = ref_tod[:, 7:]

        # Add detector info
        obs.setattr_det_global("row_int", np.arange(3))
        obs.setattr_det_global("row_str", np.array("0 1 2".split()))
        assert_det_info()

        # Two detector blocks, two time blocks
        obs.set_n_blocks(n_blocks_time=2, n_blocks_det=2)
        if comm_world.rank == 0:
            assert np.all(obs.tod == ref_tod[:2, :5])
        elif comm_world.rank == 1:
            assert np.all(obs.tod == ref_tod[:2, 5:])
        elif comm_world.rank == 2:
            assert np.all(obs.tod == ref_tod[2:, :5])
        elif comm_world.rank == 3:
            assert np.all(obs.tod == ref_tod[2:, 5:])
        assert_det_info()

    elif comm_world.size == 6:
        obs = lbs.Observation(
            detectors=3,
            n_samples_observation=9,
            start_time_observation=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=6,
            comm=comm_world,
        )

        # Six time blocks
        ref_tod = np.arange(27, dtype=np.float32).reshape(3, 9)
        if comm_world.rank == 0:
            obs.tod[:] = ref_tod[:, :2]
        elif comm_world.rank == 1:
            obs.tod[:] = ref_tod[:, 2:4]
        elif comm_world.rank == 2:
            obs.tod[:] = ref_tod[:, 4:6]
        elif comm_world.rank == 3:
            obs.tod[:] = ref_tod[:, 6:7]
        elif comm_world.rank == 4:
            obs.tod[:] = ref_tod[:, 7:8]
        elif comm_world.rank == 5:
            obs.tod[:] = ref_tod[:, 8:]

        # Add detector info
        obs.setattr_det_global("row_int", np.arange(3))
        obs.setattr_det_global("row_str", np.array("0 1 2".split()))
        assert_det_info()

        # Two detector blocks, three time blocks
        obs.set_n_blocks(n_blocks_time=3, n_blocks_det=2)
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
        assert_det_info()

        # Three detector blocks, two time blocks
        obs.set_n_blocks(n_blocks_time=2, n_blocks_det=3)
        if comm_world.rank == 0:
            assert np.all(obs.tod == ref_tod[0, :5])
        elif comm_world.rank == 1:
            assert np.all(obs.tod == ref_tod[0, 5:])
        elif comm_world.rank == 2:
            assert np.all(obs.tod == ref_tod[1, :5])
        elif comm_world.rank == 3:
            assert np.all(obs.tod == ref_tod[1, 5:])
        elif comm_world.rank == 4:
            assert np.all(obs.tod == ref_tod[2, :5])
        elif comm_world.rank == 5:
            assert np.all(obs.tod == ref_tod[2, 5:])
        assert_det_info()

        # Six time blocks
        obs.set_n_blocks(n_blocks_time=6, n_blocks_det=1)
        if comm_world.rank == 0:
            assert np.all(obs.tod == ref_tod[:, :2])
        elif comm_world.rank == 1:
            assert np.all(obs.tod == ref_tod[:, 2:4])
        elif comm_world.rank == 2:
            assert np.all(obs.tod == ref_tod[:, 4:6])
        elif comm_world.rank == 3:
            assert np.all(obs.tod == ref_tod[:, 6:7])
        elif comm_world.rank == 4:
            assert np.all(obs.tod == ref_tod[:, 7:8])
        elif comm_world.rank == 5:
            assert np.all(obs.tod == ref_tod[:, 8:])
        assert_det_info()


def test_observation_flags_set_blocks():
    # Flags must follow the samples of the TOD when the blocks change; the
    # TOD encodes the global (detector, sample) indices of each element
    comm_world = lbs.MPI_COMM_WORLD
    n_dets, n_samples = 3, 9

    obs = lbs.Observation(
        detectors=n_dets,
        n_samples_observation=n_samples,
        start_time_observation=0.0,
        sampling_rate_hz=1.0,
        n_blocks_time=comm_world.size,
        comm=comm_world,
    )
    ref_tod = np.arange(n_dets * n_samples, dtype=np.float32).reshape(n_dets, -1)
    obs.tod[:] = ref_tod[
        :,
        obs.start_sample_observation : obs.start_sample_observation
        + obs.n_samples_process,
    ]

    # Only process 0 writes the flags (the first n0 samples): the others
    # must allocate them
    n0 = (
        comm_world.bcast(obs.n_samples_process)
        if lbs.MPI_ENABLED
        else obs.n_samples_process
    )
    if comm_world.rank == 0:
        lbs.get_local_flags(obs)[:] = obs.tod % 7
        lbs.get_global_flags(obs)[:] = np.arange(obs.n_samples_process) % 5

    layouts = [
        (n_det, comm_world.size // n_det)
        for n_det in range(1, n_dets + 1)
        if comm_world.size % n_det == 0
    ]
    layouts += layouts[::-1]
    for n_blocks_det, n_blocks_time in layouts:
        obs.set_n_blocks(n_blocks_det=n_blocks_det, n_blocks_time=n_blocks_time)

        sample = obs.tod.astype(int) % n_samples
        assert obs.local_flags.shape == obs.tod.shape
        np.testing.assert_array_equal(
            obs.local_flags, np.where(sample < n0, obs.tod % 7, 0)
        )
        assert obs.global_flags.shape == (obs.tod.shape[1],)
        if obs.tod.size > 0:
            np.testing.assert_array_equal(
                obs.global_flags, np.where(sample[0] < n0, sample[0] % 5, 0)
            )


def test_write_hdf5_mpi(mpi_tmp_path):
    start_time = 0
    time_span_s = 60
    sampling_hz = 10

    sim = lbs.Simulation(
        base_path=mpi_tmp_path,
        start_time=start_time,
        duration_s=time_span_s,
        random_seed=12345,
    )

    det = lbs.DetectorInfo(
        name="Dummy detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    num_of_obs = 12
    sim.create_observations(detectors=[det], num_of_obs_per_detector=num_of_obs)

    file_names = sim.write_observations(
        subdir_name="tod", file_name_mask="litebird_tod{global_index:04d}.h5"
    )

    assert len(file_names) == len(sim.observations)

    if lbs.MPI_ENABLED:
        # Wait that all the processes have completed writing the files
        lbs.MPI_COMM_WORLD.barrier()

    tod_path = sim.base_path / "tod"
    files_found = list(tod_path.glob("litebird_tod*.h5"))
    assert len(files_found) == num_of_obs, (
        f"{len(files_found)} files found in {tod_path} instead of "
        + f"{num_of_obs}: {files_found}"
    )
    for idx in range(num_of_obs):
        cur_tod = tod_path / f"litebird_tod{idx:04d}.h5"
        assert cur_tod.is_file(), f"File {cur_tod} was expected but not found"

    sim.flush()


def test_simulation_random():
    comm_world = lbs.MPI_COMM_WORLD

    # First, we want to test that by using the same seed, the results are the same
    sim1 = lbs.Simulation(random_seed=12345)
    sim2 = lbs.Simulation(random_seed=12345)
    assert sim1.random is not None
    assert sim2.random is not None

    state1 = sim1.random.bit_generator.state
    state2 = sim2.random.bit_generator.state

    assert state1["bit_generator"] == "PCG64"
    assert state2["bit_generator"] == "PCG64"
    assert state1["has_uint32"] == 0
    assert state2["has_uint32"] == 0
    assert state1["uinteger"] == 0
    assert state2["uinteger"] == 0

    # We only check the state of the first four MPI process. It's important
    # to ensure that they are all different, but there is little sense in
    # checking *every* process.
    if comm_world.rank == 0:
        assert state1["state"]["state"] == 24896973052328222577814399574126207392
        assert state2["state"]["state"] == 24896973052328222577814399574126207392
    elif comm_world.rank == 1:
        assert state1["state"]["state"] == 158287254809478086677339590508859947181
        assert state2["state"]["state"] == 158287254809478086677339590508859947181
    elif comm_world.rank == 2:
        assert state1["state"]["state"] == 133763967953742274472419503117976972596
        assert state2["state"]["state"] == 133763967953742274472419503117976972596
    elif comm_world.rank == 3:
        assert state1["state"]["state"] == 233910118701024945237145923486727240452
        assert state2["state"]["state"] == 233910118701024945237145923486727240452

    # Second, we want to test that by using None as seed, the results are different
    sim3 = lbs.Simulation(random_seed=None)
    sim4 = lbs.Simulation(random_seed=None)
    # Even if random_seed=None, we want a RNG
    assert sim3.random is not None
    assert sim4.random is not None

    state3 = sim3.random.bit_generator.state
    state4 = sim4.random.bit_generator.state

    # Even if random_seed=None, the RNG is still a PCG64
    assert state3["bit_generator"] == "PCG64"
    assert state4["bit_generator"] == "PCG64"
    assert state3["has_uint32"] == 0
    assert state4["has_uint32"] == 0
    assert state3["uinteger"] == 0
    assert state4["uinteger"] == 0

    # We only check the state of the first four MPI process. It's important
    # to ensure that they are all different, but there is little sense in
    # checking *every* process.
    if comm_world.rank == 0:
        assert state3["state"]["state"] != state4["state"]["state"]
    elif comm_world.rank == 1:
        assert state3["state"]["state"] != state4["state"]["state"]
    elif comm_world.rank == 2:
        assert state3["state"]["state"] != state4["state"]["state"]
    elif comm_world.rank == 3:
        assert state3["state"]["state"] != state4["state"]["state"]


def test_issue314(mpi_tmp_path):
    """Check if issue 314 is solved

    See https://github.com/litebird/litebird_sim/issues/314
    """
    if MPI_COMM_WORLD.size != 2:
        # This test is meant to be executed with 2 MPI tasks, as
        # `__write_complex_observation` creates 2 observations
        pytest.skip("This test is meant to be executed with exactly 2 MPI tasks")

    rank = lbs.MPI_COMM_WORLD.rank

    tmp_path = mpi_tmp_path

    start_time = 0
    time_span_s = 60
    sampling_hz = 10

    sim = lbs.Simulation(
        base_path=tmp_path,
        start_time=start_time,
        duration_s=time_span_s,
        random_seed=12345,
        imo=lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION),
        mpi_comm=lbs.MPI_COMM_WORLD,
    )

    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy.from_imo(
            sim.imo, "/releases/vPTEP/satellite/scanning_parameters"
        ),
        delta_time_s=1.0,
    )

    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            sim.imo, "/releases/vPTEP/satellite/LFT/instrument_info"
        )
    )

    hwp = lbs.IdealHWP(
        ang_speed_radpsec=1.0,
        start_angle_rad=2.0,
    )
    sim.set_hwp(hwp)

    det1 = lbs.DetectorInfo(
        name="Dummy detector # 1",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    det2 = lbs.DetectorInfo(
        name="Dummy detector # 1",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    sim.create_observations(
        detectors=[det1, det2],
        n_blocks_det=2,
        split_list_over_processes=False,
    )
    sim.prepare_pointings(append_to_report=False)

    obs = sim.observations[0]
    obs.tod[:] = np.random.random(obs.tod.shape)

    if rank == 0:
        assert obs.det_idx == [0]
    elif rank == 1:
        assert obs.det_idx == [1]
    else:
        assert False, "This should not happen!"

    sim.write_observations(
        subdir_name="",
        gzip_compression=False,
        write_full_pointings=False,
    )

    if lbs.MPI_ENABLED:
        lbs.MPI_COMM_WORLD.barrier()

    observations = lbs.read_list_of_observations(
        file_name_list=tmp_path.glob("*.h5"),
    )
    assert len(observations) == 1

    obs = observations[0]

    if rank == 0:
        assert obs.det_idx == [0]
    elif rank == 1:
        assert obs.det_idx == [1]
    else:
        assert False, "This should not happen!"

    sim.flush()


@pytest.mark.parametrize("use_mjd", [False, True])
def test_read_hwp_angle_of_observation_split_in_time(mpi_tmp_path, use_mjd):
    # If an observation is split in time among the processes, the HWP angle of
    # the observations read back must still be referred to the global start
    comm = lbs.MPI_COMM_WORLD
    start_time = astrotime.Time("2021-01-01") if use_mjd else 0.0
    sim = lbs.Simulation(
        base_path=mpi_tmp_path / f"mjd{use_mjd}",
        start_time=start_time,
        duration_s=100.0,
        random_seed=12345,
        mpi_comm=comm,
    )
    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=0.785_398_163_397_448_3,
            precession_rate_hz=8.664_850_513_998_931e-05,
            spin_rate_hz=0.000_833_333_333_333_333_4,
            start_time=start_time,
        ),
        append_to_report=False,
    )
    sim.set_instrument(
        lbs.InstrumentInfo(spin_boresight_angle_rad=0.872_664_625_997_164_8)
    )
    sim.set_hwp(lbs.IdealHWP(ang_speed_radpsec=0.1))
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name="det",
                sampling_rate_hz=10.0,
                bandcenter_ghz=100.0,
                quat=[0.0, 0.0, 0.0, 1.0],
            )
        ],
        n_blocks_det=1,
        n_blocks_time=comm.size,
        split_list_over_processes=False,
    )
    sim.prepare_pointings(append_to_report=False)
    _, expected_hwp_angle = sim.observations[0].get_pointings(0)

    file_list = sim.write_observations(subdir_name="", append_to_report=False)
    assert len(file_list) == 1

    obs = lbs.io.read_one_observation(file_list[0], limit_mpi_rank=True)
    assert obs.start_time_process == sim.observations[0].start_time_process
    if use_mjd:
        assert abs((obs.start_time_observation - start_time).to("s").value) < 1e-6
    else:
        assert obs.start_time_observation == start_time
    _, hwp_angle = obs.get_pointings(0)
    np.testing.assert_allclose(hwp_angle, expected_hwp_angle, atol=1e-8)


def test_nullify_mpi(mpi_tmp_path):
    start_time = 0
    time_span_s = 2
    sampling_hz = 12

    sim = lbs.Simulation(
        base_path=mpi_tmp_path,
        start_time=start_time,
        duration_s=time_span_s,
        random_seed=12345,
    )

    det = lbs.DetectorInfo(
        name="Dummy detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    num_of_obs = 12
    sim.create_observations(detectors=[det], num_of_obs_per_detector=num_of_obs)

    # Assert TOD is initially non-zero and specific to rank
    for obs in sim.observations:
        obs.tod[:, :] = lbs.MPI_COMM_WORLD.rank + 1

    sim.nullify_tod()

    # Assert TOD is now zero
    for obs in sim.observations:
        assert np.all(obs.tod == 0)

    sim.flush()


def test_non_linearity_seeding():
    """Check if the seed for each detector is consistent even when
    different MPI tasks share the same detector in different time samples
    """

    if lbs.MPI_COMM_WORLD.size != 2:
        pytest.skip("This test requires exactly 2 MPI processes")

    rank = lbs.MPI_COMM_WORLD.rank

    start_time = 0
    time_span_s = 4
    sampling_hz = 1

    sim = lbs.Simulation(
        start_time=start_time,
        duration_s=time_span_s,
        random_seed=12345,
        imo=lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION),
    )

    det1 = lbs.DetectorInfo(
        name="Dummy detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    det2 = lbs.DetectorInfo(
        name="Dummy detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    sim.create_observations(
        detectors=[det1, det2],
        n_blocks_time=2,
        n_blocks_det=1,
        split_list_over_processes=False,
    )

    nl_params = lbs.NonLinParams(
        sampling_gaussian_loc=0.0, sampling_gaussian_scale=2e-3
    )

    sim.observations[0].tod = np.ones_like(sim.observations[0].tod)

    sim.apply_quadratic_nonlin(nl_params=nl_params, user_seed=1234)

    tods = lbs.MPI_COMM_WORLD.gather(sim.observations[0].tod, root=0)

    if rank == 0:
        np.testing.assert_equal(tods[0], tods[1])


def test_noise_user_seed_differs_across_ranks():
    """With user_seed, detectors on different MPI processes must not get the
    same noise realization"""

    comm = lbs.MPI_COMM_WORLD
    if comm.size < 2:
        pytest.skip("This test requires at least 2 MPI processes")

    sim = lbs.Simulation(start_time=0.0, duration_s=100.0, random_seed=1, mpi_comm=comm)
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(name=f"d{i}", sampling_rate_hz=1.0, net_ukrts=1.0)
            for i in range(comm.size)
        ],
        n_blocks_det=comm.size,
        split_list_over_processes=False,
    )
    sim.add_noise(noise_type="white", user_seed=7, append_to_report=False)

    tods = comm.allgather(sim.observations[0].tod[0].copy())
    assert not np.allclose(tods[0], tods[1])


def test_simulation_get_hitmap():
    # The hit map must count the samples of all the observations of all the
    # MPI processes
    sim = lbs.Simulation(
        start_time=0.0,
        duration_s=600.0,
        random_seed=1,
        mpi_comm=lbs.MPI_COMM_WORLD,
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45),
            precession_rate_hz=1 / 3600 / 3.2,
            spin_rate_hz=0.05 / 60,
        )
    )
    sim.set_instrument(lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(50)))
    dets = [lbs.DetectorInfo(name=n, sampling_rate_hz=1.0) for n in ("a", "b")]
    # At least one observation per process, as `prepare_pointings` needs it
    sim.create_observations(
        detectors=dets, num_of_obs_per_detector=2 * lbs.MPI_COMM_WORLD.size
    )
    sim.prepare_pointings()

    hitmap = sim.get_hitmap(nside=16)

    # Each process only holds some of the samples: the total proves the reduction
    assert hitmap.sum() == len(dets) * 600


def test_binner_split_list_over_processes():
    # With split_list_over_processes=True each observation is local to one
    # process: the binner must still sum all of them. The inverse covariance
    # of the binned map depends only on the pointings, so it must be the same
    # as with the observations distributed over the processes
    size = lbs.MPI_COMM_WORLD.size
    invnpp = []
    for split in (True, False):
        sim = lbs.Simulation(
            start_time=0.0, duration_s=600.0, random_seed=1, mpi_comm=lbs.MPI_COMM_WORLD
        )
        sim.set_scanning_strategy(
            lbs.SpinningScanningStrategy(
                spin_sun_angle_rad=np.deg2rad(45),
                precession_rate_hz=1 / 3600 / 3.2,
                spin_rate_hz=0.05 / 60,
            )
        )
        sim.set_instrument(lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(50)))
        dets = [
            lbs.DetectorInfo(name=n, sampling_rate_hz=1.0, net_ukrts=50.0)
            for n in ("a", "b")
        ]
        # At least one observation per process, as `prepare_pointings` needs it
        sim.create_observations(
            detectors=dets,
            num_of_obs_per_detector=2 * size,
            split_list_over_processes=split,
            n_blocks_time=1 if split else size,
        )
        sim.prepare_pointings()
        invnpp.append(sim.make_binned_map(nside=8).invnpp)
    assert np.allclose(invnpp[0], invnpp[1])


def test_h_maps_observations_split_over_processes(mpi_tmp_path):
    # With num_of_obs_per_detector > 1 the observations of each detector are
    # spread over the processes: the h maps of a detector must sum all of
    # them, as if its samples were in one observation
    from litebird_sim.mapmaking.h_maps import load_h_maps_from_file

    comm = lbs.MPI_COMM_WORLD
    sim = lbs.Simulation(start_time=0.0, duration_s=600.0, random_seed=1, mpi_comm=comm)
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45),
            precession_rate_hz=1 / 3600 / 3.2,
            spin_rate_hz=0.05 / 60,
        )
    )
    sim.set_instrument(lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(50)))
    dets = [
        lbs.DetectorInfo(name=n, sampling_rate_hz=1.0, pol_angle_rad=angle)
        for n, angle in (("a", 0.0), ("b", 0.4))
    ]
    sim.create_observations(detectors=dets, num_of_obs_per_detector=2 * comm.size)
    sim.prepare_pointings()

    n_m_couples = np.array([[0, 0], [2, 0], [4, 0]])
    kwargs = dict(
        nside=4,
        n_m_couples=n_m_couples,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
    )
    result = sim.make_h_maps(output_directory=str(mpi_tmp_path), **kwargs)
    # The duration covers the observations of all the processes
    np.testing.assert_allclose(result.duration_s, 600.0)

    # Reference: one serial observation with the samples of all processes
    local = [(obs.start_time, obs.get_pointings("all")[0]) for obs in sim.observations]
    all_local = comm.allgather(local) if lbs.MPI_ENABLED else [local]
    chunks = sorted(sum(all_local, []), key=lambda x: x[0])
    ref_pointings = np.concatenate([ptg for _, ptg in chunks], axis=1)
    ref_obs = lbs.Observation(
        detectors=[{"name": d.name, "pol_angle_rad": d.pol_angle_rad} for d in dets],
        n_samples_global=ref_pointings.shape[1],
        start_time_global=0.0,
        sampling_rate_hz=1.0,
        allocate_tod=False,
    )
    reference = lbs.make_h_maps(
        observations=ref_obs, pointings=ref_pointings, save_to_file=False, **kwargs
    )

    # The first process observes both detectors, so it owns (and saves) their maps
    if comm.rank != 0:
        assert result.h_maps == {}
        return
    assert set(result.h_maps) == {"a", "b"}
    for name in ("a", "b"):
        loaded = load_h_maps_from_file(mpi_tmp_path / f"h_maps_det_{name}.h5")
        for pair, ref_map in reference.h_maps[name].items():
            for cur_map in (result.h_maps[name][pair], loaded.h_maps[name][pair]):
                np.testing.assert_allclose(cur_map.real, ref_map.real, atol=1e-12)
                np.testing.assert_allclose(cur_map.imag, ref_map.imag, atol=1e-12)
    assert result.h_maps["a"][0, 0].real.sum() == ref_pointings.shape[1]


def test_h_maps_files_of_all_owners_readable_after_call(mpi_tmp_path):
    # Detectors split over the processes have different owners: each process
    # returns only its own maps, but must find the files of all of them as soon
    # as make_h_maps returns
    from litebird_sim.mapmaking.h_maps import load_h_maps_from_file

    comm = lbs.MPI_COMM_WORLD
    sim = lbs.Simulation(start_time=0.0, duration_s=600.0, random_seed=1, mpi_comm=comm)
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45),
            precession_rate_hz=1 / 3600 / 3.2,
            spin_rate_hz=0.05 / 60,
        )
    )
    sim.set_instrument(lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(50)))
    names = [f"det{i}" for i in range(comm.size)]
    dets = [
        lbs.DetectorInfo(name=n, sampling_rate_hz=1.0, pol_angle_rad=0.1 * i)
        for i, n in enumerate(names)
    ]
    sim.create_observations(
        detectors=dets, n_blocks_det=comm.size, split_list_over_processes=False
    )
    sim.prepare_pointings()

    n_m_couples = np.array([[0, 0], [2, 0]])
    result = sim.make_h_maps(
        nside=4,
        n_m_couples=n_m_couples,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        output_directory=str(mpi_tmp_path),
    )

    owned = set(result.h_maps)
    assert set(result.filenames) == owned
    all_owned = comm.allgather(owned) if lbs.MPI_ENABLED else [owned]
    assert sum(len(x) for x in all_owned) == len(names)
    assert set().union(*all_owned) == set(names)

    for name in names:
        loaded = load_h_maps_from_file(mpi_tmp_path / f"h_maps_det_{name}.h5")
        assert loaded.h_maps[name][0, 0].real.sum() == sim.observations[0].n_samples
        if name in owned:
            for pair, cur_map in result.h_maps[name].items():
                np.testing.assert_array_equal(
                    loaded.h_maps[name][pair].real, cur_map.real
                )
                np.testing.assert_array_equal(
                    loaded.h_maps[name][pair].imag, cur_map.imag
                )


def test_reduction_for_observations():
    # Whatever the distribution of the observations, the reduction of the
    # map-makers must sum over all the processes holding them
    from litebird_sim.mapmaking.reduction import reduction_for_observations

    size = lbs.MPI_COMM_WORLD.size
    for split in (True, False):
        sim = lbs.Simulation(
            start_time=0.0, duration_s=600.0, random_seed=1, mpi_comm=lbs.MPI_COMM_WORLD
        )
        sim.create_observations(
            detectors=[lbs.DetectorInfo(name="a", sampling_rate_hz=1.0)],
            num_of_obs_per_detector=2 * size,
            split_list_over_processes=split,
            n_blocks_time=1 if split else size,
        )
        red = reduction_for_observations(sim.observations)
        assert red.size == size

        values = np.full(3, lbs.MPI_COMM_WORLD.rank + 1.0)
        red.sum_in_place(values)
        np.testing.assert_array_equal(values, size * (size + 1) / 2)
        assert red.sum(1) == size
        assert red.max(lbs.MPI_COMM_WORLD.rank) == size - 1


def _group_simulation(comm, duration_s):
    sim = lbs.Simulation(
        start_time=0.0, duration_s=duration_s, random_seed=1, mpi_comm=comm
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45),
            precession_rate_hz=1 / 3600 / 3.2,
            spin_rate_hz=0.05 / 60,
        )
    )
    sim.set_instrument(lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(50)))
    # A T/B pair, so that pair-differencing can run too
    dets = [
        lbs.DetectorInfo(
            name=f"a_{pol}",
            sampling_rate_hz=1.0,
            net_ukrts=50.0,
            pol_angle_rad=angle,
            wafer="w",
            pixel=0,
            pol=pol,
        )
        for pol, angle in (("T", 0.0), ("B", np.pi / 2))
    ]
    sim.create_observations(
        detectors=dets, n_blocks_time=comm.size, split_list_over_processes=False
    )
    sim.prepare_pointings()
    for cur_obs in sim.observations:
        cur_obs.tod[:] = 1.0
    return sim


def _map_maker_outputs(sim):
    """The outputs of every map-maker that depend only on the pointings"""
    nside = 8
    result = {
        "binner": sim.make_binned_map(nside=nside).invnpp,
        "pair_differencing": lbs.make_pair_differenced_map(
            nside=nside, observations=sim.observations
        ).invnpp,
        "destriper": sim.make_destriped_map(
            nside=nside,
            params=lbs.DestriperParameters(samples_per_baseline=100, iter_max=5),
        ).hit_map,
    }
    try:
        import pysanepic  # noqa: F401
    except ImportError:
        pass
    else:
        result["sanepic"] = np.asarray(sim.make_sanepic_gls_map(nside=nside).hit_map)
    return result


def test_map_makers_sum_only_over_their_communicator():
    # Two groups of processes run independent simulations: each map-maker must
    # sum only the observations of its own group, not those of MPI_COMM_WORLD.
    # The reference is the same simulation of the group on one process
    world = lbs.MPI_COMM_WORLD
    if world.size < 2:
        return
    from litebird_sim.mpi import MPI

    color = world.rank % 2
    sub_comm = world.Split(color, world.rank)
    duration_s = 600.0 * (color + 1)  # The two groups differ

    outputs = _map_maker_outputs(_group_simulation(sub_comm, duration_s))
    reference = _map_maker_outputs(_group_simulation(MPI.COMM_SELF, duration_s))

    # Summing over MPI_COMM_WORLD would mix the groups and give them the same
    # outputs, also in the reference, which is why both checks are needed
    totals = world.allgather(
        (color, {name: float(np.sum(x[x > -1e29])) for name, x in outputs.items()})
    )
    group_totals = {c: t for c, t in totals}
    for name in outputs:
        assert not np.isclose(group_totals[0][name], group_totals[1][name]), name

    assert set(outputs) == set(reference)
    for name, ref in reference.items():
        # Near-singular pixels amplify the rounding of a different summation
        # order: compare with a tolerance relative to the largest value
        seen = ref > -1e29  # Exclude UNSEEN pixels from the scale
        atol = 1e-8 * np.max(np.abs(ref[seen]))
        np.testing.assert_allclose(outputs[name], ref, atol=atol, err_msg=name)

    sub_comm.Free()


def test_describe_mpi_layout():
    # A compact summary of the distribution, consistent on every process
    size = lbs.MPI_COMM_WORLD.size
    sim = lbs.Simulation(
        start_time=0.0,
        duration_s=400.0 * size,
        random_seed=1,
        mpi_comm=lbs.MPI_COMM_WORLD,
    )
    sim.set_instrument(lbs.InstrumentInfo(name="instr"))
    dets = [lbs.DetectorInfo(name=n, sampling_rate_hz=1.0) for n in ("a", "b")]
    sim.create_observations(
        detectors=dets, num_of_obs_per_detector=2 * size, split_list_over_processes=True
    )
    text = sim.describe_mpi_layout()
    assert f"{size} process" in text
    assert f"Observations: {2 * size} in total" in text
    if size > 1:
        assert "split_list_over_processes=True" in text
    assert "warnings: none" in text
    if lbs.MPI_ENABLED:
        assert lbs.MPI_COMM_WORLD.allgather(text) == [text] * size


def test_chunked_noise_does_not_depend_on_the_mpi_layout():
    # Noise generated in chunks must be the same whether the detectors or the
    # time are split among the processes, and the same as in a serial run
    from dataclasses import asdict

    size = lbs.MPI_COMM_WORLD.size
    n_samples, chunk = 1200, 64  # the time blocks do not end on chunk boundaries
    dets = [
        lbs.DetectorInfo(
            name=f"det{i}",
            sampling_rate_hz=1.0,
            net_ukrts=50.0,
            fknee_mhz=100.0,
            fmin_hz=1e-3,
            alpha=1.0,
        )
        for i in range(max(2, size) + 1)
    ]
    tods = [
        lbs.TodDescription(
            name="tod", dtype=np.float64, units=lbs.Units.K_CMB, description=""
        )
    ]

    # Serial reference, computed by every process on its own
    reference = lbs.Observation(
        detectors=[asdict(d) for d in dets],
        n_samples_global=n_samples,
        start_time_global=0.0,
        sampling_rate_hz=1.0,
        tods=tods,
    )
    lbs.add_noise_to_observations(
        reference,
        "one_over_f",
        user_seed=42,
        chunk_samples=chunk,
        chunk_reference_time=0.0,
    )
    ref_tod = {name: reference.tod[i] for i, name in enumerate(reference.name)}

    for n_blocks_det, n_blocks_time in ((size, 1), (1, size)):
        sim = lbs.Simulation(
            start_time=0.0,
            duration_s=float(n_samples),
            random_seed=42,
            mpi_comm=lbs.MPI_COMM_WORLD,
        )
        sim.create_observations(
            detectors=dets,
            n_blocks_det=n_blocks_det,
            n_blocks_time=n_blocks_time,
            split_list_over_processes=False,
            tods=tods,
        )
        sim.add_noise(
            noise_type="one_over_f", chunk_samples=chunk, append_to_report=False
        )
        for obs in sim.observations:
            first, n = obs.start_sample, obs.n_samples
            for i, name in enumerate(obs.name):
                np.testing.assert_array_equal(
                    obs.tod[i], ref_tod[name][first : first + n]
                )


if __name__ == "__main__":
    pytest.main([f"{__file__}"])


def test_mpi_disabled_does_not_import_mpi4py():
    # With LITEBIRD_SIM_MPI=0, importing litebird_sim must not initialize MPI:
    # importing `mpi4py.MPI` calls MPI_Init, which hangs outside `srun` on some
    # clusters (e.g., Intel MPI)
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, litebird_sim; sys.exit('mpi4py.MPI' in sys.modules)",
        ],
        env={**os.environ, "LITEBIRD_SIM_MPI": "0"},
    )
    assert result.returncode == 0


def test_mpi_excepthook():
    # With more than one process, an unhandled exception must abort the job
    # instead of leaving the other processes hanging
    import sys

    from litebird_sim.mpi import _mpi_excepthook

    if lbs.MPI_ENABLED and MPI_COMM_WORLD.size > 1:
        assert sys.excepthook is _mpi_excepthook
    else:
        assert sys.excepthook is not _mpi_excepthook


@pytest.mark.parametrize("keep_maps", [False, True])
@pytest.mark.parametrize("num_observations", [1, 2])
def test_h_maps_idle_processes(mpi_tmp_path, keep_maps, num_observations):
    from litebird_sim.mapmaking.h_maps import load_h_maps_from_file

    comm = lbs.MPI_COMM_WORLD
    sim = lbs.Simulation(start_time=0.0, duration_s=8.0, random_seed=1, mpi_comm=comm)
    sim.create_observations(
        detectors=[lbs.DetectorInfo(name="A", sampling_rate_hz=1.0)],
        num_of_obs_per_detector=num_observations,
        allocate_tod=False,
    )
    for obs in sim.observations:
        obs.pointing_matrix = np.broadcast_to(
            np.array([1.0, 0.2, 0.3]),
            (obs.n_detectors, obs.n_samples_process, 3),
        ).copy()
    result = sim.make_h_maps(
        nside=1,
        n_m_couples=np.array([[0, 0], [2, 0]]),
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        output_directory=str(mpi_tmp_path),
        keep_maps=keep_maps,
    )
    if comm.rank != 0:
        assert result.h_maps == {}
        assert result.filenames == {}
        return
    assert set(result.filenames) == {"A"}
    loaded = load_h_maps_from_file(result.filenames["A"])
    assert loaded.h_maps["A"][0, 0].real.sum() == 8
    observed = loaded.h_maps["A"][0, 0].real > 0
    np.testing.assert_allclose(loaded.h_maps["A"][2, 0].real[observed], np.cos(0.6))
    np.testing.assert_allclose(loaded.h_maps["A"][2, 0].imag[observed], np.sin(0.6))
    assert set(result.h_maps) == ({"A"} if keep_maps else set())


def test_h_maps_empty_global_list():
    with pytest.raises(ValueError, match="At least one observation"):
        lbs.make_h_maps([], nside=1, comm=lbs.MPI_COMM_WORLD)


def test_h_maps_stream_detector_subsets(mpi_tmp_path, monkeypatch):
    import weakref
    import healpy as hp
    from litebird_sim.mapmaking.h_maps import load_h_maps_from_file

    comm = lbs.MPI_COMM_WORLD
    rank = comm.rank
    observations = []
    records = []
    for chunk in range(2):
        # Shared by everyone, shared by a pair of ranks, and local to one rank.
        # Both ordering and the number of samples change between observations.
        names = ["shared", f"pair_{rank // 2}", f"local_{rank}"]
        if chunk:
            names.reverse()
        nsamples = 3 + rank + 2 * chunk
        obs = lbs.Observation(
            detectors=[{"name": name, "pol_angle_rad": 0.0} for name in names],
            n_samples_observation=nsamples,
            start_time_observation=float(chunk * 10),
            sampling_rate_hz=1.0,
            allocate_tod=False,
        )
        if lbs.MPI_ENABLED:
            obs.list_comm = comm
        obs.has_hwp = True
        obs.hwp_angle = np.arange(nsamples) * 0.13 + chunk * 0.07
        obs.pointing_matrix = np.empty((len(names), nsamples, 3))
        for idet, name in enumerate(names):
            pix = (np.arange(nsamples) + idet + rank) % 12
            psi = np.arange(nsamples) * 0.17 + rank * 0.11 + chunk * 0.23
            theta, phi = hp.pix2ang(1, pix)
            obs.pointing_matrix[idet] = np.column_stack((theta, phi, psi))
            records.append((name, pix, psi, obs.hwp_angle))
        observations.append(obs)

    pairs = np.array([[0, 0], [2, 0], [2, -4]])
    original_zeros = np.zeros
    refs = []
    allocations = 0

    def tracked_zeros(shape, *args, **kwargs):
        nonlocal allocations
        if isinstance(shape, tuple) and shape == (12, 3):
            array = original_zeros(shape, *args, **kwargs)
            refs.append(weakref.ref(array))
            allocations += 1
            # Includes non-owners and ranks without samples for this detector.
            # No process may hold more than one detector's working matrices.
            assert sum(ref() is not None for ref in refs) <= len(pairs)
            return array
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", tracked_zeros)
    result = lbs.make_h_maps(
        observations,
        nside=1,
        n_m_couples=pairs,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        output_directory=str(mpi_tmp_path),
        keep_maps=False,
    )
    assert result.h_maps == {}
    assert allocations == len(pairs)
    assert all(ref() is None for ref in refs)
    all_records = comm.allgather(records) if lbs.MPI_ENABLED else [records]
    all_files = (
        comm.allgather(result.filenames) if lbs.MPI_ENABLED else [result.filenames]
    )
    expected = {}
    owners = {}
    for irank, local in enumerate(all_records):
        for name, pix, psi, hwp in local:
            expected.setdefault(name, []).append((pix, psi, hwp))
            owners.setdefault(name, irank)
    for name, chunks in expected.items():
        assert [i for i, files in enumerate(all_files) if name in files] == [
            owners[name]
        ]
        if rank != owners[name]:
            continue
        pix, psi, hwp = [np.concatenate(items) for items in zip(*chunks)]
        counts = np.bincount(pix, minlength=12)
        loaded = load_h_maps_from_file(result.filenames[name])
        for n, m in pairs:
            if (n, m) == (0, 0):
                real, imag = counts, np.zeros(12)
            else:
                phase = n * psi + m * hwp
                real = np.divide(
                    np.bincount(pix, weights=np.cos(phase), minlength=12),
                    counts,
                    out=np.zeros(12),
                    where=counts > 0,
                )
                imag = np.divide(
                    np.bincount(pix, weights=np.sin(phase), minlength=12),
                    counts,
                    out=np.zeros(12),
                    where=counts > 0,
                )
            np.testing.assert_allclose(loaded.h_maps[name][n, m].real, real, atol=1e-14)
            np.testing.assert_allclose(loaded.h_maps[name][n, m].imag, imag, atol=1e-14)


def test_h_maps_stream_with_idle_rank_zero(mpi_tmp_path):
    from litebird_sim.mapmaking.h_maps import load_h_maps_from_file

    comm = lbs.MPI_COMM_WORLD
    # In a four-process run only ranks 1 and 3 have data; rank 1 owns A.
    active = comm.size == 1 or comm.rank % 2 == 1
    observations = []
    count = 3 + comm.rank if active else 0
    if active:
        obs = lbs.Observation(
            detectors=[{"name": "A", "pol_angle_rad": 0.0}],
            n_samples_observation=count,
            start_time_observation=float(comm.rank * 10),
            sampling_rate_hz=1.0,
            allocate_tod=False,
        )
        obs.pointing_matrix = np.broadcast_to(
            np.array([1.0, 0.2, 0.3]), (1, count, 3)
        ).copy()
        observations.append(obs)
    result = lbs.make_h_maps(
        observations,
        nside=1,
        n_m_couples=np.array([[0, 0], [2, 0]]),
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        output_directory=str(mpi_tmp_path),
        keep_maps=False,
        comm=comm,
    )
    assert result.h_maps == {}
    owner = 0 if comm.size == 1 else 1
    if comm.rank != owner:
        assert result.filenames == {}
        return
    assert set(result.filenames) == {"A"}
    loaded = load_h_maps_from_file(result.filenames["A"])
    expected_hits = (
        3 if comm.size == 1 else sum(3 + rank for rank in range(1, comm.size, 2))
    )
    assert loaded.h_maps["A"][0, 0].real.sum() == expected_hits
    observed = loaded.h_maps["A"][0, 0].real > 0
    np.testing.assert_allclose(loaded.h_maps["A"][2, 0].real[observed], np.cos(0.6))
    np.testing.assert_allclose(loaded.h_maps["A"][2, 0].imag[observed], np.sin(0.6))
