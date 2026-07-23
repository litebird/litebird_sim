# -*- encoding: utf-8 -*-
# NOTE: all the following tests should be valid also in a serial execution
from pathlib import Path
from tempfile import TemporaryDirectory

import astropy.time as astrotime
import numpy as np
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
        start_time_global=0.0,
        sampling_rate_hz=5.0,
        n_samples_global=5,
        n_blocks_time=comm_world.size,
        comm=comm_world,
    )
    obs_mjd_astropy = lbs.Observation(
        detectors=1,
        start_time_global=ref_time,
        sampling_rate_hz=5.0,
        n_samples_global=5,
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
        n_samples_global=100,
        start_time_global=0.0,
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
            n_samples_global=comm_world.size - 1,
            start_time_global=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=comm_world.size,
            n_blocks_det=1,
            comm=comm_world,
        )

    # more blocks than the number of detectors
    with pytest.raises(ValueError):
        obs = lbs.Observation(
            detectors=comm_world.size - 1,
            n_samples_global=9,
            start_time_global=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=1,
            n_blocks_det=comm_world.size,
            comm=comm_world,
        )

    # more blocks than the number of MPI processes
    with pytest.raises(ValueError):
        obs = lbs.Observation(
            detectors=3,
            n_samples_global=9,
            start_time_global=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=comm_world.size,
            n_blocks_det=2,
            comm=comm_world,
        )

    # when set_n_blocks is called with more blocks than the number of MPI processes
    with pytest.raises(ValueError):
        obs = lbs.Observation(
            detectors=3,
            n_samples_global=9,
            start_time_global=0.0,
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
        n_samples_global=nsamples_global,
        start_time_global=0.0,
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
            n_samples_global=9,
            start_time_global=0.0,
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
            obs.row_int == (obs.tod[:, 0] // obs._n_samples_global).astype(int)
        )
        assert np.all(obs.row_int.astype(str) == obs.row_str)

    if comm_world.size == 2:
        obs = lbs.Observation(
            detectors=3,
            n_samples_global=9,
            start_time_global=0.0,
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
            n_samples_global=9,
            start_time_global=0.0,
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
            n_samples_global=9,
            start_time_global=0.0,
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
            n_samples_global=9,
            start_time_global=0.0,
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

    if lbs.MPI_COMM_WORLD.size < 2:
        return

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


if __name__ == "__main__":
    pytest.main([f"{__file__}"])
