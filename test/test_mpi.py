# -*- encoding: utf-8 -*-
# NOTE: all the following tests should be valid also in a serial execution
from pathlib import Path
from sys import stderr
from tempfile import TemporaryDirectory
from typing import Callable

import astropy.time as astrotime
import numpy as np

import litebird_sim as lbs
from litebird_sim import MPI_COMM_WORLD


def test_observation_time():
    comm_world = lbs.MPI_COMM_WORLD
    ref_time = astrotime.Time("2020-02-20", format="iso")

    obs_no_mjd = lbs.Observation(
        detectors=1,
        start_time_global=0.0,
        sampling_rate_hz=5.0,
        n_samples_global=5,
        comm=comm_world,
    )
    obs_mjd_astropy = lbs.Observation(
        detectors=1,
        start_time_global=ref_time,
        sampling_rate_hz=5.0,
        n_samples_global=5,
        comm=comm_world,
    )

    res_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    res_mjd = np.array(
        [0.0, 2.314_816_81e-06, 4.629_626_35e-06, 6.944_443_16e-06, 9.259_259_97e-06]
    )
    res_cxcsec = np.array(
        [6.985_440_69e8, 6.985_440_69e8, 6.985_440_70e8, 6.985_440_70e8, 6.985_440_70e8]
    )

    if not comm_world or comm_world.rank == 0:
        assert np.allclose(obs_no_mjd.get_times(), res_times)
        assert np.allclose(
            (obs_mjd_astropy.get_times(astropy_times=True) - ref_time).jd, res_mjd
        )
        assert np.allclose(
            obs_mjd_astropy.get_times(normalize=False, astropy_times=False), res_cxcsec
        )
    else:
        assert obs_no_mjd.get_times().size == 0
        assert obs_mjd_astropy.get_times(astropy_times=True).size == 0
        assert obs_mjd_astropy.get_times(normalize=False, astropy_times=False).size == 0

    if not comm_world or comm_world.size == 1:
        return
    obs_no_mjd.set_n_blocks(n_blocks_time=2)
    obs_mjd_astropy.set_n_blocks(n_blocks_time=2)
    if comm_world.rank == 0:
        assert np.allclose(obs_no_mjd.get_times(), res_times[:3])
        assert np.allclose(
            (obs_mjd_astropy.get_times(astropy_times=True) - ref_time).jd, res_mjd[:3]
        )
        assert np.allclose(
            obs_mjd_astropy.get_times(normalize=False, astropy_times=False),
            res_cxcsec[:3],
        )
    elif comm_world.rank == 1:
        assert np.allclose(obs_no_mjd.get_times(), res_times[3:])
        assert np.allclose(
            (obs_mjd_astropy.get_times(astropy_times=True) - ref_time).jd, res_mjd[3:]
        )
        assert np.allclose(
            obs_mjd_astropy.get_times(normalize=False, astropy_times=False),
            res_cxcsec[3:],
        )
    else:
        assert obs_no_mjd.get_times().size == 0
        assert obs_mjd_astropy.get_times().size == 0


def test_construction_from_detectors():
    comm_world = lbs.MPI_COMM_WORLD

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
        comm=comm_world,
        root=0,
    )

    if comm_world.rank == 0:
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

    if comm_world.size == 1:
        return

    obs.set_n_blocks(n_blocks_time=1, n_blocks_det=2)
    if comm_world.rank == 0:
        assert obs.name[0] == "pol01"
        assert obs.wafer[0] == "mywafer"
        assert obs.pixel[0] == 1
        assert obs.pixtype[0] == "A"
        assert obs.ellipticity[0] == 1.0
        assert np.all(obs.quat[0] == np.zeros(4))
        assert obs.alpha[0] == 1.0
    elif comm_world.rank == 1:
        assert obs.name[0] == "pol02"
        assert obs.wafer[0] == "mywafer"
        assert obs.pixel[0] == 2
        assert obs.pixtype[0] is None
        assert obs.ellipticity[0] == 2.0
        assert np.all(obs.quat[0] == np.ones(4))
        assert np.isnan(obs.alpha[0])
    else:
        assert obs.name == [None]
        assert obs.wafer == [None]
        assert obs.pixel == [None]
        assert obs.pixtype == [None]
        assert obs.quat == [None]
        # On the processes, that does not own any detector (and TOD), the numerical
        # attributes of `DetectorInfo()` are assigned to zero
        assert obs.ellipticity == 0
        assert obs.alpha == 0

    obs.set_n_blocks(n_blocks_time=1, n_blocks_det=1)
    if comm_world.rank == 0:
        assert obs.name[0] == "pol01"
        assert obs.name[1] == "pol02"
        assert obs.wafer[0] == "mywafer"
        assert obs.wafer[1] == "mywafer"
        assert obs.pixel[0] == 1
        assert obs.pixel[1] == 2
        assert obs.pixtype[0] == "A"
        assert obs.pixtype[1] is None
        assert obs.ellipticity[0] == 1.0
        assert obs.ellipticity[1] == 2.0
        assert obs.alpha[0] == 1.0
        assert np.isnan(obs.alpha[1])
        assert np.allclose(obs.quat, np.arange(2)[:, None])


def test_observation_tod_single_block():
    comm_world = lbs.MPI_COMM_WORLD
    obs = lbs.Observation(
        detectors=3,
        n_samples_global=9,
        start_time_global=0.0,
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
            detectors=3,
            n_samples_global=9,
            start_time_global=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=2,
            comm=comm_world,
        )
    except ValueError:
        # Not enough processes to split the TOD, constructor expected to rise
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
            detectors=3,
            n_samples_global=9,
            start_time_global=0.0,
            sampling_rate_hz=1.0,
            n_blocks_det=2,
            comm=comm_world,
        )
    except ValueError:
        # Not enough processes to split the TOD, constructor expected to rise
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
            detectors=3,
            n_samples_global=9,
            start_time_global=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=2,
            comm=comm_world,
        )
    except ValueError:
        # Not enough processes to split the TOD, constructor expected to rise
        if comm_world.size < 2:
            return

    def assert_det_info():
        if comm_world.rank < obs._n_blocks_time * obs._n_blocks_det:
            assert np.all(
                obs.row_int == (obs.tod[:, 0] // obs._n_samples_global).astype(int)
            )
            assert np.all(obs.row_int.astype(str) == obs.row_str)
        else:
            assert obs.row_int == [None]
            assert obs.row_str == [None]

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
    else:
        assert obs.tod.size == 0
    assert_det_info()

    # One block
    obs.set_n_blocks(n_blocks_det=1, n_blocks_time=1)
    if comm_world.rank == 0:
        assert np.all(obs.tod == ref_tod)
    else:
        assert obs.tod.size == 0
    assert_det_info()

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
    assert_det_info()

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
    assert_det_info()

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
    assert_det_info()

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
    assert_det_info()


def test_write_hdf5_mpi(tmp_path):
    start_time = 0
    time_span_s = 60
    sampling_hz = 10

    sim = lbs.Simulation(
        base_path=tmp_path,
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


def test_issue314(tmp_path):
    """Check if issue 314 is solved

    See https://github.com/litebird/litebird_sim/issues/314
    """
    if MPI_COMM_WORLD.size != 2:
        # This test is meant to be executed with 2 MPI tasks, as
        # `__write_complex_observation` creates 2 observations
        return

    rank = lbs.MPI_COMM_WORLD.rank

    tmp_path = Path(tmp_path)

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


def __run_test_in_same_folder(test_fn: Callable) -> None:
    if not lbs.MPI_ENABLED:
        return

    # It's critical that all MPI processes use the same output directory
    if lbs.MPI_COMM_WORLD.rank == 0:
        tmp_dir = TemporaryDirectory()
        tmp_path = tmp_dir.name
        lbs.MPI_COMM_WORLD.bcast(tmp_path, root=0)
    else:
        tmp_dir = None
        tmp_path = lbs.MPI_COMM_WORLD.bcast(None, root=0)

    failure = False
    try:
        test_fn(tmp_path)
    except Exception:
        failure = True

        from traceback import format_exc

        print(
            "MPI process #{rank} failed with exception: {exc}".format(
                rank=lbs.MPI_COMM_WORLD.rank,
                exc=format_exc(),
            ),
            file=stderr,
        )

    if tmp_dir:
        tmp_dir.cleanup()

    if failure:
        lbs.MPI_COMM_WORLD.Abort(1)


def test_nullify_mpi(tmp_path):
    start_time = 0
    time_span_s = 2
    sampling_hz = 12

    sim = lbs.Simulation(
        base_path=tmp_path,
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


if __name__ == "__main__":
    test_functions = [
        test_observation_time,
        test_construction_from_detectors,
        test_observation_tod_single_block,
        test_observation_tod_two_block_time,
        test_observation_tod_two_block_det,
        test_observation_tod_set_blocks,
        test_simulation_random,
    ]

    for cur_test_fn in test_functions:
        if MPI_COMM_WORLD.rank == 0:
            print("Running test function {}".format(str(cur_test_fn)), file=stderr)
        cur_test_fn()

    same_folder_test_functions = [
        test_write_hdf5_mpi,
        test_issue314,
        test_nullify_mpi,
    ]

    for cur_test_fn in same_folder_test_functions:
        if MPI_COMM_WORLD.rank == 0:
            print("Running test function {}".format(str(cur_test_fn)), file=stderr)
        __run_test_in_same_folder(cur_test_fn)
