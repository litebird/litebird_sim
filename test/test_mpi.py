# -*- encoding: utf-8 -*-
# NOTE: all the following tests should be valid also in a serial execution

from tempfile import TemporaryDirectory

import numpy as np
import astropy.time as astrotime
import litebird_sim as lbs


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

    if comm_world.rank == 0:
        print(f"MPI configuration: {lbs.MPI_CONFIGURATION}")

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
        assert obs.name is None
        assert obs.wafer is None
        assert obs.pixel is None
        assert obs.pixtype is None
        assert obs.ellipticity is None
        assert obs.quat is None
        assert obs.alpha is None

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
            detectors=3,
            n_samples_global=9,
            start_time_global=0.0,
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
            detectors=3,
            n_samples_global=9,
            start_time_global=0.0,
            sampling_rate_hz=1.0,
            n_blocks_time=2,
            comm=comm_world,
        )
    except ValueError:
        # Not enough processes to split the TOD, constuctor expected to rise
        if comm_world.size < 2:
            return

    def assert_det_info():
        if comm_world.rank < obs._n_blocks_time * obs._n_blocks_det:
            assert np.all(
                obs.row_int == (obs.tod[:, 0] // obs._n_samples_global).astype(int)
            )
            assert np.all(obs.row_int.astype(str) == obs.row_str)
        else:
            assert obs.row_int is None
            assert obs.row_str is None

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
    )

    det = lbs.DetectorInfo(
        name="Dummy detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    num_of_obs = 12
    sim.create_observations(detectors=[det], num_of_obs_per_detector=num_of_obs)

    file_names = lbs.write_observations(
        sim, subdir_name="tod", file_name_mask="litebird_tod{global_index:04d}.h5"
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
    sim = lbs.Simulation()
    assert sim.random is not None

    state = sim.random.bit_generator.state

    comm_world = lbs.MPI_COMM_WORLD

    assert state["bit_generator"] == "PCG64"
    assert state["has_uint32"] == 0
    assert state["uinteger"] == 0

    # We only check the state of the first four MPI process. It's important
    # to ensure that they are all different, but there is little sense in
    # checking *every* process.
    if comm_world.rank == 0:
        assert state["state"]["state"] == 24896973052328222577814399574126207392
    elif comm_world.rank == 1:
        assert state["state"]["state"] == 158287254809478086677339590508859947181
    elif comm_world.rank == 2:
        assert state["state"]["state"] == 133763967953742274472419503117976972596
    elif comm_world.rank == 3:
        assert state["state"]["state"] == 233910118701024945237145923486727240452


def main():
    test_observation_time()
    test_construction_from_detectors()
    test_observation_tod_single_block()
    test_observation_tod_two_block_time()
    test_observation_tod_two_block_det()
    test_observation_tod_set_blocks()
    test_simulation_random()

    # It's critical that all MPI processes use the same output directory
    if lbs.MPI_ENABLED:
        if lbs.MPI_COMM_WORLD.rank == 0:
            tmp_dir = TemporaryDirectory()
            tmp_path = tmp_dir.name
            lbs.MPI_COMM_WORLD.bcast(tmp_path, root=0)
        else:
            tmp_dir = None
            tmp_path = lbs.MPI_COMM_WORLD.bcast(None, root=0)
    else:
        tmp_dir = TemporaryDirectory()
        tmp_path = tmp_dir.name

    try:
        test_write_hdf5_mpi(tmp_path)
    finally:
        # Now we can remove the temporary directory, but first make
        # sure that there are no other MPI processes still waiting to
        # finish
        if lbs.MPI_ENABLED:
            lbs.MPI_COMM_WORLD.barrier()

        if tmp_dir:
            tmp_dir.cleanup()


main()
