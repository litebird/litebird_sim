# -*- encoding: utf-8 -*-

import os
import pathlib
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from uuid import UUID

import astropy
import numpy as np
import pytest

import litebird_sim as lbs


class MockPlot:
    def savefig(*args, **kwargs):
        pass


def test_healpix_map_write(tmp_path):
    sim = lbs.Simulation(base_path=tmp_path / "simulation_dir", random_seed=12345)
    output_file = sim.write_healpix_map(filename="test.fits.gz", pixels=np.zeros(12))

    assert isinstance(output_file, pathlib.Path)
    assert output_file.exists()

    sim.append_to_report(
        """Here is a plot:

 ![](myplot.png)
 """,
        [(MockPlot(), "myplot.png")],
    )

    sim.flush()


def test_markdown_report(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        name="My simulation",
        description="Lorem ipsum",
        start_time=1.0,
        duration_s=3600.0,
        random_seed=12345,
    )
    output_file = sim.write_healpix_map(filename="test.fits.gz", pixels=np.zeros(12))

    assert isinstance(output_file, pathlib.Path)
    assert output_file.exists()

    sim.append_to_report(
        """Here is a plot:

![](myplot.png)

And here are the data points:
{% for sample in data_points -%}
- {{ sample }}
{% endfor %}
 """,
        figures=[(MockPlot(), "myplot.png")],
        data_points=[0, 1, 2],
    )

    reference = """# My simulation

Lorem ipsum


The simulation starts at t0=1.0 and lasts 3600.0 seconds.

The seed used for the random number generator is 12345.

[TOC]



Here is a plot:

![](myplot.png)

And here are the data points:
- 0
- 1
- 2
"""

    print(sim.report)
    assert reference.strip() in sim.report.strip()


def test_imo_in_report(tmp_path):
    curpath = pathlib.Path(__file__).parent
    imo = lbs.Imo(flatfile_location=curpath / "test_imo")

    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        name="My simulation",
        description="Lorem ipsum",
        imo=imo,
        random_seed=12345,
    )

    entity_uuid = UUID("dd32cb51-f7d5-4c03-bf47-766ce87dc3ba")
    _ = sim.imo.query(f"/entities/{entity_uuid}")

    quantity_uuid = UUID("e9916db9-a234-4921-adfd-6c3bb4f816e9")
    _ = sim.imo.query(f"/quantities/{quantity_uuid}")

    data_file_uuid = UUID("37bb70e4-29b2-4657-ba0b-4ccefbc5ae36")
    _ = sim.imo.query(f"/data_files/{data_file_uuid}")

    # This data file is an older version of 37bb70e4
    data_file_uuid = UUID("bd8e16eb-2e9d-46dd-a971-f446e953b9dc")
    _ = sim.imo.query(f"/data_files/{data_file_uuid}")

    html_file = sim.flush()
    assert isinstance(html_file, pathlib.Path)
    assert html_file.exists()


def test_parameter_dict(tmp_path):
    from datetime import date

    sim = lbs.Simulation(
        parameters={
            "simulation": {
                "random_seed": 12345,
            },
            "general": {
                "a": 10,
                "b": 20.0,
                "c": False,
                "subtable": {"d": date(2020, 7, 1), "e": "Hello, world!"},
            },
        }
    )

    assert not sim.parameter_file
    assert isinstance(sim.parameters, dict)

    assert "simulation" in sim.parameters
    assert sim.parameters["simulation"]["random_seed"] == 12345

    assert "general" in sim.parameters
    assert sim.parameters["general"]["a"] == 10
    assert sim.parameters["general"]["b"] == 20.0
    assert not sim.parameters["general"]["c"]

    assert "subtable" in sim.parameters["general"]
    assert sim.parameters["general"]["subtable"]["d"] == date(2020, 7, 1)
    assert sim.parameters["general"]["subtable"]["e"] == "Hello, world!"

    try:
        sim = lbs.Simulation(parameter_file="dummy", parameters={"a": 12345})
        assert False, "Simulation object should have asserted"
    except AssertionError:
        pass


def test_parameter_file():
    from datetime import date

    with NamedTemporaryFile(mode="wt", delete=False) as conf_file:
        conf_file_name = conf_file.name
        conf_file.write(
            """# test_parameter_file
[simulation]
start_time = "2020-01-01T00:00:00"
duration_s = 11.0
description = "Dummy description"
random_seed = 12345

[general]
a = 10
b = 20.0
c = false

[general.subtable]
d = 2020-07-01
e = "Hello, world!"
"""
        )

    with TemporaryDirectory() as tmpdirname:
        sim = lbs.Simulation(base_path=tmpdirname, parameter_file=conf_file_name)

        assert isinstance(sim.parameter_file, pathlib.Path)
        assert isinstance(sim.parameters, dict)

        assert "simulation" in sim.parameters
        assert isinstance(sim.start_time, astropy.time.Time)
        assert sim.duration_s == 11.0
        assert sim.description == "Dummy description"
        assert sim.random_seed == 12345

        assert "general" in sim.parameters
        assert sim.parameters["general"]["a"] == 10
        assert sim.parameters["general"]["b"] == 20.0
        assert not sim.parameters["general"]["c"]

        assert "subtable" in sim.parameters["general"]
        assert sim.parameters["general"]["subtable"]["d"] == date(2020, 7, 1)
        assert sim.parameters["general"]["subtable"]["e"] == "Hello, world!"

    # Check that the code does not complain if the output directory is
    # the same as the one containing the parameter file

    sim = lbs.Simulation(
        base_path=Path(conf_file_name).parent, parameter_file=conf_file_name
    )

    os.unlink(conf_file_name)


def test_duration_units_in_parameter_file():
    with NamedTemporaryFile(mode="wt", delete=False) as conf_file:
        conf_file_name = conf_file.name
        conf_file.write(
            """# test_duration_units_in_parameter_file
[simulation]
start_time = "2020-01-01T00:00:00"
duration_s = "1 day"
random_seed = 12345
"""
        )

    with TemporaryDirectory() as tmpdirname:
        sim = lbs.Simulation(base_path=tmpdirname, parameter_file=conf_file_name)

        assert "simulation" in sim.parameters
        assert isinstance(sim.start_time, astropy.time.Time)
        assert sim.duration_s == 86400.0
        assert sim.random_seed == 12345


def test_distribute_observation(tmp_path):
    for dtype in (np.float16, np.float32, np.float64):
        sim = lbs.Simulation(
            base_path=tmp_path / "simulation_dir",
            start_time=1.0,
            duration_s=11.0,
            random_seed=12345,
        )
        det = lbs.DetectorInfo("dummy", sampling_rate_hz=15)
        obs_list = sim.create_observations(
            detectors=[det], num_of_obs_per_detector=5, tod_dtype=dtype
        )

        assert len(obs_list) == 5
        assert int(obs_list[-1].get_times()[-1] - obs_list[0].get_times()[0]) == 10
        assert (
            sum([o.n_samples for o in obs_list])
            == sim.duration_s * det.sampling_rate_hz
        )


def test_distribute_observation_many_tods(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=1.0,
        duration_s=11.0,
        random_seed=12345,
    )
    det = lbs.DetectorInfo("dummy", sampling_rate_hz=15)
    sim.create_observations(
        detectors=[det],
        num_of_obs_per_detector=5,
        tods=[
            lbs.TodDescription(name="tod1", dtype=np.float32, description="TOD 1"),
            lbs.TodDescription(name="tod2", dtype=np.float64, description="TOD 2"),
        ],
    )

    assert sim.get_tod_names() == ["tod1", "tod2"]
    assert sim.get_tod_descriptions() == ["TOD 1", "TOD 2"]
    assert sim.get_tod_dtypes() == [np.float32, np.float64]

    for cur_obs in sim.observations:
        assert "tod1" in dir(cur_obs)
        assert "tod2" in dir(cur_obs)

        assert cur_obs.tod1.shape == cur_obs.tod2.shape
        assert cur_obs.tod1.dtype == np.float32
        assert cur_obs.tod2.dtype == np.float64

    assert len(sim.observations) == 5
    assert (
        int(sim.observations[-1].get_times()[-1] - sim.observations[0].get_times()[0])
        == 10
    )
    assert (
        sum([o.n_samples for o in sim.observations])
        == sim.duration_s * det.sampling_rate_hz
    )


def test_distribute_observation_astropy(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=astropy.time.Time("2020-01-01T00:00:00"),
        duration_s=11.0,
        random_seed=12345,
    )
    det = lbs.DetectorInfo("dummy", sampling_rate_hz=15)
    obs_list = sim.create_observations(detectors=[det], num_of_obs_per_detector=5)

    assert len(obs_list) == 5
    assert int(obs_list[-1].get_times()[-1] - obs_list[0].get_times()[0]) == 10
    assert sum([o.n_samples for o in obs_list]) == sim.duration_s * det.sampling_rate_hz


def test_describe_distribution(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=0.0,
        duration_s=40.0,
        random_seed=12345,
    )
    det = lbs.DetectorInfo("dummy", sampling_rate_hz=10.0)

    sim.create_observations(
        detectors=[det],
        num_of_obs_per_detector=4,
        tods=[
            lbs.TodDescription(name="tod", dtype="float32", description="Signal"),
            lbs.TodDescription(
                name="fg_tod", dtype="float64", description="Foregrounds"
            ),
            lbs.TodDescription(
                name="dipole_tod", dtype="float32", description="Dipole"
            ),
        ],
    )

    for cur_obs in sim.observations:
        assert "tod" in dir(cur_obs)
        assert "fg_tod" in dir(cur_obs)
        assert "dipole_tod" in dir(cur_obs)

    descr = sim.describe_mpi_distribution()

    assert len(descr.detectors) == 1
    assert len(descr.mpi_processes) == lbs.MPI_COMM_WORLD.size

    for mpi_proc in descr.mpi_processes:
        for obs in mpi_proc.observations:
            assert obs.det_names == ["dummy"]
            assert obs.tod_names == ["tod", "fg_tod", "dipole_tod"]
            assert obs.tod_shape == (1, 100)
            assert obs.tod_dtype == ["float32", "float64", "float32"]


def test_profile_information(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=0.0,
        duration_s=61.0,
        random_seed=12345,
        imo=lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION),
    )
    det = lbs.DetectorInfo.from_imo(
        sim.imo,
        "/releases/vPTEP/satellite/LFT/L1-040/000_000_003_QA_040_T/detector_info",
    )

    sim.create_observations(
        detectors=[det], num_of_obs_per_detector=1, split_list_over_processes=False
    )

    sstr = lbs.SpinningScanningStrategy.from_imo(
        sim.imo, "/releases/vPTEP/satellite/scanning_parameters"
    )
    sim.set_scanning_strategy(scanning_strategy=sstr, delta_time_s=0.5)

    instr = lbs.InstrumentInfo.from_imo(
        sim.imo, "/releases/vPTEP/satellite/LFT/instrument_info"
    )
    sim.set_instrument(instr)

    sim.prepare_pointings()

    sim.flush(profile_file_name="profile.json")
    profile_file_path = sim.base_path / "profile.json"
    assert profile_file_path.exists()


def _configure_simulation_for_pointings(
    tmp_path: Path,
    include_hwp: bool,
    store_full_pointings: bool,
    num_of_detectors: int = 1,
    dtype=np.float32,
) -> lbs.Simulation:
    detector_paths = [
        "/releases/vPTEP/satellite/LFT/L1-040/000_000_003_QA_040_T/detector_info",
        "/releases/vPTEP/satellite/LFT/L1-040/000_000_003_QA_040_B/detector_info",
        "/releases/vPTEP/satellite/LFT/L1-040/000_000_004_QB_040_T/detector_info",
        "/releases/vPTEP/satellite/LFT/L1-040/000_000_004_QB_040_B/detector_info",
    ]
    assert num_of_detectors <= len(detector_paths), (
        "num_of_detectors must be ≤ {}".format(len(detector_paths))
    )

    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=0.0,
        duration_s=61.0,
        random_seed=12345,
        imo=lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION),
    )

    detector_list = [
        lbs.DetectorInfo.from_imo(
            sim.imo,
            url=url,
        )
        for url in detector_paths
    ]

    for cur_det in detector_list:
        # Force a round number for the sampling rate, as tests are much easier to write!
        cur_det.sampling_rate_hz = 1.0

    sim.create_observations(
        detectors=detector_list,
        num_of_obs_per_detector=1,
        split_list_over_processes=False,
    )

    if include_hwp:
        hwp = lbs.IdealHWP(
            ang_speed_radpsec=1.0,
            start_angle_rad=5.0,
        )
        sim.set_hwp(hwp)

    sstr = lbs.SpinningScanningStrategy.from_imo(
        sim.imo, "/releases/vPTEP/satellite/scanning_parameters"
    )
    sim.set_scanning_strategy(scanning_strategy=sstr, delta_time_s=0.5)

    instr = lbs.InstrumentInfo.from_imo(
        sim.imo, "/releases/vPTEP/satellite/LFT/instrument_info"
    )
    sim.set_instrument(instr)

    sim.prepare_pointings()

    if store_full_pointings:
        sim.precompute_pointings(pointings_dtype=dtype)

    return sim


def test_smart_pointings_consistency_with_hwp(tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path, include_hwp=True, store_full_pointings=False
    )

    for obs in sim.observations:
        assert obs.pointing_provider is not None

        for det_idx in range(obs.n_detectors):
            (pointings, hwp_angle) = obs.get_pointings(det_idx)
            assert pointings.shape == (obs.n_samples, 3)
            assert hwp_angle.shape == (obs.n_samples,)


def test_smart_pointings_consistency_without_hwp(tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path, include_hwp=False, store_full_pointings=False
    )

    for obs in sim.observations:
        assert obs.pointing_provider is not None

        for det_idx in range(obs.n_detectors):
            (pointings, hwp_angle) = obs.get_pointings(det_idx)
            assert pointings.shape == (obs.n_samples, 3)
            assert hwp_angle is None


def test_smart_pointings_angles(tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path, include_hwp=True, store_full_pointings=False
    )

    assert len(sim.observations) == 1
    obs = sim.observations[0]

    (pointings, hwp_angle) = obs.get_pointings(0)

    np.testing.assert_allclose(
        actual=pointings[0:10, :],
        desired=np.array(
            [
                [1.6580627894, 0.0000000000, -1.57079633],
                [1.6580531309, 0.0040741359, -1.56653329],
                [1.6580241558, 0.0081481804, -1.56227035],
                [1.6579758652, 0.0122220423, -1.55800759],
                [1.6579082610, 0.0162956301, -1.55374512],
                [1.6578213461, 0.0203688526, -1.54948303],
                [1.6577151239, 0.0244416186, -1.54522141],
                [1.6575895986, 0.0285138369, -1.54096035],
                [1.6574447752, 0.0325854166, -1.53669996],
                [1.6572806597, 0.0366562667, -1.53244032],
            ]
        ),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        actual=hwp_angle[0:10],
        desired=np.array(
            [
                -1.2831853071795865,
                -0.28318530717958657,
                0.7168146928204134,
                1.7168146928204133,
                2.7168146928204133,
                -2.5663706143591734,
                -1.5663706143591734,
                -0.5663706143591734,
                0.43362938564082665,
                1.4336293856408266,
            ]
        ),
        rtol=1e-6,
    )


def test_smart_pointings_preallocation_with_hwp(tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path, include_hwp=True, store_full_pointings=False
    )

    # Allocate one buffer for the pointings and one buffer for the HWP angle
    n_samples = sim.observations[0].n_samples
    pointings_buf = np.empty(shape=(n_samples, 3))
    hwp_angle_buf = np.empty(shape=(n_samples, 1))

    for obs in sim.observations:
        for det_idx in range(obs.n_detectors):
            # Force obs.get_pointings to use the buffer we allocated once for all
            # before this double `for` loop
            (pointings, hwp_angle) = obs.get_pointings(
                det_idx, pointing_buffer=pointings_buf, hwp_buffer=hwp_angle_buf
            )

            # numpy.shares_memory() tells if the two arrays use the same memory
            # buffer
            assert np.shares_memory(pointings, pointings_buf)
            assert np.shares_memory(hwp_angle, hwp_angle_buf)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_smart_pointings_store_matrices_without_hwp(dtype, tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path,
        include_hwp=False,
        store_full_pointings=True,
        dtype=dtype,
    )

    for cur_obs in sim.observations:
        assert "pointing_matrix" in dir(cur_obs)

        assert cur_obs.pointing_matrix.dtype == dtype
        assert cur_obs.pointing_matrix.shape == (
            cur_obs.n_detectors,
            cur_obs.n_samples,
            3,
        )
        assert cur_obs.hwp_angle is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_smart_pointings_store_matrices_with_hwp(dtype, tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path,
        include_hwp=True,
        store_full_pointings=True,
        dtype=dtype,
    )

    for cur_obs in sim.observations:
        assert "pointing_matrix" in dir(cur_obs)

        assert cur_obs.pointing_matrix.dtype == dtype
        assert cur_obs.pointing_matrix.shape == (
            cur_obs.n_detectors,
            cur_obs.n_samples,
            3,
        )

        assert cur_obs.hwp_angle is not None
        assert cur_obs.hwp_angle.dtype == dtype
        assert cur_obs.hwp_angle.shape == (cur_obs.n_samples,)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_pointings_for_one_detector(dtype, tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path,
        include_hwp=True,
        store_full_pointings=False,
        num_of_detectors=4,
    )

    for cur_obs in sim.observations:
        pointings, hwp_angle = cur_obs.get_pointings(0, pointings_dtype=dtype)

        assert pointings.dtype == dtype
        assert pointings.shape == (cur_obs.n_samples, 3)

        assert hwp_angle.dtype == dtype
        assert hwp_angle.shape == (cur_obs.n_samples,)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_store_pointings_for_two_detectors(dtype, tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path,
        include_hwp=True,
        store_full_pointings=False,
        num_of_detectors=4,
    )

    for cur_obs in sim.observations:
        for cur_pair in ([1, 3], [0, 2], [1, 2]):
            pointings, hwp_angle = cur_obs.get_pointings(
                cur_pair, pointings_dtype=dtype
            )

            assert pointings.dtype == dtype
            assert pointings.shape == (2, cur_obs.n_samples, 3)

            assert hwp_angle.dtype == dtype
            assert hwp_angle.shape == (cur_obs.n_samples,)

            for rel_det_idx, abs_det_idx in enumerate(cur_pair):
                cur_pointings, _ = cur_obs.get_pointings(abs_det_idx)
                np.testing.assert_allclose(
                    actual=pointings[rel_det_idx, :, :],
                    desired=cur_pointings[:, :],
                    rtol=1e-6,
                )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_smart_pointings_for_all_detectors(dtype, tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path,
        include_hwp=True,
        store_full_pointings=False,
        num_of_detectors=4,
    )

    for cur_obs in sim.observations:
        pointings, hwp_angle = cur_obs.get_pointings("all", pointings_dtype=dtype)

        assert pointings.dtype == dtype
        assert pointings.shape == (4, cur_obs.n_samples, 3)

        assert hwp_angle.dtype == dtype
        assert hwp_angle.shape == (cur_obs.n_samples,)

        for det_idx in range(4):
            cur_pointings, _ = cur_obs.get_pointings(det_idx, pointings_dtype=dtype)

            assert cur_pointings.dtype == dtype
            np.testing.assert_allclose(
                actual=pointings[det_idx, :, :],
                desired=cur_pointings[:, :],
                rtol=1e-6,
            )


def test_get_sky(tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path, include_hwp=True, store_full_pointings=True, num_of_detectors=4
    )

    mbs_params = lbs.MbsParameters(
        make_cmb=True,
        make_fg=True,
        make_noise=False,  # This is detector-dependent
        seed_cmb=1234,
        fg_models=["pysm_dust_0"],
        gaussian_smooth=False,  # This is detector-dependent
        bandpass_int=False,  # This is detector-dependent
        nside=16,
        units="uK_CMB",
        maps_in_ecliptic=False,
    )

    maps = sim.get_sky(parameters=mbs_params, store_in_observation=True)
    maps = np.array([maps[det] for det in set(sim.observations[0].name)])
    obs_maps = sim.observations[0].sky
    obs_maps = np.array([obs_maps[det] for det in set(sim.observations[0].name)])

    np.testing.assert_allclose(maps, obs_maps)

    ChanInfo = lbs.FreqChannelInfo(bandcenter_ghz=40)
    same_ch_map = sim.get_sky(parameters=mbs_params, channels=ChanInfo)[
        ChanInfo.channel.replace(" ", "_")
    ]

    for idx_det in range(4):
        np.testing.assert_allclose(maps[idx_det], same_ch_map)

    # Introduce a difference
    mbs_params.make_noise = True

    maps = sim.get_sky(parameters=mbs_params, store_in_observation=True)
    maps = np.array([maps[det] for det in set(sim.observations[0].name)])

    for idx_det in range(4):
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_allclose,
            maps[idx_det],
            same_ch_map,
        )


def test_convolve_and_filltods_from_obs(tmp_path):
    sim = _configure_simulation_for_pointings(
        tmp_path, include_hwp=True, store_full_pointings=True, num_of_detectors=4
    )

    mbs_params = lbs.MbsParameters(
        make_cmb=True,
        make_fg=True,
        make_noise=False,
        seed_cmb=1234,
        fg_models=["pysm_dust_0"],
        gaussian_smooth=False,
        bandpass_int=False,
        nside=16,
        units="uK_CMB",
        maps_in_ecliptic=False,
        store_alms=True,  # This will produce alms
    )

    maps = sim.get_sky(parameters=mbs_params, store_in_observation=True)
    assert maps["type"] == "alms"

    _ = sim.get_gauss_beam_alms(mbs_params.lmax_alms, store_in_observation=True)

    sim.convolve_sky()
    tod = sim.observations[0].tod[0]

    sim.nullify_tod()

    mbs_params = lbs.MbsParameters(
        make_cmb=True,
        make_fg=True,
        make_noise=False,
        seed_cmb=1234,
        fg_models=["pysm_dust_0"],
        gaussian_smooth=True,
        bandpass_int=False,
        nside=16,
        units="uK_CMB",
        maps_in_ecliptic=False,
        store_alms=False,  # This will produce maps
    )

    maps = sim.get_sky(parameters=mbs_params, store_in_observation=True)
    sim.fill_tods()

    tod_2 = sim.observations[0].tod[0]

    np.testing.assert_allclose(tod, tod_2)


test_convolve_and_filltods_from_obs(Path("test.txt"))
