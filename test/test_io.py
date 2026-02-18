# -*- encoding: utf-8 -*-
import json

import h5py
import numpy as np
from astropy.time import Time as AstroTime

import litebird_sim as lbs

NUMPY_TYPES = [
    (np.float32, "float32"),
    (np.float64, "float64"),
    (np.uint8, "uint8"),
    (np.int16, "int16"),
    (np.int32, "int32"),
    (np.int64, "int64"),
    (bool, "bool"),
]


def test_write_healpix_map_to_hdu():
    for cur_dtype, cur_name in NUMPY_TYPES:
        pixels = np.zeros(12, dtype=cur_dtype)
        hdu = lbs.write_healpix_map_to_hdu(pixels, dtype=cur_dtype, name=cur_name)

        assert hdu.header["EXTNAME"] == cur_name
        assert len(hdu.data.field(0)) == len(pixels)
        assert hdu.data.field(0).dtype == cur_dtype


def test_write_healpix_map(tmp_path):
    for cur_dtype, cur_name in NUMPY_TYPES:
        pixels = np.zeros(12, dtype=cur_dtype)
        filename = tmp_path / f"{cur_name}.fits"
        lbs.write_healpix_map_to_file(filename, pixels, dtype=cur_dtype, name=cur_name)


def __write_complex_observation(
    tmp_path,
    use_mjd: bool,
    gzip_compression: bool,
    save_pointings: bool,
):
    start_time = AstroTime("2021-01-01") if use_mjd else 0
    time_span_s = 60
    sampling_hz = 10

    sim = lbs.Simulation(
        base_path=tmp_path,
        start_time=start_time,
        duration_s=time_span_s,
        random_seed=12345,
    )

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    sim.set_scanning_strategy(
        scanning_strategy=scanning,
        delta_time_s=1.0,
        append_to_report=False,
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )
    sim.set_instrument(instr)

    hwp = lbs.IdealHWP(
        ang_speed_radpsec=1.0,
        start_angle_rad=2.0,
    )
    sim.set_hwp(hwp)

    det = lbs.DetectorInfo(
        name="Dummy detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    sim.create_observations(
        detectors=[det],
        tods=[
            lbs.TodDescription(
                name="tod1",
                units=lbs.Units.K_CMB,
                description="First TOD",
                dtype=np.float64,
            ),
            lbs.TodDescription(
                name="tod2",
                units=lbs.Units.K_CMB,
                description="Second TOD",
                dtype=np.float32,
            ),
        ],
    )

    sim.prepare_pointings(append_to_report=False)

    obs = sim.observations[0]
    obs.tod1[:] = np.random.random(obs.tod1.shape)
    obs.tod2[:] = 1.0

    obs.local_flags = np.zeros(obs.tod1.shape, dtype="uint16")
    obs.local_flags[0, 12:15] = 1

    obs.global_flags = np.zeros(obs.tod1.shape[1], dtype="uint32")
    obs.global_flags[12:15] = 15

    return (
        obs,
        det,
        sim.write_observations(
            subdir_name="",
            gzip_compression=gzip_compression,
            tod_fields=["tod1", "tod2"],
            write_full_pointings=save_pointings,
        ),
    )


def __test_write_complex_observation(tmp_path, use_mjd: bool):
    original_obs, det, file_list = __write_complex_observation(
        tmp_path=tmp_path,
        use_mjd=use_mjd,
        gzip_compression=False,
        save_pointings=True,
    )

    assert len(file_list) == 1

    with h5py.File(file_list[0], "r") as inpf:
        assert "tod1" in inpf
        assert "tod2" in inpf
        assert "pointing_provider_rot_quaternion" in inpf
        assert "pointing_provider_hwp" in inpf
        assert "global_flags" in inpf
        assert "flags_0000" in inpf

        assert "mpi_rank" in inpf.attrs
        assert "mpi_size" in inpf.attrs
        assert "global_index" in inpf.attrs
        assert "local_index" in inpf.attrs

        tod1_dataset = inpf["tod1"]
        tod2_dataset = inpf["tod2"]
        pointing_provider_quat_dataset = inpf["pointing_provider_rot_quaternion"]
        pointings = inpf["pointings"]
        det0_quat_dataset = inpf["rot_quaternion_0000"]
        global_flags = inpf["global_flags"]
        local_flags = inpf["flags_0000"]

        assert tod1_dataset.shape == (1, 600)
        assert tod2_dataset.shape == (1, 600)
        assert pointing_provider_quat_dataset.shape == (61, 4)
        assert pointings.shape == (1, 600, 3)
        assert det0_quat_dataset.shape == (1, 4)
        assert global_flags.shape == (2, 3)
        assert local_flags.shape == (2, 3)

        for cur_dataset, description in [
            (tod1_dataset, "First TOD"),
            (tod2_dataset, "Second TOD"),
        ]:
            if use_mjd:
                assert (
                    AstroTime(cur_dataset.attrs["start_time"], format="mjd")
                    == original_obs.start_time
                )
            else:
                assert cur_dataset.attrs["start_time"] == original_obs.start_time

            assert cur_dataset.attrs["description"] == description
            assert cur_dataset.attrs["mjd_time"] == use_mjd
            assert (
                cur_dataset.attrs["sampling_rate_hz"] == original_obs.sampling_rate_hz
            )

            detectors = cur_dataset.attrs["detectors"]
            assert isinstance(detectors, str)

            det_dictionary = json.loads(detectors)
            assert len(det_dictionary) == 1
            assert det_dictionary[0]["name"] == det.name
            assert det_dictionary[0]["bandcenter_ghz"] == det.bandcenter_ghz
            assert det_dictionary[0]["quat"]["quats"] == det.quat.quats.tolist()
            assert det_dictionary[0]["quat"]["start_time"] == det.quat.start_time
            assert (
                det_dictionary[0]["quat"]["sampling_rate_hz"]
                == det.quat.sampling_rate_hz
            )

        np.testing.assert_allclose(tod1_dataset, original_obs.tod1)
        np.testing.assert_allclose(tod2_dataset, original_obs.tod2)
        np.testing.assert_allclose(
            pointing_provider_quat_dataset,
            original_obs.pointing_provider.bore2ecliptic_quats.quats,
        )

        assert np.all(
            global_flags[:]
            == np.array(
                [[12, 3, 585], [0, 15, 0]],
                dtype="uint32",
            )
        )

        assert np.all(
            local_flags[:]
            == np.array(
                [[12, 3, 585], [0, 1, 0]],
                dtype="uint16",
            )
        )


def test_write_complex_observation_mjd(tmp_path):
    __test_write_complex_observation(tmp_path, use_mjd=True)


def test_write_complex_observation_no_mjd(tmp_path):
    __test_write_complex_observation(tmp_path, use_mjd=False)


def __test_read_complex_observation(tmp_path, use_mjd: bool, gzip_compression: bool):
    original_obs, det, file_list = __write_complex_observation(
        tmp_path,
        use_mjd,
        gzip_compression,
        save_pointings=False,
    )

    observations = lbs.read_list_of_observations(
        file_name_list=tmp_path.glob("*.h5"), tod_fields=["tod1", "tod2"]
    )
    assert len(observations) == 1

    obs = observations[0]
    assert isinstance(obs, lbs.Observation)
    assert obs.start_time == original_obs.start_time
    assert obs.sampling_rate_hz == original_obs.sampling_rate_hz

    # Check that the TodDescription objects have been restored correctly
    assert len(obs.tod_list) == 2
    assert isinstance(obs.tod_list[0], lbs.TodDescription)
    assert obs.tod_list[0].name == "tod1"
    assert obs.tod_list[0].description == "First TOD"
    assert obs.tod_list[0].dtype == np.float32
    assert isinstance(obs.tod_list[1], lbs.TodDescription)
    assert obs.tod_list[1].name == "tod2"
    assert obs.tod_list[1].description == "Second TOD"
    assert obs.tod_list[1].dtype == np.float32

    assert obs.tod1.shape == (1, 600)
    np.testing.assert_allclose(actual=obs.tod1, desired=original_obs.tod1)

    assert obs.pointing_provider.bore2ecliptic_quats.quats.shape == (61, 4)
    np.testing.assert_allclose(
        actual=obs.pointing_provider.bore2ecliptic_quats.quats,
        desired=original_obs.pointing_provider.bore2ecliptic_quats.quats,
    )

    ref_flags = np.zeros(obs.tod1.shape, dtype="uint16")
    ref_flags[0, 12:15] = 1

    assert np.all(ref_flags == obs.local_flags)


def test_read_complex_observation_mjd(tmp_path):
    __test_read_complex_observation(
        tmp_path,
        use_mjd=True,
        gzip_compression=False,
    )


def test_read_complex_observation_no_mjd(tmp_path):
    __test_read_complex_observation(
        tmp_path,
        use_mjd=False,
        gzip_compression=False,
    )


def test_gzip_compression_in_obs(tmp_path):
    __test_read_complex_observation(
        tmp_path,
        use_mjd=False,
        gzip_compression=True,
    )
