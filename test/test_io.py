# -*- encoding: utf-8 -*-
import json

import numpy as np
import litebird_sim as lbs
import h5py

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


def test_write_simple_observation(tmp_path):
    obs = lbs.Observation(
        detectors=3, n_samples_global=10, start_time_global=0.0, sampling_rate_hz=1.0
    )

    files = lbs.write_list_of_observations(obs=obs, path=tmp_path)
    assert len(files) == 1
    assert files[0].exists()

    # Try to open the file to check that it's a real HDF5 file
    with h5py.File(files[0], "r"):
        pass


def test_write_complex_observation(tmp_path):
    start_time = 0
    time_span_s = 60
    sampling_hz = 1

    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=start_time,
        duration_s=time_span_s,
    )
    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
        start_time, time_span_s, delta_time_s=1.0
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    det = lbs.DetectorInfo(
        name="Dummy detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    sim.create_observations(detectors=[det])
    sim.observations[0].tod = np.random.random(sim.observations[0].tod.shape)

    sim.observations[0].pointings = lbs.scanning.get_pointings(
        sim.observations[0],
        spin2ecliptic_quats=spin2ecliptic_quats,
        detector_quats=[det.quat],
        bore2spin_quat=instr.bore2spin_quat,
    )

    file_list = lbs.write_observations(sim=sim)

    assert len(file_list) == 1

    import sys

    with h5py.File(file_list[0], "r") as inpf:
        assert "tod" in inpf
        assert "pointings" in inpf

        tod_dataset = inpf["tod"]
        pointings_dataset = inpf["pointings"]

        assert tod_dataset.shape == (1, 60)
        assert pointings_dataset.shape == (1, 60, 3)

        assert tod_dataset.attrs["start_time"] == 0.0
        assert not tod_dataset.attrs["mjd_time"]
        assert tod_dataset.attrs["sampling_rate_hz"] == sampling_hz
        assert "mpi_rank" in tod_dataset.attrs
        assert "mpi_size" in tod_dataset.attrs

        detectors = tod_dataset.attrs["detectors"]
        assert isinstance(detectors, str)

        det_dictionary = json.loads(detectors)
        assert len(det_dictionary) == 1
        assert det_dictionary[0]["name"] == det.name
        assert det_dictionary[0]["bandcenter_ghz"] == det.bandcenter_ghz
        assert det_dictionary[0]["quat"] == list(det.quat)

        assert np.allclose(tod_dataset, sim.observations[0].tod)
        assert np.allclose(pointings_dataset, sim.observations[0].pointings)
