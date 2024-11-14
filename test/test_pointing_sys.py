# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs
import json
import gzip
import tomlkit
from pathlib import Path
import pytest

make_reference_file = False # if True, generate reference file at `path_of_reference`.

telescopes = ["LFT", "MFT", "HFT"]
start_time = 0
time_span_s = 10
sampling_hz = 1
results_dict = {}
rtol = 1e-7
atol = 1e-7
path_of_reference = (
    Path(__file__).parent / "pointing_sys_reference/pointing_sys_reference.json.gz"
)


def load_pointing_sys_reference():
    with gzip.open(path_of_reference, "rt", encoding="utf-8") as f:
        json_str = f.read()
    return json.loads(json_str)


if not make_reference_file:
    result_reference = load_pointing_sys_reference()


def gen_simulation_and_dets(telescope):
    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=None
    )
    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            spin_rate_hz=0.5 / 60.0,
            precession_rate_hz=1.0 / (3.2 * 60 * 60),
        ),
        delta_time_s=1.0 / sampling_hz,
    )
    sim.spin2ecliptic_quats.start_time = np.float64(start_time)
    sim.set_instrument(
        lbs.InstrumentInfo(
            name="mock_LiteBIRD",
            spin_boresight_angle_rad=np.deg2rad(50.0),
        ),
    )
    sim.set_hwp(lbs.IdealHWP(sim.instrument.hwp_rpm * 2 * np.pi / 60))
    dets = []
    path_of_toml = Path(__file__).parent / "pointing_sys_reference/mock_focalplane.toml"
    with open(path_of_toml, "r", encoding="utf-8") as toml_file:
        toml_data = tomlkit.parse(toml_file.read())
        for i in range(len(toml_data[telescope])):
            dets.append(lbs.DetectorInfo.from_dict(toml_data[telescope][f"det_{i:03}"]))
    return sim, dets


@pytest.mark.parametrize("telescope", telescopes)
def test_get_detector_orientation(telescope):
    if telescope == "LFT":
        orient_reference = [0.0, 90.0, 0.0, 90.0, 45.0, 135.0, 45.0, 135.0, 0.0, 90.0]
    elif telescope == "MFT":
        orient_reference = [
            0.0,
            90.0,
            15.0,
            105.0,
            -30.0,
            -120.0,
            -45.0,
            -135.0,
            60.0,
            150.0,
            -75.0,
            -165.0,
            90.0,
            180.0,
        ]
    elif telescope == "HFT":
        orient_reference = [45.0, 135.0, 0.0, 90.0, 45.0, 135.0, 45.0, 135.0]

    _, dets = gen_simulation_and_dets(telescope)
    for i, det in enumerate(dets):
        orient = lbs.get_detector_orientation(det)
        assert np.allclose(orient, np.deg2rad(orient_reference[i]))


@pytest.mark.parametrize("telescope", telescopes)
def test_PointingSys_add_single_offset_to_FP(
    telescope, make_reference_file=make_reference_file
):
    func = test_PointingSys_add_single_offset_to_FP
    sim, dets = gen_simulation_and_dets(telescope)
    (obs,) = sim.create_observations(detectors=dets)

    pointing_sys = lbs.PointingSys(sim, obs, dets)
    single_offset = np.deg2rad(1.0)
    axis = "x"
    pointing_sys.focalplane.add_offset(single_offset, axis)

    lbs.prepare_pointings(
        sim.observations, sim.instrument, sim.spin2ecliptic_quats, hwp=sim.hwp
    )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(
                det_idx, pointings_dtype=np.float32
            )
            pointings_list.append(pointings.tolist())

    if func.__name__ not in results_dict:
        results_dict[func.__name__] = {}

    results_dict[func.__name__][telescope] = pointings_list
    if not make_reference_file:
        np.testing.assert_allclose(
            pointings_list,
            result_reference[func.__name__][telescope],
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("telescope", telescopes)
def test_PointingSys_add_multiple_offsets_to_FP(
    telescope, make_reference_file=make_reference_file
):
    func = test_PointingSys_add_multiple_offsets_to_FP
    sim, dets = gen_simulation_and_dets(telescope)
    (obs,) = sim.create_observations(detectors=dets)

    pointing_sys = lbs.PointingSys(sim, obs, dets)
    multiple_offsets = np.linspace(0, np.deg2rad(1), len(dets))
    axis = "x"
    pointing_sys.focalplane.add_offset(multiple_offsets, axis)

    lbs.prepare_pointings(
        sim.observations, sim.instrument, sim.spin2ecliptic_quats, hwp=sim.hwp
    )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(
                det_idx, pointings_dtype=np.float32
            )
            pointings_list.append(pointings.tolist())

    if func.__name__ not in results_dict:
        results_dict[func.__name__] = {}

    results_dict[func.__name__][telescope] = pointings_list
    if not make_reference_file:
        np.testing.assert_allclose(
            pointings_list,
            result_reference[func.__name__][telescope],
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("telescope", telescopes)
def test_PointingSys_add_uncommon_disturb_to_FP(
    telescope, make_reference_file=make_reference_file
):
    func = test_PointingSys_add_uncommon_disturb_to_FP
    sim, dets = gen_simulation_and_dets(telescope)
    (obs,) = sim.create_observations(detectors=dets)

    nquats = obs.n_samples + 1
    noise_rad_matrix = np.zeros([len(dets), nquats])
    sigmas = np.linspace(0, np.deg2rad(1), len(dets))
    sim.init_random(random_seed=12_345)
    for i in range(len(dets)):
        lbs.add_white_noise(
            noise_rad_matrix[i, :], sigma=np.deg2rad(sigmas[i]), random=sim.random
        )

    pointing_sys = lbs.PointingSys(sim, obs, dets)
    axis = "x"
    pointing_sys.focalplane.add_disturb(noise_rad_matrix, axis)

    lbs.prepare_pointings(
        sim.observations, sim.instrument, sim.spin2ecliptic_quats, hwp=sim.hwp
    )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(
                det_idx, pointings_dtype=np.float32
            )
            pointings_list.append(pointings.tolist())

    if func.__name__ not in results_dict:
        results_dict[func.__name__] = {}

    results_dict[func.__name__][telescope] = pointings_list
    if not make_reference_file:
        np.testing.assert_allclose(
            pointings_list,
            result_reference[func.__name__][telescope],
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("telescope", telescopes)
def test_PointingSys_add_common_disturb_to_FP(
    telescope, make_reference_file=make_reference_file
):
    func = test_PointingSys_add_common_disturb_to_FP
    sim, dets = gen_simulation_and_dets(telescope)
    (obs,) = sim.create_observations(detectors=dets)
    nquats = obs.n_samples + 1
    noise_rad_1d_array = np.zeros(nquats)

    sim.init_random(random_seed=12_345)
    lbs.add_white_noise(noise_rad_1d_array, sigma=np.deg2rad(1), random=sim.random)

    pointing_sys = lbs.PointingSys(sim, obs, dets)
    axis = "x"
    pointing_sys.focalplane.add_disturb(noise_rad_1d_array, axis)
    sim.create_observations(detectors=dets)

    lbs.prepare_pointings(
        sim.observations, sim.instrument, sim.spin2ecliptic_quats, hwp=sim.hwp
    )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(
                det_idx, pointings_dtype=np.float32
            )
            pointings_list.append(pointings.tolist())

    if func.__name__ not in results_dict:
        results_dict[func.__name__] = {}

    results_dict[func.__name__][telescope] = pointings_list
    if not make_reference_file:
        np.testing.assert_allclose(
            pointings_list,
            result_reference[func.__name__][telescope],
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("telescope", telescopes)
def test_PointingSys_add_single_offset_to_spacecraft(
    telescope, make_reference_file=make_reference_file
):
    func = test_PointingSys_add_single_offset_to_spacecraft
    sim, dets = gen_simulation_and_dets(telescope)
    (obs,) = sim.create_observations(detectors=dets)

    pointing_sys = lbs.PointingSys(sim, obs, dets)
    single_offset = np.deg2rad(1.0)
    axis = "x"
    pointing_sys.spacecraft.add_offset(single_offset, axis)


    lbs.prepare_pointings(
        sim.observations, sim.instrument, sim.spin2ecliptic_quats, hwp=sim.hwp
    )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(
                det_idx, pointings_dtype=np.float32
            )
            pointings_list.append(pointings.tolist())

    if func.__name__ not in results_dict:
        results_dict[func.__name__] = {}

    results_dict[func.__name__][telescope] = pointings_list
    if not make_reference_file:
        np.testing.assert_allclose(
            pointings_list,
            result_reference[func.__name__][telescope],
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("telescope", telescopes)
def test_PointingSys_add_common_disturb_to_spacecraft(
    telescope, make_reference_file=make_reference_file
):
    func = test_PointingSys_add_common_disturb_to_spacecraft
    sim, dets = gen_simulation_and_dets(telescope)
    (obs,) = sim.create_observations(detectors=dets)
    nquats = obs.n_samples + 1

    noise_rad_1d_array = np.zeros(nquats)
    sim.init_random(random_seed=12_345)
    lbs.add_white_noise(noise_rad_1d_array, sigma=np.deg2rad(1), random=sim.random)

    pointing_sys = lbs.PointingSys(sim, obs, dets)
    axis = "x"
    pointing_sys.spacecraft.add_disturb(noise_rad_1d_array, axis)


    lbs.prepare_pointings(
        sim.observations, sim.instrument, sim.spin2ecliptic_quats, hwp=sim.hwp
    )

    pointings_list = []
    for cur_obs in sim.observations:
        for det_idx in range(cur_obs.n_detectors):
            pointings, hwp_angle = cur_obs.get_pointings(
                det_idx, pointings_dtype=np.float32
            )
            pointings_list.append(pointings.tolist())

    if func.__name__ not in results_dict:
        results_dict[func.__name__] = {}

    results_dict[func.__name__][telescope] = pointings_list
    if not make_reference_file:
        np.testing.assert_allclose(
            pointings_list,
            result_reference[func.__name__][telescope],
            rtol=rtol,
            atol=atol,
        )


if make_reference_file:
    print("make_reference_file == True: Generating reference file.")
    for telescope in telescopes:
        test_PointingSys_add_single_offset_to_FP(telescope, make_reference_file)
        test_PointingSys_add_multiple_offsets_to_FP(telescope, make_reference_file)
        test_PointingSys_add_uncommon_disturb_to_FP(telescope, make_reference_file)
        test_PointingSys_add_common_disturb_to_FP(telescope, make_reference_file)
        test_PointingSys_add_single_offset_to_spacecraft(telescope, make_reference_file)
        test_PointingSys_add_common_disturb_to_spacecraft(
            telescope, make_reference_file
        )
    with gzip.open(path_of_reference, "wt", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
