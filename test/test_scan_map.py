# -*- encoding: utf-8 -*-

import litebird_sim as lbs
import numpy as np
import healpy as hp


def test_scan_map():

    # The purpose of this test is to simulate the motion of the spacecraft
    # for one year (see `time_span_s`) and produce *two* maps: the first
    # is associated with the Observation `obs1` and is built using
    # `scan_map_in_observations` and `make_bin_map`, the second is associated
    # the Observation `obs2` and is built directly filling in the test
    # `tod`, `psi` and `pixind` and then using `make_bin_map`
    # In the second test `out_map1` is compared with both `out_map2` and the
    # input map. Both simulations use two orthogonal detectors at the boresight
    # and input maps generated with `np.random.normal`.
    # The final test verifies that scan_map with the option `input_map_in_galactic`
    # activated correctly handles internaly the coordinate rotation

    start_time = 0
    time_span_s = 365 * 24 * 3600
    nside = 256
    sampling_hz = 1
    hwp_radpsec = 4.084_070_449_666_731

    npix = hp.nside2npix(nside)

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s)

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
        start_time, time_span_s, delta_time_s=60
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    detT = lbs.DetectorInfo(
        name="Boresight_detector_T",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    detB = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
    )

    np.random.seed(seed=123_456_789)
    maps = np.random.normal(0, 1, (3, npix))

    # This part tests the ecliptic coordinates
    in_map = {"Boresight_detector_T": maps, "Boresight_detector_B": maps}

    (obs1,) = sim.create_observations(detectors=[detT, detB])
    (obs2,) = sim.create_observations(detectors=[detT, detB])

    pointings = lbs.get_pointings(
        obs1,
        spin2ecliptic_quats=spin2ecliptic_quats,
        detector_quats=[detT.quat, detB.quat],
        bore2spin_quat=instr.bore2spin_quat,
    )

    lbs.scan_map_in_observations(
        obs1,
        pointings,
        hwp_radpsec,
        in_map,
        input_map_in_galactic=False,
        fill_psi_and_pixind_in_obs=True,
    )
    out_map1 = lbs.make_bin_map(obs1, nside).T

    times = obs2.get_times()
    obs2.pixind = np.empty(obs2.tod.shape, dtype=np.int32)
    obs2.psi = np.empty(obs2.tod.shape)
    for idet in range(obs2.n_detectors):
        obs2.pixind[idet, :] = hp.ang2pix(
            nside, pointings[idet, :, 0], pointings[idet, :, 1]
        )
        obs2.psi[idet, :] = np.mod(
            pointings[idet, :, 2] + 2 * times * hwp_radpsec, 2 * np.pi
        )

    for idet in range(obs2.n_detectors):
        obs2.tod[idet, :] = (
            maps[0, obs2.pixind[idet, :]]
            + np.cos(2 * obs2.psi[idet, :]) * maps[1, obs2.pixind[idet, :]]
            + np.sin(2 * obs2.psi[idet, :]) * maps[2, obs2.pixind[idet, :]]
        )

    out_map2 = lbs.make_bin_map(obs2, nside).T

    np.testing.assert_allclose(
        out_map1, in_map["Boresight_detector_T"], rtol=1e-6, atol=1e-6
    )

    np.testing.assert_allclose(out_map1, out_map2, rtol=1e-6, atol=1e-6)

    # This part tests the galactic coordinates
    r = hp.Rotator(coord=["E", "G"])
    maps = r.rotate_map_alms(maps, use_pixel_weights=False)

    in_map_G = {"Boresight_detector_T": maps, "Boresight_detector_B": maps}

    (obs1,) = sim.create_observations(detectors=[detT, detB])

    pointings = lbs.get_pointings(
        obs1,
        spin2ecliptic_quats=spin2ecliptic_quats,
        detector_quats=[detT.quat, detB.quat],
        bore2spin_quat=instr.bore2spin_quat,
    )

    lbs.scan_map_in_observations(
        obs1,
        pointings,
        hwp_radpsec,
        in_map_G,
        input_map_in_galactic=True,
        fill_psi_and_pixind_in_obs=True,
    )
    out_map1 = lbs.make_bin_map(obs1, nside).T

    np.testing.assert_allclose(
        out_map1, in_map_G["Boresight_detector_T"], rtol=1e-6, atol=1e-6
    )
