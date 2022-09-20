# -*- encoding: utf-8 -*-

import litebird_sim as lbs
import numpy as np
import healpy as hp


def test_coordinates():

    # The purpose of this test is to validate the rotation from
    # ecliptic to galactic coordinates performed by the function
    # in coordinates.py with the same rotation provided by healpy
    # the pointings tested are generated with get_pointings

    start_time = 0
    time_span_s = 10 * 24 * 3600
    sampling_hz = 1

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s)

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
        start_time, time_span_s, delta_time_s=7200
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    det = lbs.DetectorInfo(
        name="Boresight_detector_T",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    (obs,) = sim.create_observations(detectors=[det])

    pointings = lbs.get_pointings(
        obs,
        spin2ecliptic_quats=spin2ecliptic_quats,
        detector_quats=[det.quat],
        bore2spin_quat=instr.bore2spin_quat,
    )

    r = hp.Rotator(coord=["E", "G"])

    pointings_gal_hp = np.empty_like(pointings[0])

    pointings_gal_hp[:, 0:2] = r(pointings[0, :, 0], pointings[0, :, 1]).T
    pointings_gal_hp[:, 2] = pointings[0, :, 2] + r.angle_ref(
        pointings[0, :, 0], pointings[0, :, 1]
    )

    pointings_gal_lbs, psi_gal_lbs = lbs.coordinates.rotate_coordinates_e2g(
        pointings[0, :, 0:2], pointings[0, :, 2]
    )

    np.testing.assert_allclose(
        pointings_gal_hp[:, 0:2], pointings_gal_lbs, rtol=1e-6, atol=1e-6
    )

    np.testing.assert_allclose(
        pointings_gal_hp[:, 2], psi_gal_lbs, rtol=1e-6, atol=1e-6
    )
