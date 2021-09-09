# -*- encoding: utf-8 -*-

import litebird_sim as lbs
import numpy as np
import healpy as hp
import astropy
import unittest


def test_solar_dipole_fit():

    test = unittest.TestCase()

    start_time = astropy.time.Time("2022-01-01")
    time_span = 365 * 24 * 3600
    nside = 256
    sampling_hz = 1

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span)

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.7853981633974483,
        precession_rate_hz=8.664850513998931e-05,
        spin_rate_hz=0.0008333333333333334,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
        start_time, time_span, 100
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.8726646259971648,
        spin_rotangle_rad=3.141592653589793,
    )

    det = lbs.DetectorInfo(
        name="Boresight_detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    obs_S_O, = sim.create_observations(detectors=[det])
    obs_O, = sim.create_observations(detectors=[det])

    pointings = lbs.scanning.get_pointings(
        obs_S_O,
        spin2ecliptic_quats=spin2ecliptic_quats,
        detector_quats=[det.quat],
        bore2spin_quat=instr.bore2spin_quat,
    )

    orbit_S_O = lbs.SpacecraftOrbit(obs_S_O.start_time)
    orbit_O = lbs.SpacecraftOrbit(obs_O.start_time, solar_velocity_km_s=0.0)

    assert orbit_S_O.solar_velocity_km_s == 369.8160
    assert orbit_O.solar_velocity_km_s == 0.0

    pos_vel_S_O = lbs.l2_pos_and_vel_in_obs(orbit_S_O, obs_S_O)
    pos_vel_O = lbs.l2_pos_and_vel_in_obs(orbit_O, obs_O)

    assert pos_vel_S_O.velocities_km_s.shape == (366, 3)
    assert pos_vel_O.velocities_km_s.shape == (366, 3)

    lbs.add_dipole_to_observation(
        obs_S_O, pointings, pos_vel_S_O, dipole_type=lbs.DipoleType.LINEAR
    )
    lbs.add_dipole_to_observation(
        obs_O, pointings, pos_vel_O, dipole_type=lbs.DipoleType.LINEAR
    )

    npix = hp.nside2npix(nside)

    # Let's make maps

    h = np.zeros(npix)
    m = np.zeros(npix)
    pixidx = hp.ang2pix(nside, pointings[0, :, 0], pointings[0, :, 1])
    pixel_occurrences = np.bincount(pixidx)
    h[0 : len(pixel_occurrences)] += pixel_occurrences
    for isamp, ipix in enumerate(pixidx):
        m[ipix] += obs_S_O.tod[0, isamp]

    map_S_O = m / h

    h = np.zeros(npix)
    m = np.zeros(npix)
    pixidx = hp.ang2pix(nside, pointings[0, :, 0], pointings[0, :, 1])
    pixel_occurrences = np.bincount(pixidx)
    h[0 : len(pixel_occurrences)] += pixel_occurrences
    for isamp, ipix in enumerate(pixidx):
        m[ipix] += obs_O.tod[0, isamp]

    map_O = m / h

    dip_map = map_S_O - map_O

    assert np.abs(map_S_O.mean() * 1e6) < 1
    assert np.abs(map_O.mean() * 1e6) < 1
    assert np.abs(dip_map.mean() * 1e6) < 1

    mono, dip = hp.fit_dipole(dip_map)

    r = hp.Rotator(coord=["E", "G"])
    l, b = hp.vec2ang(r(dip), lonlat=True)

    # Amplitude, longitude and latitude
    test.assertAlmostEqual(np.sqrt(np.sum(dip ** 2)) * 1e6, 3362.08, 1)
    test.assertAlmostEqual(l[0], 264.021, 1)
    test.assertAlmostEqual(b[0], 48.253, 1)
