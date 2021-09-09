# -*- encoding: utf-8 -*-

import litebird_sim as lbs
import numpy as np
import healpy as hp
from numba import njit
from astropy.time import Time
import unittest

@njit
def bin_map(tod, pixel_indexes, binned_map, accum_map, hit_map):
    # This is a helper function that implements a quick-and-dirty mapmaker.
    # We implement here a simple binner that works only in temperature (unlike
    # lbs.make_bin_maps, which solves for the I/Q/U Stokes components and it
    # an overkill here).

    for idx in range(len(accum_map)):
        hit_map[idx] = 0
        accum_map[idx] = 0.0

    for sample_idx, pix_idx in enumerate(pixel_indexes):
        accum_map[pix_idx] += tod[0, sample_idx]
        hit_map[pix_idx] += 1

    for idx in range(len(binned_map)):
        binned_map[idx] = accum_map[idx] / hit_map[idx]


def test_solar_dipole_fit():
    test = unittest.TestCase()

    # The purpose of this test is to simulate the motion of the spacecraft
    # for one year (see `time_span_s`) and produce *two* timelines: the first
    # is associated with variables `*_s_o` and refers to the case of a nonzero
    # velocity of the Solar System, and the second is associated with variables
    # `*_o` and assumes that the reference frame of the Solar System is the
    # same as the CMB's (so that there is no dipole).

    start_time = Time("2022-01-01")
    time_span_s = 365 * 24 * 3600
    nside = 256
    sampling_hz = 1

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s)

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.7853981633974483,
        precession_rate_hz=8.664850513998931e-05,
        spin_rate_hz=0.0008333333333333334,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
        start_time, time_span_s, 100
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

    obs_s_o, = sim.create_observations(detectors=[det])
    obs_o, = sim.create_observations(detectors=[det])

    pointings = lbs.scanning.get_pointings(
        obs_s_o,
        spin2ecliptic_quats=spin2ecliptic_quats,
        detector_quats=[det.quat],
        bore2spin_quat=instr.bore2spin_quat,
    )

    orbit_s_o = lbs.SpacecraftOrbit(obs_s_o.start_time)
    orbit_o = lbs.SpacecraftOrbit(obs_o.start_time, solar_velocity_km_s=0.0)

    assert orbit_s_o.solar_velocity_km_s == 369.8160
    assert orbit_o.solar_velocity_km_s == 0.0

    pos_vel_s_o = lbs.l2_pos_and_vel_in_obs(orbit_s_o, obs_s_o)
    pos_vel_o = lbs.l2_pos_and_vel_in_obs(orbit_o, obs_o)

    assert pos_vel_s_o.velocities_km_s.shape == (366, 3)
    assert pos_vel_o.velocities_km_s.shape == (366, 3)

    lbs.add_dipole_to_observation(
        obs_s_o, pointings, pos_vel_s_o, dipole_type=lbs.DipoleType.LINEAR
    )
    lbs.add_dipole_to_observation(
        obs_o, pointings, pos_vel_o, dipole_type=lbs.DipoleType.LINEAR
    )

    npix = hp.nside2npix(nside)
    pix_indexes = hp.ang2pix(nside, pointings[0, :, 0], pointings[0, :, 1])

    h = np.zeros(npix)
    m = np.zeros(npix)
    map_s_o = np.zeros(npix)
    map_o = np.zeros(npix)

    bin_map(
        tod=obs_s_o.tod,
        pixel_indexes=pix_indexes,
        binned_map=map_s_o,
        accum_map=m,
        hit_map=h,
    )
    import healpy
    healpy.write_map("map_s_o.fits.gz", map_s_o, overwrite=True)

    bin_map(
        tod=obs_o.tod,
        pixel_indexes=pix_indexes,
        binned_map=map_o,
        accum_map=m,
        hit_map=h,
    )
    import healpy
    healpy.write_map("map_o.fits.gz", map_s_o, overwrite=True)

    dip_map = map_s_o - map_o

    assert np.abs(map_s_o.mean() * 1e6) < 1
    assert np.abs(map_o.mean() * 1e6) < 1
    assert np.abs(dip_map.mean() * 1e6) < 1

    mono, dip = hp.fit_dipole(dip_map)

    r = hp.Rotator(coord=["E", "G"])
    l, b = hp.vec2ang(r(dip), lonlat=True)

    # Amplitude, longitude and latitude
    test.assertAlmostEqual(np.sqrt(np.sum(dip ** 2)) * 1e6, 3362.08, 1)
    test.assertAlmostEqual(l[0], 264.021, 1)
    test.assertAlmostEqual(b[0], 48.253, 1)
