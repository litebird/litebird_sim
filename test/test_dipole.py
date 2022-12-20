# -*- encoding: utf-8 -*-

from pathlib import Path
import litebird_sim as lbs
import numpy as np
import healpy as hp
from numba import njit
from astropy.time import Time
import unittest


def test_dipole_models():
    pointings = np.deg2rad(
        np.array(
            [
                [
                    [0, 0, 0],  # Theta, phi, psi for the #1 sample
                    [90, 0, 0],  # Theta, phi, psi for the #2 sample
                    [180, 0, 0],  # Theta, phi, psi for the #3 sample
                ]
            ]
        )
    )
    tod = np.empty((1, 3))

    # These velocities are expressed as a fraction of the speed of light
    velocity = 299_792.458 * np.array(
        [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]
    )

    # All these numbers have been calculated using a Mathematica notebook
    reference = {
        lbs.DipoleType.LINEAR: [[0.0, 0.1, 0.0]],
        lbs.DipoleType.QUADRATIC_EXACT: [[0.0, 0.11, 0.0]],
        lbs.DipoleType.TOTAL_EXACT: [[-0.005_012_56, 0.105_542, -0.005_012_56]],
        lbs.DipoleType.QUADRATIC_FROM_LIN_T: [[0.0, 0.124_395, 0.0]],
        lbs.DipoleType.TOTAL_FROM_LIN_T: [[-0.004_976, 0.121_683, -0.004_976]],
    }

    for (cur_type, cur_ref) in reference.items():
        tod[:] = 0.0
        lbs.add_dipole(
            tod,
            pointings,
            velocity,
            t_cmb_k=1.0,
            frequency_ghz=[100],
            dipole_type=cur_type,
        )
        np.testing.assert_allclose(tod, cur_ref, rtol=1e-6, atol=1e-6)


@njit
def bin_map(tod, pixel_indexes, binned_map, accum_map, hit_map):
    # This is a helper function that implements a quick-and-dirty mapmaker.
    # We implement here a simple binner that works only in temperature (unlike
    # lbs.make_bin_maps, which solves for the I/Q/U Stokes components and is
    # an overkill here).

    for idx in range(len(accum_map)):
        hit_map[idx] = 0
        accum_map[idx] = 0.0

    for sample_idx, pix_idx in enumerate(pixel_indexes):
        accum_map[pix_idx] += tod[0, sample_idx]
        hit_map[pix_idx] += 1

    for idx in range(len(binned_map)):
        if hit_map[idx] > 0:
            binned_map[idx] = accum_map[idx] / hit_map[idx]
        else:
            binned_map[idx] = np.nan


def test_solar_dipole_fit(tmpdir):
    tmpdir = Path(tmpdir)

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
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.set_scanning_strategy(
        start_time, time_span_s, delta_time_s=7200
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    det = lbs.DetectorInfo(
        name="Boresight_detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    (obs_s_o,) = sim.create_observations(detectors=[det])
    (obs_o,) = sim.create_observations(detectors=[det])

    pointings = lbs.get_pointings(
        obs_s_o,
        spin2ecliptic_quats=spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
    )

    orbit_s_o = lbs.SpacecraftOrbit(obs_s_o.start_time)
    orbit_o = lbs.SpacecraftOrbit(obs_o.start_time, solar_velocity_km_s=0.0)

    assert orbit_s_o.solar_velocity_km_s == 369.8160
    assert orbit_o.solar_velocity_km_s == 0.0

    pos_vel_s_o = lbs.spacecraft_pos_and_vel(orbit_s_o, obs_s_o, delta_time_s=86400.0)
    pos_vel_o = lbs.spacecraft_pos_and_vel(orbit_o, obs_o, delta_time_s=86400.0)

    assert pos_vel_s_o.velocities_km_s.shape == (367, 3)
    assert pos_vel_o.velocities_km_s.shape == (367, 3)

    lbs.add_dipole_to_observations(
        obs_s_o, pos_vel_s_o, dipole_type=lbs.DipoleType.LINEAR
    )
    lbs.add_dipole_to_observations(
        obs_o, pos_vel_o, pointings=pointings, dipole_type=lbs.DipoleType.LINEAR
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

    healpy.write_map(tmpdir / "map_s_o.fits.gz", map_s_o, overwrite=True)

    bin_map(
        tod=obs_o.tod,
        pixel_indexes=pix_indexes,
        binned_map=map_o,
        accum_map=m,
        hit_map=h,
    )

    healpy.write_map(tmpdir / "map_o.fits.gz", map_o, overwrite=True)

    dip_map = map_s_o - map_o

    assert np.abs(np.nanmean(map_s_o) * 1e6) < 1
    assert np.abs(np.nanmean(map_o) * 1e6) < 1
    assert np.abs(np.nanmean(dip_map) * 1e6) < 1

    dip_map[np.isnan(dip_map)] = healpy.UNSEEN
    mono, dip = hp.fit_dipole(dip_map)

    r = hp.Rotator(coord=["E", "G"])
    l, b = hp.vec2ang(r(dip), lonlat=True)

    # Amplitude, longitude and latitude
    test.assertAlmostEqual(np.sqrt(np.sum(dip**2)) * 1e6, 3362.08, 1)
    test.assertAlmostEqual(l[0], 264.021, 1)
    test.assertAlmostEqual(b[0], 48.253, 1)


def test_dipole_list_of_obs(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=Time("2020-01-01T00:00:00"),
        duration_s=100.0,
    )
    dets = [
        lbs.DetectorInfo(name="A", sampling_rate_hz=1),
        lbs.DetectorInfo(name="B", sampling_rate_hz=1),
    ]

    sim.create_observations(
        detectors=dets,
        num_of_obs_per_detector=2,
    )

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=sim.start_time,
    )

    spin2ecliptic_quats = scanning.set_scanning_strategy(
        sim.start_time,
        sim.duration_s,
        delta_time_s=60,
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    pointings = lbs.get_pointings_for_observations(
        sim.observations,
        spin2ecliptic_quats=spin2ecliptic_quats,
        bore2spin_quat=instr.bore2spin_quat,
    )

    orbit = lbs.SpacecraftOrbit(sim.start_time)
    pos_vel = lbs.spacecraft_pos_and_vel(orbit, obs=sim.observations, delta_time_s=10.0)

    # Just check that the call works
    lbs.add_dipole_to_observations(
        obs=sim.observations,
        pos_and_vel=pos_vel,
        pointings=pointings,
        frequency_ghz=[100.0],
    )
