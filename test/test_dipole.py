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

    for cur_type, cur_ref in reference.items():
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
    # `lbs.make_binned_map`, which solves for the I/Q/U Stokes components and is
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
    nside = 32
    sampling_hz = 0.1

    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=12345
    )

    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=0.785_398_163_397_448_3,
            precession_rate_hz=8.664_850_513_998_931e-05,
            spin_rate_hz=0.000_833_333_333_333_333_4,
            start_time=start_time,
        ),
        delta_time_s=7200,
    )

    sim.set_instrument(
        lbs.InstrumentInfo(
            boresight_rotangle_rad=0.0,
            spin_boresight_angle_rad=0.872_664_625_997_164_8,
            spin_rotangle_rad=3.141_592_653_589_793,
        )
    )

    det = lbs.DetectorInfo(
        name="Boresight_detector",
        sampling_rate_hz=sampling_hz,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
    )

    (obs_s_o,) = sim.create_observations(detectors=[det])
    (obs_o,) = sim.create_observations(detectors=[det])

    lbs.prepare_pointings(obs_s_o, sim.instrument, sim.spin2ecliptic_quats, hwp=None)
    pointings = obs_s_o.get_pointings("all")[0]

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
    longitude, latitude = hp.vec2ang(r(dip), lonlat=True)

    # Amplitude, longitude and latitude
    test.assertAlmostEqual(np.sqrt(np.sum(dip**2)) * 1e6, 3362.08, delta=1)
    test.assertAlmostEqual(longitude[0], 264.021, 1)
    test.assertAlmostEqual(latitude[0], 48.253, 1)


def test_dipole_list_of_obs(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=Time("2020-01-01T00:00:00"),
        duration_s=100.0,
        random_seed=12345,
    )
    dets = [
        lbs.DetectorInfo(name="A", sampling_rate_hz=1),
        lbs.DetectorInfo(name="B", sampling_rate_hz=1),
    ]

    sim.create_observations(
        detectors=dets,
        num_of_obs_per_detector=2,
    )

    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=0.785_398_163_397_448_3,
            precession_rate_hz=8.664_850_513_998_931e-05,
            spin_rate_hz=0.000_833_333_333_333_333_4,
            start_time=sim.start_time,
        ),
        delta_time_s=60,
    )

    sim.set_instrument(
        lbs.InstrumentInfo(
            boresight_rotangle_rad=0.0,
            spin_boresight_angle_rad=0.872_664_625_997_164_8,
            spin_rotangle_rad=3.141_592_653_589_793,
        )
    )

    lbs.prepare_pointings(
        sim.observations, sim.instrument, sim.spin2ecliptic_quats, hwp=None
    )
    lbs.precompute_pointings(sim.observations)

    orbit = lbs.SpacecraftOrbit(sim.start_time)
    pos_vel = lbs.spacecraft_pos_and_vel(
        orbit, observations=sim.observations, delta_time_s=10.0
    )

    # Just check that the call works
    lbs.add_dipole_to_observations(
        observations=sim.observations,
        pos_and_vel=pos_vel,
        frequency_ghz=np.array([100.0]),
    )


def test_dipole_convolved_through_observations(tmp_path):
    """add_dipole_to_observations must build S-parameters and convolve.

    Exercises the wrapper convolution path (apply_convolution=True), which
    computes BeamSParams.from_beam_alm internally. Checks that:
      * a single beam and a per-detector dict ({"0": beam}) give the same TOD;
      * the convolved TOD differs from the pencil-beam TOD.
    """
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=Time("2020-01-01T00:00:00"),
        duration_s=100.0,
        random_seed=12345,
    )
    sim.create_observations(
        detectors=[lbs.DetectorInfo(name="A", sampling_rate_hz=1, bandcenter_ghz=100.0)],
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=0.785_398_163_397_448_3,
            precession_rate_hz=8.664_850_513_998_931e-05,
            spin_rate_hz=0.000_833_333_333_333_333_4,
            start_time=sim.start_time,
        ),
        delta_time_s=60,
    )
    sim.set_instrument(
        lbs.InstrumentInfo(
            boresight_rotangle_rad=0.0,
            spin_boresight_angle_rad=0.872_664_625_997_164_8,
            spin_rotangle_rad=3.141_592_653_589_793,
        )
    )
    lbs.prepare_pointings(
        sim.observations, sim.instrument, sim.spin2ecliptic_quats, hwp=None
    )
    lbs.precompute_pointings(sim.observations)

    orbit = lbs.SpacecraftOrbit(sim.start_time)
    pos_vel = lbs.spacecraft_pos_and_vel(
        orbit, observations=sim.observations, delta_time_s=10.0
    )

    beam_alm = lbs.gauss_beam_to_alm(
        lmax=32, mmax=32, fwhm_rad=np.deg2rad(30.0), psi_pol_rad=None
    )

    obs = sim.observations[0]

    def run(**kwargs):
        obs.tod[0][:] = 0.0
        lbs.add_dipole_to_observations(
            observations=sim.observations,
            pos_and_vel=pos_vel,
            dipole_type=lbs.DipoleType.QUADRATIC_FROM_LIN_T,
            **kwargs,
        )
        return obs.tod[0].copy()

    tod_pencil = run()
    tod_single = run(apply_convolution=True, beam_alms=beam_alm)
    tod_dict = run(apply_convolution=True, beam_alms={"0": beam_alm})

    # Single beam and per-detector dict must agree exactly.
    np.testing.assert_allclose(tod_single, tod_dict, rtol=1e-12, atol=1e-12)
    # Convolution with a wide beam must change the signal.
    assert not np.allclose(tod_single, tod_pencil)


def test_compute_s_params_from_beam_alm():
    """BeamSParams.from_beam_alm must work with beam alms only."""
    beam_alm = lbs.gauss_beam_to_alm(
        lmax=128,
        mmax=128,
        fwhm_rad=np.deg2rad(1.0),
        psi_pol_rad=None,
    )

    s_params = lbs.BeamSParams.from_beam_alm(beam_alm)

    assert s_params.s_vec.shape == (3,)
    assert s_params.s_mat.shape == (3, 3)
    np.testing.assert_allclose(s_params.s_mat, s_params.s_mat.T, atol=1e-12)


def test_dipole_convolved():
    """A pencil-beam S-parameter set must reproduce QUADRATIC_FROM_LIN_T.

    When the S-parameters correspond to a perfect pencil beam aligned with the
    boresight (S_vec = ẑ, S_mat = diag(0,0,1)), Eq. (C.5) of the NPIPE paper
    reduces to T₀ [β·n̂₀ + q(x) (β·n̂₀)²], which is identical to
    DipoleType.QUADRATIC_FROM_LIN_T evaluated at the same pointing.
    """
    # Same pointings and velocity used in test_dipole_models
    pointings = np.deg2rad(
        np.array(
            [
                [
                    [0, 0, 0],  # north pole
                    [90, 0, 0],  # equator, φ = 0
                    [180, 0, 0],  # south pole
                ]
            ]
        )
    )
    velocity = 299_792.458 * np.array(
        [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]
    )

    # Pencil-beam S-parameters: delta function at the boresight (ẑ in beam frame)
    s_params = lbs.BeamSParams(
        s_vec=np.array([0.0, 0.0, 1.0]),
        s_mat=np.diag([0.0, 0.0, 1.0]),
    )

    tod_quad = np.zeros((1, 3))
    lbs.add_dipole(
        tod_quad,
        pointings,
        velocity,
        t_cmb_k=1.0,
        frequency_ghz=[100],
        dipole_type=lbs.DipoleType.QUADRATIC_FROM_LIN_T,
    )

    tod_conv = np.zeros((1, 3))
    lbs.add_dipole(
        tod_conv,
        pointings,
        velocity,
        t_cmb_k=1.0,
        frequency_ghz=[100],
        dipole_type=lbs.DipoleType.QUADRATIC_FROM_LIN_T,
        s_params=s_params,
    )

    np.testing.assert_allclose(tod_conv, tod_quad, rtol=1e-12, atol=1e-12)


def test_dipole_convolved_beam_suppression():
    """Isotropic beam must suppress the dipole to zero and reduce the quadrupole.

    For a uniform isotropic beam S_vec = 0, so the dipole term vanishes.
    The quadrupole contracts to T₀ q(x) S_ii β_i β_i = T₀ q(x) β² / 3,
    i.e. only the monopole contribution survives, which is independent of pointing.
    """
    pointings = np.deg2rad(
        np.array(
            [
                [
                    [90, 0, 0],  # equator, φ = 0
                    [90, 90, 0],  # equator, φ = 90°
                    [90, 180, 0],  # equator, φ = 180°
                ]
            ]
        )
    )
    velocity = 299_792.458 * np.array(
        [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]
    )

    # Isotropic beam: S_vec = 0, S_mat = I/3
    s_params = lbs.BeamSParams(
        s_vec=np.zeros(3),
        s_mat=np.eye(3) / 3.0,
    )

    tod_conv = np.zeros((1, 3))
    lbs.add_dipole(
        tod_conv,
        pointings,
        velocity,
        t_cmb_k=1.0,
        frequency_ghz=[100],
        dipole_type=lbs.DipoleType.QUADRATIC_FROM_LIN_T,
        s_params=s_params,
    )

    # All samples should give the same value: T₀ q(x) β² / 3
    # (a pointing-independent monopole-like term)
    assert np.allclose(tod_conv[0], tod_conv[0, 0]), (
        "Isotropic beam should give a pointing-independent result"
    )

    # The value should equal T₀ * q(x) * β² / 3
    frequency_hz = 100e9
    x = lbs.H_OVER_K_B * frequency_hz / 1.0  # t_cmb_k = 1.0
    q_x = 0.5 * x * (np.exp(x) + 1) / (np.exp(x) - 1)
    beta_sq = 0.1**2
    expected = q_x * beta_sq / 3.0
    np.testing.assert_allclose(tod_conv[0, 0], expected, rtol=1e-10)


