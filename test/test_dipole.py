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


def test_compute_s_params_from_beam_alm():
    """compute_s_params_from_beam_alm must work with beam alms only."""
    beam_alm = lbs.gauss_beam_to_alm(
        lmax=128,
        mmax=128,
        fwhm_rad=np.deg2rad(1.0),
        psi_pol_rad=None,
    )

    s_params = lbs.compute_s_params_from_beam_alm(beam_alm)

    assert s_params.s_vec.shape == (3,)
    assert s_params.s_mat.shape == (3, 3)
    np.testing.assert_allclose(s_params.s_mat, s_params.s_mat.T, atol=1e-12)


def test_compute_beam_convolution_data_from_beam_alm():
    """compute_beam_convolution_data_from_beam_alm must work with beam alms only."""
    nside = 32
    beam_alm = lbs.gauss_beam_to_alm(
        lmax=64,
        mmax=64,
        fwhm_rad=np.deg2rad(2.0),
        psi_pol_rad=None,
    )

    beam_data = lbs.compute_beam_convolution_data_from_beam_alm(beam_alm, nside=nside)

    assert beam_data.pixel_vecs.shape == (hp.nside2npix(nside), 3)
    assert beam_data.pixel_weights.shape == (hp.nside2npix(nside),)
    np.testing.assert_allclose(np.sum(beam_data.pixel_weights), 1.0, atol=1e-4)


def test_dipole_convolved():
    """DipoleType.CONVOLVED with a pencil-beam must reproduce QUADRATIC_FROM_LIN_T.

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
                    [0, 0, 0],    # north pole
                    [90, 0, 0],   # equator, φ = 0
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
        dipole_type=lbs.DipoleType.CONVOLVED,
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
                    [90, 0, 0],   # equator, φ = 0
                    [90, 90, 0],  # equator, φ = 90°
                    [90, 180, 0], # equator, φ = 180°
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
        dipole_type=lbs.DipoleType.CONVOLVED,
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


def test_dipole_convolved_total_pencil_beam():
    """Single-pixel beam gives the exact TOTAL_FROM_LIN_T at that pixel's direction.

    For a beam with all weight at pixel p₀ in the beam frame, the convolution
    integral collapses to D(β_beam · n̂_{p₀}).  This is an exact identity that
    holds for any pointing, velocity, and frequency.
    """
    import healpy as hp
    from litebird_sim.dipole import (
        _rotate_velocity_to_beam_frame,
        planck as lbs_planck,
    )

    nside = 16
    npix = hp.nside2npix(nside)
    p0 = 100

    # Build alms for a delta beam centered on pixel p0.
    beam_map = np.zeros((1, npix))
    beam_map[0, p0] = 1.0
    beam_map_hpx = lbs.HealpixMap(values=beam_map, nside=nside)
    beam_alm = lbs.estimate_alm(beam_map_hpx, lmax=3 * nside - 1, mmax=3 * nside - 1)

    beam_data = lbs.compute_beam_convolution_data_from_beam_alm(beam_alm, nside=nside)

    # Arbitrary pointing and velocity.
    theta_p = np.deg2rad(45.0)
    phi_p   = np.deg2rad(30.0)
    psi_p   = np.deg2rad(20.0)
    v_km_s  = np.array([100.0, 50.0, -30.0])

    pointings = np.array([[[theta_p, phi_p, psi_p]]])
    velocity  = v_km_s[np.newaxis, :]

    tod = np.zeros((1, 1))
    lbs.add_dipole(
        tod,
        pointings,
        velocity,
        t_cmb_k=lbs.T_CMB_K,
        frequency_ghz=[100.0],
        dipole_type=lbs.DipoleType.CONVOLVED_TOTAL_FROM_LIN_T,
        beam_conv_data=beam_data,
    )

    # Reference: D(β_beam · n̂_{p₀}) evaluated directly.
    vx, vy, vz = _rotate_velocity_to_beam_frame(theta_p, phi_p, psi_p, v_km_s)
    bx = vx / lbs.C_LIGHT_KM_OVER_S
    by = vy / lbs.C_LIGHT_KM_OVER_S
    bz = vz / lbs.C_LIGHT_KM_OVER_S
    beta_sq = bx**2 + by**2 + bz**2
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)

    n0x, n0y, n0z = hp.pix2vec(nside, p0)  # direction of p0 in beam frame
    mu = bx * n0x + by * n0y + bz * n0z

    nu_hz = 100e9
    x = lbs.H_OVER_K_B * nu_hz / lbs.T_CMB_K
    f_x = x * np.exp(x) / (np.exp(x) - 1)
    planck_t0 = lbs_planck(nu_hz, lbs.T_CMB_K)
    planck_t  = lbs_planck(nu_hz, lbs.T_CMB_K / gamma / (1.0 - mu))
    expected = lbs.T_CMB_K / f_x * (planck_t / planck_t0 - 1.0)

    np.testing.assert_allclose(tod[0, 0], expected, rtol=1e-10)


def test_dipole_convolved_total_vs_quadratic():
    """CONVOLVED_TOTAL_FROM_LIN_T and CONVOLVED differ by a known monopole.

    The full TOTAL_FROM_LIN_T formula includes the relativistic γ factor:

        D_total(μ) ≈ T₀(μ - β²/2 + q(x) μ²) + O(β³)

    whereas CONVOLVED (Eq. C.5) uses QUADRATIC_FROM_LIN_T:

        D_quad(μ) = T₀(μ + q(x) μ²)

    Integrating over the beam: D̃_total - D̃_quad ≈ -T₀β²/2 (constant).
    """
    nside = 64

    # Narrow Gaussian beam, FWHM = 2*sqrt(2*ln2)*sigma = 11.7741...°
    sigma_rad = np.deg2rad(5.0)
    fwhm_rad = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_rad
    beam_alm = lbs.gauss_beam_to_alm(
        lmax=3 * nside - 1,
        mmax=3 * nside - 1,
        fwhm_rad=fwhm_rad,
        psi_pol_rad=None,
    )

    s_params = lbs.compute_s_params_from_beam_alm(beam_alm, nside=nside)
    beam_data = lbs.compute_beam_convolution_data_from_beam_alm(beam_alm, nside=nside)

    # β = 0.001 along x, six equatorial samples.
    beta = 0.001
    v_km_s = lbs.C_LIGHT_KM_OVER_S * beta
    pointings = np.deg2rad(
        np.array(
            [[[90, 0, 0], [90, 60, 0], [90, 120, 0],
              [90, 180, 0], [90, 240, 0], [90, 300, 0]]]
        )
    )
    n_samples = pointings.shape[1]
    velocity  = np.tile([v_km_s, 0.0, 0.0], (n_samples, 1))

    tod_quad = np.zeros((1, n_samples))
    lbs.add_dipole(
        tod_quad, pointings, velocity, t_cmb_k=lbs.T_CMB_K,
        frequency_ghz=[100.0], dipole_type=lbs.DipoleType.CONVOLVED,
        s_params=s_params,
    )

    tod_total = np.zeros((1, n_samples))
    lbs.add_dipole(
        tod_total, pointings, velocity, t_cmb_k=lbs.T_CMB_K,
        frequency_ghz=[100.0],
        dipole_type=lbs.DipoleType.CONVOLVED_TOTAL_FROM_LIN_T,
        beam_conv_data=beam_data,
    )

    diff = tod_total[0] - tod_quad[0]

    # 1. The difference is nearly pointing-independent; the residual variation
    #    is O(β³) ~ T₀β³ ≈ 2.7e-9 K for β = 0.001.
    spread = diff.max() - diff.min()
    assert spread < 10 * lbs.T_CMB_K * beta**3, f"diff spread = {spread:.2e} K"

    # 2. The mean offset equals -T₀β²/2 to better than 0.1%.
    expected_offset = -lbs.T_CMB_K * beta**2 / 2.0
    np.testing.assert_allclose(np.mean(diff), expected_offset, rtol=1e-3)

