import astropy.time
import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim import planets as lbs_planets
from litebird_sim.constants import H_OVER_K_B, T_CMB_K

START_TIME = astropy.time.Time("2032-04-01T00:00:00")

# Leap seconds are unknown in 2032: ERFA warns about a "dubious year"
pytestmark = pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")


def test_planet_temperature():
    # The measurements are reproduced, and kept constant outside their range
    assert lbs.planet_temperature_k("jupiter", 143.0) == pytest.approx(173.6)
    assert lbs.planet_temperature_k("neptune", 40.0) == pytest.approx(117.4)
    assert lbs.planet_temperature_k("uranus", 1000.0) == pytest.approx(66.2)

    # WMAP Rayleigh-Jeans temperatures are converted to thermodynamic ones
    freq_hz = 93.32e9
    t_rj = 120.0
    t = lbs_planets._rj_to_thermodynamic(t_rj, freq_hz / 1e9)
    x = H_OVER_K_B * freq_hz
    assert x / np.expm1(x / t) == pytest.approx(t_rj)


def test_planet_brightness():
    # Monochromatic: B(ν, T) / (∂B/∂T at T_CMB), away from the nodes of the
    # table, where T(ν) has a kink
    freq_hz = 120e9
    t = lbs.planet_temperature_k("mars", 120.0)
    x = H_OVER_K_B * freq_hz
    x_cmb = x / T_CMB_K
    expected = (1 / np.expm1(x / t)) / (
        x_cmb * np.exp(x_cmb) / np.expm1(x_cmb) ** 2 / T_CMB_K
    )
    assert lbs.planet_brightness_k_cmb("mars", 120.0) == pytest.approx(expected)

    # A narrow band gives the same value
    narrow = np.linspace(119.999, 120.001, 11)
    assert lbs.planet_brightness_k_cmb("mars", narrow) == pytest.approx(
        expected, rel=1e-6
    )

    # The weights select the part of the band that counts, and their
    # normalization does not matter
    freqs = np.linspace(119.0, 121.0, 2001)
    weights = np.exp(-(((freqs - 120.0) / 0.05) ** 2))
    actual = lbs.planet_brightness_k_cmb("mars", freqs, 3 * weights)
    assert actual == pytest.approx(expected, rel=1e-6)


def _make_sim(tmp_path, fwhm_arcmin, tods=None):
    sim = lbs.Simulation(
        base_path=tmp_path, start_time=START_TIME, duration_s=6 * 3600.0, random_seed=1
    )
    sim.set_instrument(
        lbs.InstrumentInfo(name="instr", spin_boresight_angle_rad=np.deg2rad(50))
    )

    def quat(theta_deg, phi_deg):
        return lbs.RotQuaternion(
            np.array(lbs.quat_rotation_z(np.deg2rad(phi_deg)))
        ) * lbs.RotQuaternion(np.array(lbs.quat_rotation_y(np.deg2rad(theta_deg))))

    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name=f"det{i}",
                sampling_rate_hz=5.0,
                fwhm_arcmin=fwhm_arcmin,
                bandcenter_ghz=140.0,
                bandwidth_ghz=42.0,
                quat=quat(theta, phi),
            )
            for i, (theta, phi) in enumerate([(0, 0), (8, 0), (8, 120), (5, 240)])
        ],
        **({"tods": tods} if tods else {}),
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45),
            precession_rate_hz=1 / (3.2 * 3600),
            spin_rate_hz=0.05 / 60,
        )
    )
    sim.prepare_pointings()
    return sim


def test_add_planets(tmp_path):
    if lbs.MPI_COMM_WORLD.size > 1:
        pytest.skip("This test uses one observation, which is not split over MPI")

    # Wide beams (5°), so that Jupiter enters the field in a few hours
    fwhm_arcmin = 300.0
    sim = _make_sim(tmp_path, fwhm_arcmin)
    sim.add_planets(planets="jupiter")
    obs = sim.observations[0]
    assert np.any(obs.tod != 0)

    # The signal is nonzero exactly where flag_planets flags samples with the
    # same radius as the truncation of the beam
    sim.flag_planets(planets="jupiter", radius_fwhm=3.0, append_to_report=False)
    np.testing.assert_array_equal(obs.tod != 0, obs.local_flags != 0)

    # Compare with the Gaussian beam evaluated on all the pointings
    pointings, _ = obs.get_pointings("all")
    times_s = np.arange(obs.n_samples) / obs.sampling_rate_hz
    grid_s, rel_pos = lbs_planets.planet_ephemeris(
        "jupiter", sim.pos_and_vel, obs.start_time, obs.n_samples / 5.0
    )
    planet_dir, distance_km = lbs_planets.planet_position(grid_s, rel_pos, times_s)
    vectors = lbs_planets._ang2vec(pointings[..., 0], pointings[..., 1])
    angle = np.arccos(np.clip(np.sum(vectors * planet_dir, axis=-1), -1, 1))
    sigma = np.deg2rad(fwhm_arcmin / 60) / np.sqrt(8 * np.log(2))
    peak = (
        lbs.planet_brightness_k_cmb("jupiter", 140.0)
        * lbs_planets.planet_solid_angle_sr("jupiter", distance_km)
        / (2 * np.pi * sigma**2)
    )
    expected = np.where(
        obs.local_flags != 0, peak * np.exp(-(angle**2) / (2 * sigma**2)), 0.0
    )
    np.testing.assert_allclose(obs.tod, expected, rtol=1e-5, atol=1e-12)


def test_jupiter_peak():
    # Order of magnitude: Jupiter (~40") in a 30' beam at 140 GHz gives
    # tens of mK at the centre of the beam
    sigma = np.deg2rad(0.5) / np.sqrt(8 * np.log(2))
    distance_km = 5.0 * 1.496e8
    peak = (
        lbs.planet_brightness_k_cmb("jupiter", 140.0)
        * lbs_planets.planet_solid_angle_sr("jupiter", distance_km)
        / (2 * np.pi * sigma**2)
    )
    assert 0.01 < peak < 0.1


def test_flag_planets_with_threshold(tmp_path):
    if lbs.MPI_COMM_WORLD.size > 1:
        pytest.skip("This test uses one observation, which is not split over MPI")

    sim = _make_sim(tmp_path, fwhm_arcmin=300.0)
    sim.add_planets(planets="jupiter")
    obs = sim.observations[0]

    # The flags are where the signal added by add_planets exceeds the threshold
    threshold_k_cmb = 1e-2 * obs.tod.max()
    num = sim.flag_planets(
        planets="jupiter", threshold_k_cmb=threshold_k_cmb, append_to_report=False
    )
    bright = obs.tod > threshold_k_cmb
    assert 0 < num["jupiter"] == bright.sum() < (obs.tod != 0).sum()
    np.testing.assert_array_equal(obs.local_flags != 0, bright)

    # Planets fainter than the threshold are not flagged
    obs.local_flags = None
    num = sim.flag_planets(planets="jupiter", threshold_k_cmb=1.0)
    assert num["jupiter"] == 0


def test_add_planets_units_and_bandpass(tmp_path):
    if lbs.MPI_COMM_WORLD.size > 1:
        pytest.skip("This test uses one observation, which is not split over MPI")

    tods = [
        lbs.TodDescription(name=name, units=units, dtype=np.float64, description="")
        for name, units in [
            ("tod", lbs.Units.K_CMB),
            ("tod_uk", lbs.Units.uK_CMB),
            ("tod_band", lbs.Units.K_CMB),
        ]
    ]
    sim = _make_sim(tmp_path, fwhm_arcmin=300.0, tods=tods)
    for component in ("tod", "tod_uk"):
        sim.add_planets(planets="jupiter", component=component)
    sim.add_planets(planets="jupiter", component="tod_band", bandpass_integration=True)
    obs = sim.observations[0]
    assert np.any(obs.tod != 0)

    # The signal is converted into the units of the component
    np.testing.assert_allclose(obs.tod_uk, 1e6 * obs.tod, rtol=1e-10)

    # Without a bandpass from the IMo, the band is a top-hat
    band = lbs.BandPassInfo(140.0, 42.0)
    ratio = lbs.planet_brightness_k_cmb(
        "jupiter", band.freqs_ghz, band.weights
    ) / lbs.planet_brightness_k_cmb("jupiter", 140.0)
    assert ratio != pytest.approx(1.0, rel=1e-3)
    np.testing.assert_allclose(obs.tod_band, ratio * obs.tod, rtol=1e-10)

    # The bandpasses of the detectors (e.g., from the IMo) are used if present
    cosine = lbs.BandPassInfo(140.0, 42.0, bandtype="top-hat-cosine")
    obs.band = [cosine] * obs.n_detectors
    _, peak = lbs_planets.planet_beam_parameters(obs, "jupiter", True)
    _, peak_mono = lbs_planets.planet_beam_parameters(obs, "jupiter", False)
    expected = lbs.planet_brightness_k_cmb(
        "jupiter", cosine.freqs_ghz, cosine.weights
    ) / lbs.planet_brightness_k_cmb("jupiter", 140.0)
    np.testing.assert_allclose(peak / peak_mono, expected)
    assert expected != pytest.approx(ratio, rel=1e-4)
