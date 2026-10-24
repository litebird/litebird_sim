import astropy.time
import numpy as np
import pytest
from astropy.coordinates import GeocentricMeanEcliptic, get_body

import litebird_sim as lbs
from litebird_sim import flags as lbs_flags

START_TIME = astropy.time.Time("2032-04-01T00:00:00")

# Leap seconds are unknown in 2032: ERFA warns about a "dubious year"
pytestmark = pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")


def test_flag_planet():
    # Two detectors scanning the Ecliptic, the planet at φ = 0
    n_samples = 21
    phi = np.deg2rad(np.linspace(-5.0, 5.0, n_samples))
    pointings = np.zeros((2, n_samples, 3))
    pointings[:, :, 0] = np.pi / 2
    pointings[:, :, 1] = phi
    planet_dir = np.tile([1.0, 0.0, 0.0], (n_samples, 1))
    radius_rad = np.deg2rad([1.1, 2.1])

    flags = np.zeros((2, n_samples), dtype=lbs.FLAG_DTYPE)
    flags[:, 0] = 4  # Bits set by others must survive
    num = lbs.flag_planet(flags, pointings, planet_dir, radius_rad)

    close = np.abs(phi)[None, :] < radius_rad[:, None]
    assert num == close.sum() == 5 + 9
    np.testing.assert_array_equal(flags & lbs.FLAG_PLANET, close)
    assert np.all(flags[:, 0] == 4)


def test_planet_direction():
    # Seen from L2, Neptune is where astropy puts it for an observer on the
    # Earth, apart from a parallax of ~0.02°
    orbit = lbs.SpacecraftOrbit(START_TIME)
    pos_and_vel = lbs.spacecraft_pos_and_vel(
        orbit, start_time=START_TIME, time_span_s=86400.0
    )
    grid_s, rel_pos = lbs_flags._planet_ephemeris(
        "neptune", pos_and_vel, START_TIME, 86400.0
    )
    times_s = np.array([0.0, 43200.0])
    actual = lbs_flags._planet_directions(grid_s, rel_pos, times_s)

    coords = get_body(
        "neptune", START_TIME + astropy.time.TimeDelta(times_s, format="sec")
    ).transform_to(GeocentricMeanEcliptic())
    desired = coords.cartesian.xyz.value.T
    desired /= np.linalg.norm(desired, axis=1, keepdims=True)

    angle_deg = np.rad2deg(np.arccos(np.clip(np.sum(actual * desired, 1), -1, 1)))
    assert np.all(angle_deg < 0.05)


def _make_sim(tmp_path):
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
                fwhm_arcmin=30.0,
                quat=quat(theta, phi),
            )
            for i, (theta, phi) in enumerate([(0, 0), (8, 0), (8, 120), (5, 240)])
        ]
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


def test_flag_planets_matches_brute_force(tmp_path):
    # The prefilter on the boresight must not miss any sample: compare with
    # the flags computed on all the pointings
    if lbs.MPI_COMM_WORLD.size > 1:
        pytest.skip("This test uses one observation, which is not split over MPI")

    sim = _make_sim(tmp_path)
    radius_fwhm = 20.0  # 10°, so that Jupiter and Uranus enter the field
    num = sim.flag_planets(radius_fwhm=radius_fwhm)
    assert num["jupiter"] > 0 and num["uranus"] > 0

    obs = sim.observations[0]
    reference = np.zeros_like(obs.local_flags)
    pointings, _ = obs.get_pointings("all")
    times_s = np.arange(obs.n_samples) / obs.sampling_rate_hz
    radius_rad = np.deg2rad(radius_fwhm * obs.fwhm_arcmin / 60)
    for planet in lbs.PLANETS:
        grid_s, rel_pos = lbs_flags._planet_ephemeris(
            planet, sim.pos_and_vel, obs.start_time, obs.n_samples / 5.0
        )
        planet_dir = lbs_flags._planet_directions(grid_s, rel_pos, times_s)
        assert (
            lbs.flag_planet(reference, pointings, planet_dir, radius_rad)
            == (num[planet])
        )

    np.testing.assert_array_equal(obs.local_flags, reference)

    # A different radius for each planet
    obs.local_flags = None
    num = sim.flag_planets(radius_fwhm=dict.fromkeys(lbs.PLANETS, 20.0) | {"uranus": 3})
    assert num["jupiter"] > 0 and num["uranus"] == 0


def test_flag_planets_needs_astropy_times():
    sim = lbs.Simulation(start_time=0.0, duration_s=10.0, random_seed=1)
    sim.create_observations(
        detectors=[lbs.DetectorInfo(name="a", sampling_rate_hz=1.0, fwhm_arcmin=30.0)]
    )
    with pytest.raises(ValueError, match="astropy"):
        lbs.flag_planets_in_observations(sim.observations, pos_and_vel=None)
    with pytest.raises(ValueError, match="astropy"):
        sim.flag_planets()
