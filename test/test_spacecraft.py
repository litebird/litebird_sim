# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs
import astropy


def test_spacecraft_orbit():
    start_time = astropy.time.Time("2020-01-01")

    # We simulate two orbits: the first one is the «nominal» orbit, which includes
    # a Lissajous motion around the L2 point, while the second one follows the L2
    # point (no Lissajous).
    orbit = lbs.SpacecraftOrbit(start_time=start_time, solar_velocity_km_s=0)
    no_lj_orbit = lbs.SpacecraftOrbit(
        start_time=start_time, radius1_km=0.0, radius2_km=0.0, solar_velocity_km_s=0
    )

    time_span_s = astropy.time.TimeDelta(86400.0 * 365, format="sec").to("s").value
    posvel = lbs.spacecraft_pos_and_vel(
        orbit, start_time=start_time, time_span_s=time_span_s
    )
    posvel_no_lissajous = lbs.spacecraft_pos_and_vel(
        no_lj_orbit, start_time=start_time, time_span_s=time_span_s
    )

    assert posvel.start_time == start_time
    assert posvel_no_lissajous.start_time == start_time

    assert posvel.time_span_s == time_span_s
    assert posvel_no_lissajous.time_span_s == time_span_s

    assert posvel.orbit.radius1_km > 0.0
    assert posvel.orbit.radius2_km > 0.0
    assert posvel_no_lissajous.orbit.radius1_km == 0.0
    assert posvel_no_lissajous.orbit.radius2_km == 0.0

    assert posvel.positions_km.shape == (366, 3)
    assert posvel_no_lissajous.positions_km.shape == (366, 3)

    assert posvel.velocities_km_s.shape == (366, 3)
    assert posvel_no_lissajous.velocities_km_s.shape == (366, 3)

    # Compute the average distance from the Sun and check that
    # it's approximately 150 Mkm
    assert (
        np.round(np.median(np.linalg.norm(posvel.positions_km, axis=1)) / 1e6) == 151.0
    )
    assert (
        np.round(
            np.median(np.linalg.norm(posvel_no_lissajous.positions_km, axis=1)) / 1e6
        )
        == 151.0
    )

    # Compute the average speed with respect to the Barycentric
    # Ecliptic system and check that it's approximately 30 km/s
    assert np.round(np.median(np.linalg.norm(posvel.velocities_km_s, axis=1))) == 30.0
    assert (
        np.round(np.median(np.linalg.norm(posvel_no_lissajous.velocities_km_s, axis=1)))
        == 30.0
    )

    # Now let's check the Lissajous motion
    diff_pos_km = posvel.positions_km - posvel_no_lissajous.positions_km
    diff_vel_km_s = posvel.velocities_km_s - posvel_no_lissajous.velocities_km_s

    # Check that the maximum distance of the spacecraft from L2 is ~300,000 km
    assert np.round(np.max(np.linalg.norm(diff_pos_km, axis=1)) / 1e5) == 3.0

    # Check that the maximum speed of the spacecraft with respect to L2 is ~110 m/s
    assert np.round(np.max(np.linalg.norm(diff_vel_km_s, axis=1) * 1e2)) == 11.0


def test_start_and_span_of_many_observations():
    # The time span must cover all the observations, up to the latest end time
    day_s = 86400.0
    n_procs = lbs.MPI_COMM_WORLD.size
    start_time = astropy.time.Time("2030-01-01")
    sim = lbs.Simulation(
        start_time=start_time, duration_s=4 * n_procs * day_s, random_seed=1
    )
    # Four one-day observations on each MPI process
    sim.create_observations(
        detectors=[lbs.DetectorInfo(name="d", sampling_rate_hz=0.01)],
        num_of_obs_per_detector=4 * n_procs,
    )
    assert len(sim.observations) == 4
    local_start = min(obs.start_time for obs in sim.observations)

    # Pass the observations in reverse order, so that the result does not depend
    # on which one comes first
    obs_start, span_s = lbs.spacecraft.compute_start_and_span_for_obs(
        sim.observations[::-1]
    )
    assert abs((obs_start - local_start).to("s").value) < 1e-3
    np.testing.assert_allclose(span_s, 4 * day_s, rtol=1e-9)

    posvel = lbs.spacecraft_pos_and_vel(
        lbs.SpacecraftOrbit(start_time=start_time),
        observations=sim.observations,
        delta_time_s=day_s,
    )
    np.testing.assert_allclose(posvel.time_span_s, 4 * day_s, rtol=1e-9)


def test_spacecraft_orbit_sub_window():
    # A window starting after the beginning of the orbit (e.g., the observations
    # held by an MPI process other than the first one) must give the same positions
    # and velocities as the corresponding samples of a longer run
    start_time = astropy.time.Time("2030-01-01")
    orbit = lbs.SpacecraftOrbit(start_time=start_time)
    day_s = 86400.0

    full = lbs.spacecraft_pos_and_vel(
        orbit, start_time=start_time, time_span_s=40 * day_s, delta_time_s=day_s
    )
    window = lbs.spacecraft_pos_and_vel(
        orbit,
        start_time=start_time + astropy.time.TimeDelta(20 * day_s, format="sec"),
        time_span_s=10 * day_s,
        delta_time_s=day_s,
    )

    assert window.positions_km.shape == (11, 3)
    np.testing.assert_allclose(window.positions_km, full.positions_km[20:31], rtol=1e-6)
    np.testing.assert_allclose(
        window.velocities_km_s, full.velocities_km_s[20:31], rtol=1e-6
    )
