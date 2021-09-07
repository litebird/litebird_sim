    # -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs
import astropy


def test_spacecraft_orbit():
    start_time = astropy.time.Time("2020-01-01")

    # We simulate two orbits: the first one is the «nominal» orbit, which includes a Lissajous
    # motion around the L2 point, while the second one follows the L2 point (no Lissajous).
    orbit = lbs.SpacecraftOrbit(
        start_time=start_time, solar_velocity_km_s=0, 
    )
    no_lj_orbit = lbs.SpacecraftOrbit(
        start_time=start_time, radius1_km=0.0, radius2_km=0.0, solar_velocity_km_s=0,
    )

    time_span_s = astropy.time.TimeDelta(86400.0 * 365, format="sec").to("s").value
    posvel = lbs.l2_pos_and_vel_in_obs(
        orbit, start_time=start_time, time_span_s=time_span_s,
    )
    posvel_no_lissajous = lbs.l2_pos_and_vel_in_obs(
        no_lj_orbit, start_time=start_time, time_span_s=time_span_s,
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

    # Compute the average distance from the Sun and check that it's approximately 150 Mkm
    assert (
        np.round(np.median(np.linalg.norm(posvel.positions_km, axis=1)) / 1e6) == 151.0
    )
    assert (
        np.round(
            np.median(np.linalg.norm(posvel_no_lissajous.positions_km, axis=1)) / 1e6
        )
        == 151.0
    )

    # Compute the average speed with respect to the Barycentric Ecliptic system and check that it's
    # approximately 30 km/s
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
