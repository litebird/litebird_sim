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


def test_lissajous_velocity_is_derivative_of_position():
    # The velocity returned by compute_lissajous_pos_and_vel must be the time
    # derivative of the position, with the Earth moving along the Ecliptic
    orbit = lbs.SpacecraftOrbit(start_time=astropy.time.Time("2030-01-01"))
    earth_ang_speed_rad_s = 2 * np.pi / (365.25 * 86400.0)
    earth_angle0_rad = 1.3

    def pos_and_vel(time_s):
        return np.array(
            lbs.compute_lissajous_pos_and_vel(
                time0=time_s,
                earth_angle_rad=earth_angle0_rad + earth_ang_speed_rad_s * time_s,
                earth_ang_speed_rad_s=earth_ang_speed_rad_s,
                radius1_km=orbit.radius1_km,
                radius2_km=orbit.radius2_km,
                ang_speed1_rad_s=orbit.ang_speed1_rad_s,
                ang_speed2_rad_s=orbit.ang_speed2_rad_s,
                phase_rad=orbit.phase_rad,
            )
        )

    dt_s = 1.0
    for time_s in np.linspace(0.0, 365 * 86400.0, 20):
        vel_km_s = pos_and_vel(time_s)[3:]
        num_vel_km_s = (
            pos_and_vel(time_s + dt_s)[:3] - pos_and_vel(time_s - dt_s)[:3]
        ) / (2 * dt_s)
        np.testing.assert_allclose(
            vel_km_s, num_vel_km_s, atol=1e-6 * np.linalg.norm(vel_km_s)
        )
