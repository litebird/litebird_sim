# -*- encoding: utf-8 -*-

from astropy.time import Time
import numpy as np
import litebird_sim as lbs

x = np.array([1.0, 0.0, 0.0])
y = np.array([0.0, 1.0, 0.0])
z = np.array([0.0, 0.0, 1.0])


def test_rotation_handedness():
    result = np.empty(3)

    lbs.rotate_vector(result, *lbs.qrotation_x(np.pi / 2), y)
    assert np.allclose(result, z)
    lbs.rotate_vector(result, *lbs.qrotation_x(np.pi / 2), z)
    assert np.allclose(result, -y)
    lbs.rotate_vector(result, *lbs.qrotation_y(np.pi / 2), x)
    assert np.allclose(result, -z)
    lbs.rotate_vector(result, *lbs.qrotation_y(np.pi / 2), z)
    assert np.allclose(result, x)
    lbs.rotate_vector(result, *lbs.qrotation_z(np.pi / 2), x)
    assert np.allclose(result, y)
    lbs.rotate_vector(result, *lbs.qrotation_z(np.pi / 2), y)
    assert np.allclose(result, -x)


def test_quat_multiply_and_rotations():
    # Simple composition of rotations
    quat = np.array(lbs.qrotation_x(np.pi / 3))
    lbs.quat_right_multiply(quat, *lbs.qrotation_x(np.pi / 3))
    assert np.allclose(quat, np.array(lbs.qrotation_x(2 * np.pi / 3)))

    quat = np.array(lbs.qrotation_y(np.pi / 3))
    lbs.quat_right_multiply(quat, *lbs.qrotation_y(np.pi / 3))
    assert np.allclose(quat, np.array(lbs.qrotation_y(2 * np.pi / 3)))

    quat = np.array(lbs.qrotation_z(np.pi / 3))
    lbs.quat_right_multiply(quat, *lbs.qrotation_z(np.pi / 3))
    assert np.allclose(quat, np.array(lbs.qrotation_z(2 * np.pi / 3)))

    quat = np.array(lbs.qrotation_x(np.pi / 3))
    lbs.quat_left_multiply(quat, *lbs.qrotation_x(np.pi / 3))
    assert np.allclose(quat, np.array(lbs.qrotation_x(2 * np.pi / 3)))

    quat = np.array(lbs.qrotation_y(np.pi / 3))
    lbs.quat_left_multiply(quat, *lbs.qrotation_y(np.pi / 3))
    assert np.allclose(quat, np.array(lbs.qrotation_y(2 * np.pi / 3)))

    quat = np.array(lbs.qrotation_z(np.pi / 3))
    lbs.quat_left_multiply(quat, *lbs.qrotation_z(np.pi / 3))
    assert np.allclose(quat, np.array(lbs.qrotation_z(2 * np.pi / 3)))

    # Now we test more complex compositions

    vec = np.empty(3)

    # Right multiplication
    quat = np.array(lbs.qrotation_y(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.qrotation_x(np.pi / 2))
    lbs.rotate_vector(vec, *quat, y)
    assert np.allclose(vec, x)

    quat = np.array(lbs.qrotation_z(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.qrotation_y(np.pi / 2))
    lbs.rotate_vector(vec, *quat, z)
    assert np.allclose(vec, y)

    quat = np.array(lbs.qrotation_x(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.qrotation_y(np.pi / 2))
    lbs.rotate_vector(vec, *quat, x)
    assert np.allclose(vec, y)

    # Left multiplication
    quat = np.array(lbs.qrotation_y(np.pi / 2))
    lbs.quat_left_multiply(quat, *lbs.qrotation_z(np.pi / 2))
    lbs.rotate_vector(vec, *quat, z)
    assert np.allclose(vec, y)

    quat = np.array(lbs.qrotation_z(np.pi / 2))
    lbs.quat_left_multiply(quat, *lbs.qrotation_y(np.pi / 2))
    lbs.rotate_vector(vec, *quat, y)
    assert np.allclose(vec, z)

    quat = np.array(lbs.qrotation_z(np.pi / 2))
    lbs.quat_left_multiply(quat, *lbs.qrotation_x(np.pi / 2))
    lbs.rotate_vector(vec, *quat, x)
    assert np.allclose(vec, z)


def test_quick_rotations():
    vec = np.empty(3)

    quat = np.array(lbs.qrotation_z(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.qrotation_y(np.pi / 2))
    lbs.rotate_z_vector(vec, *quat)
    assert np.allclose(vec, y)

    quat = np.array(lbs.qrotation_x(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.qrotation_y(np.pi / 2))
    lbs.rotate_x_vector(vec, *quat)
    assert np.allclose(vec, y)

    quat = np.array(lbs.qrotation_y(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.qrotation_x(np.pi / 2))
    lbs.rotate_y_vector(vec, *quat)
    assert np.allclose(vec, x)


def test_compute_pointing_and_polangle():
    quat = np.array(lbs.qrotation_y(np.pi / 2))
    result = np.empty(3)
    lbs.compute_pointing_and_polangle(result, quat)
    assert np.allclose(result, [np.pi / 2, 0.0, -np.pi / 2])

    # We stay along the same pointing, but we're rotating the detector
    # by 90°, so the polarization angle is the only number that
    # changes
    lbs.quat_left_multiply(quat, *lbs.qrotation_x(np.pi / 4))
    lbs.compute_pointing_and_polangle(result, quat)
    assert np.allclose(result, [np.pi / 2, 0.0, -np.pi / 4])


def test_boresight_to_ecliptic():
    result = np.empty(4)

    lbs.boresight_to_ecliptic(
        result=result,
        sun_earth_angle_rad=0.0,
        spin_sun_angle_rad=0.0,
        spin_boresight_angle_rad=0.0,
        precession_rate_hz=0.0,
        spin_rate_hz=0.0,
        time_s=0.0,
    )
    assert np.allclose(result, np.array(lbs.qrotation_y(np.pi / 2)))

    for sun_earth_angle_rad in (0.1, np.pi / 2, -0.1):
        lbs.boresight_to_ecliptic(
            result=result,
            sun_earth_angle_rad=sun_earth_angle_rad,
            spin_sun_angle_rad=0.0,
            spin_boresight_angle_rad=0.0,
            precession_rate_hz=0.0,
            spin_rate_hz=0.0,
            time_s=0.0,
        )
        expected = np.array(lbs.qrotation_y(np.pi / 2))
        lbs.quat_left_multiply(expected, *lbs.qrotation_z(sun_earth_angle_rad))
        assert np.allclose(result, expected)


def test_calculate_sun_earth_angles_rad():
    # This is not the date of the 2020 Summer Solstice, as we're using
    # the barycentric mean Ecliptic frame centered on the Sun-Earth
    # barycentre, whose x and y axes are slightly tilted wrt the
    # reference frame used to compute equinoxes (which use the center
    # of the Sun instead of the barycentre)
    time = Time("2020-06-21T12:02:47")
    assert np.allclose(lbs.calculate_sun_earth_angles_rad(time), -np.pi)


def create_fake_detector(sampling_rate_hz=1):
    return lbs.Detector(
        name="dummy",
        wafer="",
        pixel="",
        pixtype="",
        channel="",
        sampling_rate_hz=sampling_rate_hz,
        fwhm_arcmin=3600.0,
        ellipticity=1.0,
        net_ukrts=1.0,
        fknee_mhz=1.0,
        fmin_hz=1.0,
        alpha=1.0,
        pol="",
        orient="",
        quat=np.array([0.0, 0.0, 0.0, 0.0]),
    )


def test_simulation_pointings_still():
    sim = lbs.Simulation()
    fakedet = create_fake_detector(sampling_rate_hz=1 / 3600)

    sim.create_observations(
        detectors=[fakedet],
        num_of_obs_per_detector=1,
        start_time=0.0,
        duration_s=86400.0,
        distribute=False,
    )

    # The spacecraft stands still in L2, with no spinning nor precession
    sstr = lbs.ScanningStrategy(
        spin_sun_angle_deg=0.0,
        spin_boresight_angle_deg=0.0,
        precession_period_min=0.0,
        spin_rate_rpm=0.0,
    )
    sim.generate_pointing_information(sstr, delta_time_s=60.0)

    assert len(sim.observations) == 1
    obs = sim.observations[0]
    assert obs.bore2ecliptic_quats.shape == (24 * 60 + 1, 4)

    # Move the Z vector manually using the last quaternion and check
    # that it's rotated by 1/365.25 of a complete circle
    boresight = np.empty(3)
    lbs.rotate_z_vector(boresight, *obs.bore2ecliptic_quats[-1, :])
    assert np.allclose(np.arctan2(boresight[1], boresight[0]), 2 * np.pi / 365.25)

    # Now redo the calculation using Observation.get_pointings
    pointings_and_polangle = obs.get_pointings(np.array([0.0, 0.0, 0.0, 1.0]))

    colatitude = pointings_and_polangle[:, 0]
    longitude = pointings_and_polangle[:, 1]
    polangle = pointings_and_polangle[:, 2]

    assert np.allclose(colatitude, np.pi / 2)
    assert np.allclose(np.abs(polangle), np.pi / 2)

    # The longitude should have changed by a fraction 23 hours /
    # 365.25 days of a complete circle (we have 24 samples, from t = 0
    # to t = 23 hr)
    assert np.allclose(
        np.abs(longitude[-1] - longitude[0]), 2 * np.pi * 23 / 365.25 / 24
    )


def test_simulation_pointings_polangle():
    sim = lbs.Simulation()
    fakedet = create_fake_detector(sampling_rate_hz=50.0)

    sim.create_observations(
        detectors=[fakedet],
        num_of_obs_per_detector=1,
        start_time=0.0,
        duration_s=61.0,
        distribute=False,
    )

    sstr = lbs.ScanningStrategy(
        spin_sun_angle_deg=0.0,
        spin_boresight_angle_deg=0.0,
        precession_period_min=0.0,
        spin_rate_rpm=1.0,
    )
    sim.generate_pointing_information(sstr, delta_time_s=0.5)

    assert len(sim.observations) == 1
    obs = sim.observations[0]
    pointings_and_polangle = obs.get_pointings(np.array([0.0, 0.0, 0.0, 1.0]))
    polangle = pointings_and_polangle[:, 2]

    # Check that the polarization angle scans every value between -π
    # and +π
    assert np.allclose(np.max(polangle), np.pi, atol=0.01)
    assert np.allclose(np.min(polangle), -np.pi, atol=0.01)


def test_simulation_pointings_spinning():
    sim = lbs.Simulation()
    fakedet = create_fake_detector(sampling_rate_hz=50.0)

    sim.create_observations(
        detectors=[fakedet],
        num_of_obs_per_detector=1,
        start_time=0.0,
        duration_s=61.0,
        distribute=False,
    )

    sstr = lbs.ScanningStrategy(
        spin_sun_angle_deg=0.0,
        spin_boresight_angle_deg=15.0,
        precession_period_min=0.0,
        spin_rate_rpm=1.0,
    )
    sim.generate_pointing_information(sstr, delta_time_s=0.5)

    assert len(sim.observations) == 1
    obs = sim.observations[0]
    pointings_and_polangle = obs.get_pointings(np.array([0.0, 0.0, 0.0, 1.0]))
    colatitude = pointings_and_polangle[:, 0]

    # Check that the colatitude does not depart more than ±15° from
    # the Ecliptic
    assert np.allclose(np.rad2deg(np.max(colatitude)), 90 + 15, atol=0.01)
    assert np.allclose(np.rad2deg(np.min(colatitude)), 90 - 15, atol=0.01)


def test_simulation_pointings_mjd():
    sim = lbs.Simulation()
    fakedet = create_fake_detector()

    sim.create_observations(
        detectors=[fakedet],
        num_of_obs_per_detector=2,
        start_time=Time("2020-01-01T00:00:00"),
        duration_s=130.0,
        distribute=False,
    )

    sstr = lbs.ScanningStrategy(
        spin_sun_angle_deg=10.0,
        spin_boresight_angle_deg=20.0,
        precession_period_min=10.0,
        spin_rate_rpm=0.1,
    )
    sim.generate_pointing_information(sstr)

    for obs in sim.observations:
        assert obs.bore2ecliptic_quats is not None
        pointings_and_polangle = obs.get_pointings(np.array([0.0, 0.0, 0.0, 1.0]))
        print(pointings_and_polangle)