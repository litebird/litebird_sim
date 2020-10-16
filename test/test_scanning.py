# -*- encoding: utf-8 -*-

from astropy.time import Time
import numpy as np
import litebird_sim as lbs


def test_compute_pointing_and_polangle():
    quat = np.array(lbs.quat_rotation_y(np.pi / 2))
    result = np.empty(3)
    lbs.compute_pointing_and_polangle(result, quat)
    assert np.allclose(result, [np.pi / 2, 0.0, -np.pi / 2])

    # We stay along the same pointing, but we're rotating the detector
    # by 90°, so the polarization angle is the only number that
    # changes
    lbs.quat_left_multiply(quat, *lbs.quat_rotation_x(np.pi / 4))
    lbs.compute_pointing_and_polangle(result, quat)
    assert np.allclose(result, [np.pi / 2, 0.0, -np.pi / 4])


def test_spin_to_ecliptic():
    result = np.empty(4)

    lbs.spin_to_ecliptic(
        result=result,
        sun_earth_angle_rad=0.0,
        spin_sun_angle_rad=0.0,
        precession_rate_hz=0.0,
        spin_rate_hz=0.0,
        time_s=0.0,
    )
    assert np.allclose(result, np.array(lbs.quat_rotation_y(np.pi / 2)))

    for sun_earth_angle_rad in (0.1, np.pi / 2, -0.1):
        lbs.spin_to_ecliptic(
            result=result,
            sun_earth_angle_rad=sun_earth_angle_rad,
            spin_sun_angle_rad=0.0,
            precession_rate_hz=0.0,
            spin_rate_hz=0.0,
            time_s=0.0,
        )
        expected = np.array(lbs.quat_rotation_y(np.pi / 2))
        lbs.quat_left_multiply(expected, *lbs.quat_rotation_z(sun_earth_angle_rad))
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
    return lbs.DetectorInfo(
        name="dummy",
        sampling_rate_hz=sampling_rate_hz,
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
    )


def test_simulation_pointings_still():
    sim = lbs.Simulation(start_time=0.0, duration_s=86400.0)
    fakedet = create_fake_detector(sampling_rate_hz=1 / 3600)

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=1, distribute=False,
    )
    assert len(sim.observations) == 1
    obs = sim.observations[0]

    # The spacecraft stands still in L2, with no spinning nor precession
    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=0.0,
    )
    sim.generate_spin2ecl_quaternions(sstr, delta_time_s=60.0)
    assert sim.spin2ecliptic_quats.quats.shape == (24 * 60 + 1, 4)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=0.0)

    # Move the Z vector manually using the last quaternion and check
    # that it's rotated by 1/365.25 of a complete circle
    boresight = np.empty(3)
    lbs.rotate_z_vector(boresight, *sim.spin2ecliptic_quats.quats[-1, :])
    assert np.allclose(np.arctan2(boresight[1], boresight[0]), 2 * np.pi / 365.25)

    # Now redo the calculation using Observation.get_pointings
    pointings_and_polangle = obs.get_pointings(
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        detector_quats=np.array([[0.0, 0.0, 0.0, 1.0]]),
        bore2spin_quat=instr.bore2spin_quat,
    )

    colatitude = pointings_and_polangle[..., 0]
    longitude = pointings_and_polangle[..., 1]
    polangle = pointings_and_polangle[..., 2]

    assert np.allclose(colatitude, np.pi / 2), colatitude
    assert np.allclose(np.abs(polangle), np.pi / 2), polangle

    # The longitude should have changed by a fraction 23 hours /
    # 365.25 days of a complete circle (we have 24 samples, from t = 0
    # to t = 23 hr)
    assert np.allclose(
        np.abs(longitude[..., -1] - longitude[..., 0]), 2 * np.pi * 23 / 365.25 / 24
    )


def test_simulation_pointings_polangle(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir", start_time=0.0, duration_s=61.0,
    )
    fakedet = create_fake_detector(sampling_rate_hz=50.0)

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=1, distribute=False,
    )
    assert len(sim.observations) == 1
    obs = sim.observations[0]

    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=1.0 / 60,
    )
    sim.generate_spin2ecl_quaternions(scanning_strategy=sstr, delta_time_s=0.5)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=0.0)

    pointings_and_polangle = obs.get_pointings(
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        detector_quats=np.array([[0.0, 0.0, 0.0, 1.0]]),
        bore2spin_quat=instr.bore2spin_quat,
    )
    polangle = pointings_and_polangle[..., 2]

    # Check that the polarization angle scans every value between -π
    # and +π
    assert np.allclose(np.max(polangle), np.pi, atol=0.01)
    assert np.allclose(np.min(polangle), -np.pi, atol=0.01)

    # Simulate the generation of a report
    sim.flush()


def test_simulation_pointings_spinning(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir", start_time=0.0, duration_s=61.0,
    )
    fakedet = create_fake_detector(sampling_rate_hz=50.0)

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=1, distribute=False,
    )
    assert len(sim.observations) == 1
    obs = sim.observations[0]

    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=1.0,
    )
    sim.generate_spin2ecl_quaternions(scanning_strategy=sstr, delta_time_s=0.5)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(15.0))

    pointings_and_polangle = obs.get_pointings(
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        detector_quats=np.array([[0.0, 0.0, 0.0, 1.0]]),
        bore2spin_quat=instr.bore2spin_quat,
    )
    colatitude = pointings_and_polangle[..., 0]

    with open("spin2ecliptic_quats.txt", "wt") as outf:
        for i in range(sim.spin2ecliptic_quats.quats.shape[0]):
            print(*sim.spin2ecliptic_quats.quats[i, :], file=outf)

    with open("pointings.txt", "wt") as outf:
        import healpy

        for i in range(pointings_and_polangle.shape[1]):
            print(
                *healpy.ang2vec(
                    pointings_and_polangle[0, i, 0], pointings_and_polangle[0, i, 1]
                ),
                file=outf,
            )

    # Check that the colatitude does not depart more than ±15° from
    # the Ecliptic
    assert np.allclose(np.rad2deg(np.max(colatitude)), 90 + 15, atol=0.01)
    assert np.allclose(np.rad2deg(np.min(colatitude)), 90 - 15, atol=0.01)

    sim.flush()


def test_simulation_pointings_mjd(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=Time("2020-01-01T00:00:00"),
        duration_s=130.0,
    )
    fakedet = create_fake_detector()

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=2, distribute=False,
    )

    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=10.0, precession_rate_hz=10.0, spin_rate_hz=0.1,
    )
    sim.generate_spin2ecl_quaternions(scanning_strategy=sstr, delta_time_s=60.0)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(20.0))

    for obs in sim.observations:
        pointings_and_polangle = obs.get_pointings(
            spin2ecliptic_quats=sim.spin2ecliptic_quats,
            detector_quats=np.array([[0.0, 0.0, 0.0, 1.0]]),
            bore2spin_quat=instr.bore2spin_quat,
        )


def test_scanning_quaternions(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir", start_time=0.0, duration_s=61.0,
    )
    fakedet = create_fake_detector(sampling_rate_hz=50.0)

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=1, distribute=False,
    )
    assert len(sim.observations) == 1
    obs = sim.observations[0]

    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=1.0,
    )
    sim.generate_spin2ecl_quaternions(scanning_strategy=sstr, delta_time_s=0.5)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(15.0))
    detector_quat = np.array([[0.0, 0.0, 0.0, 1.0]])

    det2ecl_quats = obs.get_det2ecl_quaternions(
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        detector_quats=detector_quat,
        bore2spin_quat=instr.bore2spin_quat,
    )

    ecl2det_quats = obs.get_ecl2det_quaternions(
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        detector_quats=detector_quat,
        bore2spin_quat=instr.bore2spin_quat,
    )

    identity = np.array([0, 0, 0, 1])
    det2ecl_quats = det2ecl_quats.reshape(-1, 4)
    ecl2det_quats = ecl2det_quats.reshape(-1, 4)
    for i in range(det2ecl_quats.shape[0]):
        # Check that the two quaternions (ecl2det and det2ecl) are
        # actually one the inverse of the other
        quat = np.copy(det2ecl_quats[i, :])
        lbs.quat_right_multiply(quat, *ecl2det_quats[i, :])
        assert np.allclose(quat, identity)
