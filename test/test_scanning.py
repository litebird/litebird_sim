# -*- encoding: utf-8 -*-

from pathlib import Path

import pytest
from astropy.time import Time
import numpy as np
import litebird_sim as lbs
from litebird_sim import IdealHWP


def test_compute_pointing_and_orientation():
    quat = np.array(lbs.quat_rotation_y(np.pi / 2))
    result = np.empty(3)
    lbs.compute_pointing_and_orientation(result, quat)
    assert np.allclose(result, [np.pi / 2, 0.0, -np.pi / 2])

    # We stay along the same pointing, but we're rotating the detector
    # by 90°, so the orientation angle is the only number that
    # changes
    lbs.quat_left_multiply(quat, *lbs.quat_rotation_x(np.pi / 4))
    lbs.compute_pointing_and_orientation(result, quat)
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
    assert np.allclose(
        lbs.calculate_sun_earth_angles_rad(time), -1.570_796_364_387_955_7
    )


def create_fake_detector(
    sampling_rate_hz: float = 1.0, quat=np.array([0.0, 0.0, 0.0, 1.0])
):
    return lbs.DetectorInfo(name="dummy", sampling_rate_hz=sampling_rate_hz, quat=quat)


def test_simulation_pointings_still():
    sim = lbs.Simulation(start_time=0.0, duration_s=86400.0, random_seed=12345)
    fakedet = create_fake_detector(sampling_rate_hz=1 / 3600)

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=1, split_list_over_processes=False
    )
    assert len(sim.observations) == 1
    obs = sim.observations[0]

    # The spacecraft stands still in L2, with no spinning nor precession
    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=0.0
    )
    sim.set_scanning_strategy(sstr, delta_time_s=60.0)
    assert sim.spin2ecliptic_quats.quats.shape == (24 * 60 + 1, 4)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=0.0)

    # Move the Z vector manually using the last quaternion and check
    # that it's rotated by 1/365.25 of a complete circle
    boresight = np.empty(3)
    lbs.rotate_z_vector(boresight, *sim.spin2ecliptic_quats.quats[-1, :])
    assert np.allclose(np.arctan2(boresight[1], boresight[0]), 2 * np.pi / 365.25)

    lbs.prepare_pointings(
        obs,
        instr,
        sim.spin2ecliptic_quats,
    )

    pointings_and_orientation = obs.get_pointings("all")[0]

    colatitude = pointings_and_orientation[..., 0]
    longitude = pointings_and_orientation[..., 1]
    orientation = pointings_and_orientation[..., 2]

    assert np.allclose(colatitude, np.pi / 2), colatitude
    assert np.allclose(np.abs(orientation), np.pi / 2), orientation

    # The longitude should have changed by a fraction 23 hours /
    # 365.25 days of a complete circle (we have 24 samples, from t = 0
    # to t = 23 hr)
    assert np.allclose(
        np.abs(longitude[..., -1] - longitude[..., 0]), 2 * np.pi * 23 / 365.25 / 24
    )


def test_simulation_two_detectors():
    sim = lbs.Simulation(start_time=0.0, duration_s=86400.0, random_seed=12345)

    # Two detectors, the second rotated by 45°
    quaternions = [
        lbs.RotQuaternion(),
        lbs.RotQuaternion(np.array([0.0, 0.0, 1.0, 1.0]) / np.sqrt(2)),
    ]
    fakedet2 = create_fake_detector(sampling_rate_hz=1.0 / 3600, quat=quaternions[1])
    fakedet1 = create_fake_detector(sampling_rate_hz=1.0 / 3600, quat=quaternions[0])

    sim.create_observations(
        detectors=[fakedet1, fakedet2],
        num_of_obs_per_detector=1,
        split_list_over_processes=False,
    )
    assert len(sim.observations) == 1
    obs = sim.observations[0]

    # The spacecraft stands still in L2, with no spinning nor precession
    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=0.0
    )
    sim.set_scanning_strategy(sstr, delta_time_s=60.0)
    assert sim.spin2ecliptic_quats.quats.shape == (24 * 60 + 1, 4)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=0.0)

    lbs.prepare_pointings(
        obs,
        instr,
        sim.spin2ecliptic_quats,
    )

    pointings_and_orientation = obs.get_pointings("all")[0]

    assert pointings_and_orientation.shape == (2, 24, 3)

    assert np.allclose(
        pointings_and_orientation[0, :, 0], pointings_and_orientation[1, :, 0]
    )
    assert np.allclose(
        pointings_and_orientation[0, :, 1], pointings_and_orientation[1, :, 1]
    )

    # The ψ angle should differ by 45°
    assert np.allclose(
        np.abs(pointings_and_orientation[0, :, 2] - pointings_and_orientation[1, :, 2]),
        np.pi / 2,
    )


def test_simulation_pointings_orientation(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=0.0,
        duration_s=61.0,
        random_seed=12345,
    )
    fakedet = create_fake_detector(sampling_rate_hz=50.0)

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=1, split_list_over_processes=False
    )
    assert len(sim.observations) == 1
    obs = sim.observations[0]

    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=1.0 / 60
    )
    sim.set_scanning_strategy(scanning_strategy=sstr, delta_time_s=0.5)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=0.0)

    lbs.prepare_pointings(
        obs,
        instr,
        sim.spin2ecliptic_quats,
    )

    pointings_and_orientation = obs.get_pointings("all")[0]

    orientation = pointings_and_orientation[..., 2]

    # Check that the orientation scans every value in [-π, +π]
    assert np.allclose(np.max(orientation), np.pi, atol=0.01)
    assert np.allclose(np.min(orientation), -np.pi, atol=0.01)

    # Simulate the generation of a report
    sim.flush()


def test_simulation_pointings_spinning(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=0.0,
        duration_s=61.0,
        random_seed=12345,
    )
    fakedet = create_fake_detector(sampling_rate_hz=50.0)

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=1, split_list_over_processes=False
    )
    assert len(sim.observations) == 1
    obs = sim.observations[0]

    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=1.0
    )
    sim.set_scanning_strategy(scanning_strategy=sstr, delta_time_s=0.5)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(15.0))

    lbs.prepare_pointings(
        obs,
        instr,
        sim.spin2ecliptic_quats,
    )

    pointings_and_orientation = obs.get_pointings("all")[0]

    colatitude = pointings_and_orientation[..., 0]

    reference_spin2ecliptic_file = Path(__file__).parent / "reference_spin2ecl.txt.gz"
    reference = np.loadtxt(reference_spin2ecliptic_file)
    assert np.allclose(sim.spin2ecliptic_quats.quats, reference)

    reference_pointings_file = Path(__file__).parent / "reference_pointings.txt.gz"
    reference = np.loadtxt(reference_pointings_file)
    assert np.allclose(pointings_and_orientation[0, :, :], reference)

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
        random_seed=12345,
    )
    fakedet = create_fake_detector()

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=2, split_list_over_processes=False
    )

    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=10.0, precession_rate_hz=10.0, spin_rate_hz=0.1
    )
    sim.set_scanning_strategy(scanning_strategy=sstr, delta_time_s=60.0)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(20.0))

    for idx, obs in enumerate(sim.observations):
        lbs.prepare_pointings(
            obs,
            instr,
            sim.spin2ecliptic_quats,
        )

        pointings_and_orientation = obs.get_pointings("all")[0]

        filename = Path(__file__).parent / f"reference_obs_pointings{idx:03d}.npy"
        reference = np.load(filename, allow_pickle=False)
        assert np.allclose(pointings_and_orientation, reference)


def test_simulation_pointings_hwp_mjd(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=Time("2020-01-01T00:00:00"),
        duration_s=130.0,
        random_seed=12345,
    )
    fakedet = create_fake_detector()

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=2, split_list_over_processes=False
    )

    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=10.0, precession_rate_hz=10.0, spin_rate_hz=0.1
    )
    sim.set_scanning_strategy(scanning_strategy=sstr, delta_time_s=60.0)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(20.0))

    for idx, obs in enumerate(sim.observations):
        lbs.prepare_pointings(
            obs,
            instr,
            sim.spin2ecliptic_quats,
            hwp=IdealHWP(ang_speed_radpsec=1.0, start_angle_rad=0.0),
        )

        pointings_and_orientation, hwp_angle = obs.get_pointings("all")

        pointings_and_orientation[..., 2] += hwp_angle

        filename = Path(__file__).parent / f"reference_obs_pointings_hwp{idx:03d}.npy"
        reference = np.load(filename, allow_pickle=False)
        assert np.allclose(pointings_and_orientation, reference)


def test_scanning_quaternions(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path / "simulation_dir",
        start_time=0.0,
        duration_s=61.0,
        random_seed=12345,
    )
    fakedet = create_fake_detector(sampling_rate_hz=50.0)

    sim.create_observations(
        detectors=[fakedet], num_of_obs_per_detector=1, split_list_over_processes=False
    )
    assert len(sim.observations) == 1
    obs = sim.observations[0]

    sstr = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.0, precession_rate_hz=0.0, spin_rate_hz=1.0
    )
    sim.set_scanning_strategy(scanning_strategy=sstr, delta_time_s=0.5)

    instr = lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(15.0))
    detector_quat = lbs.RotQuaternion()

    det2ecl_quats = lbs.get_det2ecl_quaternions(
        obs,
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        detector_quats=[detector_quat],
        bore2spin_quat=instr.bore2spin_quat,
    )

    ecl2det_quats = lbs.get_ecl2det_quaternions(
        obs,
        spin2ecliptic_quats=sim.spin2ecliptic_quats,
        detector_quats=[detector_quat],
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


def test_time_dependent_quaternion_constructor():
    # Constant quaternion specified by a 1D array
    q = lbs.RotQuaternion(quats=np.array([0.0, 0.0, 0.0, 1.0]))
    assert q.quats.shape == (1, 4)
    np.testing.assert_allclose(q.quats, [[0.0, 0.0, 0.0, 1.0]])
    assert q.start_time is None
    assert q.sampling_rate_hz is None

    # Constant quaternion specified by a 2D array
    q = lbs.RotQuaternion(
        quats=np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    assert q.quats.shape == (1, 4)
    np.testing.assert_allclose(q.quats, [[0.0, 0.0, 0.0, 1.0]])
    assert q.start_time is None
    assert q.sampling_rate_hz is None

    # Variable quaternion specified by a 2D array
    q = lbs.RotQuaternion(
        quats=np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        start_time=3.0,
        sampling_rate_hz=5.0,
    )
    assert q.quats.shape == (2, 4)
    assert q.start_time == 3.0
    assert q.sampling_rate_hz == 5.0

    # Copy constructor
    q_copy = lbs.RotQuaternion(q)
    assert q_copy.is_close_to(q)

    # Check that variable quaternions require both
    # `start_time` and `sampling_rate_hz`
    with pytest.raises(AssertionError):
        _ = lbs.RotQuaternion(
            quats=np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )


def test_time_dependent_quaternion_closeness():
    a = lbs.RotQuaternion(quats=np.array([0.0, 0.0, 0.0, 1.0]))
    b = lbs.RotQuaternion(quats=np.array([0.0, 0.0, 0.0, 1.0]))
    assert a.is_close_to(a)
    assert a.is_close_to(b)
    assert b.is_close_to(a)

    a = lbs.RotQuaternion(
        quats=np.array([0.0, 0.0, 0.0, 1.0]),
        start_time=0.0,
    )
    b = lbs.RotQuaternion(quats=np.array([0.0, 0.0, 0.0, 1.0]))
    assert not a.is_close_to(b)
    assert not b.is_close_to(a)

    a = lbs.RotQuaternion(
        quats=np.array([0.0, 0.0, 0.0, 1.0]),
        sampling_rate_hz=1.0,
    )
    b = lbs.RotQuaternion(quats=np.array([0.0, 0.0, 0.0, 1.0]))
    assert not a.is_close_to(b)
    assert not b.is_close_to(a)

    a = lbs.RotQuaternion(
        quats=np.array([0.0, 0.0, 0.0, 1.0]),
        start_time=1.0,
        sampling_rate_hz=10.0,
    )
    b = lbs.RotQuaternion(a)
    assert a.is_close_to(b)

    b = lbs.RotQuaternion(a)
    b.start_time = 2.0
    assert not a.is_close_to(b)

    b = lbs.RotQuaternion(a)
    b.sampling_rate_hz = 3.0
    assert not a.is_close_to(b)

    a.start_time = Time("2023-01-01T10:00:00")
    b = lbs.RotQuaternion(a)
    b.start_time = Time("2023-01-01T10:00:01")
    assert not a.is_close_to(b)


def test_time_dependent_quaternions_operations():
    qarr1 = lbs.RotQuaternion(
        quats=np.array(
            [
                [0.5, 0.0, 0.0, 0.8660254],
                [0.0, -0.38268343, 0.0, 0.92387953],
                [0.0, 0.0, 0.30901699, 0.95105652],
            ]
        ),
        start_time=0.0,
        sampling_rate_hz=1.0,
    )
    qarr2 = lbs.RotQuaternion(
        quats=np.array(
            [
                [0.0, -0.25881905, 0.0, 0.96592583],
                [0.0, 0.0, 0.22252093, 0.97492791],
                [-0.19509032, 0.0, 0.0, 0.98078528],
            ]
        ),
        start_time=0.0,
        sampling_rate_hz=1.0,
    )
    qconst1 = lbs.RotQuaternion(
        quats=np.array([[0.5, 0.0, 0.0, 0.8660254]]),
    )
    qconst2 = lbs.RotQuaternion(
        quats=np.array([[-0.19509032, 0.0, 0.0, 0.98078528]]),
    )

    # First test: array × array
    result = qarr1 * qarr2
    expected = np.empty((3, 4))
    lbs.multiply_quaternions_list_x_list(
        array_a=qarr1.quats, array_b=qarr2.quats, result=expected
    )
    np.testing.assert_allclose(actual=result.quats, desired=expected)

    # Second test: array × one
    result = qarr1 * qconst1
    expected = np.empty((3, 4))
    lbs.multiply_quaternions_list_x_one(
        array_a=qarr1.quats, single_b=qconst1.quats[0, :], result=expected
    )
    np.testing.assert_allclose(actual=result.quats, desired=expected)

    # Third test: one × array
    result = qconst1 * qarr1
    expected = np.empty((3, 4))
    lbs.multiply_quaternions_one_x_list(
        single_a=qconst1.quats[0, :], array_b=qarr1.quats, result=expected
    )
    np.testing.assert_allclose(actual=result.quats, desired=expected)

    # Fourth test: one × one
    result = qconst1 * qconst2
    expected = np.empty((1, 4))
    expected[0, :] = qconst1.quats[0, :]
    lbs.quat_right_multiply(expected[0, :], *qconst2.quats[0, :])
    np.testing.assert_allclose(actual=result.quats, desired=expected)
