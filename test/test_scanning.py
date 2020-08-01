# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs


def test_scanning_strategy_spin():
    # Simulate an acquisition of 5 samples
    q = np.empty((5, 4))
    lbs.all_boresight_to_ecliptic(
        result_matrix=q,
        sun_earth_angles_rad=np.zeros(5),
        spin_sun_angle_rad=0.0,
        spin_boresight_angle_rad=0.0,
        precession_rate_hz=0.0,
        spin_rate_hz=1.0,
        time_vector_s=np.array([0.0, 0.25, 0.50, 0.75, 1.0]),
    )

    expected_result = np.array(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]]
    )
    rot = np.empty(3)
    z = np.array([0, 0, 1])
    for row in range(q.shape[0]):
        # Rotate the [0 0 1] vector using the quaternion and check that the
        # result is what's expected
        lbs.rotate_vector(rot, *q[row, :], z)
        assert np.allclose(expected_result[row, :], rot)

    ############################################################

    # Slightly more complex case, with a 45° angle between the spin axis and the Sun

    lbs.all_boresight_to_ecliptic(
        result_matrix=q,
        sun_earth_angles_rad=np.zeros(5),
        spin_sun_angle_rad=np.deg2rad(45.0),
        spin_boresight_angle_rad=0.0,
        precession_rate_hz=0.0,
        spin_rate_hz=1.0,
        time_vector_s=np.array([0.0, 0.25, 0.50, 0.75, 1.0]),
    )

    sqr2 = np.sqrt(2) / 2
    expected_result = np.array(
        [
            [sqr2, 0, sqr2],
            [0, sqr2, sqr2],
            [-sqr2, 0, sqr2],
            [0, -sqr2, sqr2],
            [sqr2, 0, sqr2],
        ]
    )
    rot = np.empty(3)
    for row in range(q.shape[0]):
        # This time, we use the rotate_z_vector function
        lbs.rotate_z_vector(rot, *q[row, :])
        assert np.allclose(expected_result[row, :], rot)

    ############################################################

    # Even more complex case, with a 45° angle between the spin axis and the Sun

    lbs.all_boresight_to_ecliptic(
        result_matrix=q,
        sun_earth_angles_rad=np.zeros(5),
        spin_sun_angle_rad=np.deg2rad(45.0),
        spin_boresight_angle_rad=np.deg2rad(45.0),
        precession_rate_hz=1.0,
        spin_rate_hz=1.0,
        time_vector_s=np.array([0.0, 0.25, 0.50, 0.75, 1.0]),
    )

    sqr2 = np.sqrt(2) / 2
    expected_result = np.array(
        [[1, 0, 0], [sqr2, 0, -sqr2], [0, 0, -1], [sqr2, 0, -sqr2], [1, 0, 0]]
    )
    rot = np.empty(3)
    for row in range(q.shape[0]):
        # This time, we use the rotate_z_vector function
        lbs.rotate_z_vector(rot, *q[row, :])
        assert np.allclose(expected_result[row, :], rot)
