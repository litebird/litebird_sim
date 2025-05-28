# -*- encoding: utf-8 -*-

import numpy as np
import numpy.testing as nptest

import litebird_sim as lbs

x = np.array([1.0, 0.0, 0.0])
y = np.array([0.0, 1.0, 0.0])
z = np.array([0.0, 0.0, 1.0])


def test_rotation_handedness():
    result = np.empty(3)

    lbs.rotate_vector(result, *lbs.quat_rotation_x(np.pi / 2), y)
    nptest.assert_allclose(result, z, atol=1e-15)
    lbs.rotate_vector(result, *lbs.quat_rotation_x(np.pi / 2), z)
    nptest.assert_allclose(result, -y, atol=1e-15)
    lbs.rotate_vector(result, *lbs.quat_rotation_y(np.pi / 2), x)
    nptest.assert_allclose(result, -z, atol=1e-15)
    lbs.rotate_vector(result, *lbs.quat_rotation_y(np.pi / 2), z)
    nptest.assert_allclose(result, x, atol=1e-15)
    lbs.rotate_vector(result, *lbs.quat_rotation_z(np.pi / 2), x)
    nptest.assert_allclose(result, y, atol=1e-15)
    lbs.rotate_vector(result, *lbs.quat_rotation_z(np.pi / 2), y)
    nptest.assert_allclose(result, -x, atol=1e-15)


def test_quat_multiply_and_rotations():
    # Simple composition of rotations
    quat = np.array(lbs.quat_rotation_x(np.pi / 3))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_x(np.pi / 3))
    nptest.assert_allclose(quat, np.array(lbs.quat_rotation_x(2 * np.pi / 3)))

    quat = np.array(lbs.quat_rotation_y(np.pi / 3))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_y(np.pi / 3))
    nptest.assert_allclose(quat, np.array(lbs.quat_rotation_y(2 * np.pi / 3)))

    quat = np.array(lbs.quat_rotation_z(np.pi / 3))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_z(np.pi / 3))
    nptest.assert_allclose(quat, np.array(lbs.quat_rotation_z(2 * np.pi / 3)))

    quat = np.array(lbs.quat_rotation_x(np.pi / 3))
    lbs.quat_left_multiply(quat, *lbs.quat_rotation_x(np.pi / 3))
    nptest.assert_allclose(quat, np.array(lbs.quat_rotation_x(2 * np.pi / 3)))

    quat = np.array(lbs.quat_rotation_y(np.pi / 3))
    lbs.quat_left_multiply(quat, *lbs.quat_rotation_y(np.pi / 3))
    nptest.assert_allclose(quat, np.array(lbs.quat_rotation_y(2 * np.pi / 3)))

    quat = np.array(lbs.quat_rotation_z(np.pi / 3))
    lbs.quat_left_multiply(quat, *lbs.quat_rotation_z(np.pi / 3))
    nptest.assert_allclose(quat, np.array(lbs.quat_rotation_z(2 * np.pi / 3)))

    # Now we test more complex compositions

    vec = np.empty(3)

    # Right multiplication
    quat = np.array(lbs.quat_rotation_y(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_x(np.pi / 2))
    lbs.rotate_vector(vec, *quat, y)
    nptest.assert_allclose(vec, x, atol=1e-15)

    quat = np.array(lbs.quat_rotation_z(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_y(np.pi / 2))
    lbs.rotate_vector(vec, *quat, z)
    nptest.assert_allclose(vec, y, atol=1e-15)

    quat = np.array(lbs.quat_rotation_x(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_y(np.pi / 2))
    lbs.rotate_vector(vec, *quat, x)
    nptest.assert_allclose(vec, y, atol=1e-15)

    # Left multiplication
    quat = np.array(lbs.quat_rotation_y(np.pi / 2))
    lbs.quat_left_multiply(quat, *lbs.quat_rotation_z(np.pi / 2))
    lbs.rotate_vector(vec, *quat, z)
    nptest.assert_allclose(vec, y, atol=1e-15)

    quat = np.array(lbs.quat_rotation_z(np.pi / 2))
    lbs.quat_left_multiply(quat, *lbs.quat_rotation_y(np.pi / 2))
    lbs.rotate_vector(vec, *quat, y)
    nptest.assert_allclose(vec, z, atol=1e-15)

    quat = np.array(lbs.quat_rotation_z(np.pi / 2))
    lbs.quat_left_multiply(quat, *lbs.quat_rotation_x(np.pi / 2))
    lbs.rotate_vector(vec, *quat, x)
    nptest.assert_allclose(vec, z, atol=1e-15)


def test_quick_rotations():
    vec = np.empty(3)

    quat = np.array(lbs.quat_rotation_z(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_y(np.pi / 2))
    lbs.rotate_z_vector(vec, *quat)
    nptest.assert_allclose(vec, y, atol=1e-15)

    quat = np.array(lbs.quat_rotation_x(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_y(np.pi / 2))
    lbs.rotate_x_vector(vec, *quat)
    nptest.assert_allclose(vec, y, atol=1e-15)

    quat = np.array(lbs.quat_rotation_y(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_x(np.pi / 2))
    lbs.rotate_y_vector(vec, *quat)
    nptest.assert_allclose(vec, x, atol=1e-15)


def test_collective_rotations():
    vec = np.empty((1, 3))

    quat = np.array(lbs.quat_rotation_y(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_x(np.pi / 2))
    lbs.all_rotate_vectors(vec, quat.reshape(1, 4), y.reshape(1, 3))
    nptest.assert_allclose(vec, [x], atol=1e-15)

    quat = np.array(lbs.quat_rotation_z(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_y(np.pi / 2))
    lbs.all_rotate_vectors(vec, quat.reshape(1, 4), z.reshape(1, 3))
    nptest.assert_allclose(vec, [y], atol=1e-15)

    quat = np.array(lbs.quat_rotation_x(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_y(np.pi / 2))
    lbs.all_rotate_vectors(vec, quat.reshape(1, 4), x.reshape(1, 3))
    nptest.assert_allclose(vec, [y], atol=1e-15)


def test_collective_quick_rotations():
    vec = np.empty((1, 3))

    quat = np.array(lbs.quat_rotation_z(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_y(np.pi / 2))
    lbs.all_rotate_z_vectors(vec, quat.reshape(1, 4))
    nptest.assert_allclose(vec, [y], atol=1e-15)

    quat = np.array(lbs.quat_rotation_x(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_y(np.pi / 2))
    lbs.all_rotate_x_vectors(vec, quat.reshape(1, 4))
    nptest.assert_allclose(vec, [y], atol=1e-15)

    quat = np.array(lbs.quat_rotation_y(np.pi / 2))
    lbs.quat_right_multiply(quat, *lbs.quat_rotation_x(np.pi / 2))
    lbs.all_rotate_y_vectors(vec, quat.reshape(1, 4))
    nptest.assert_allclose(vec, [x], atol=1e-15)


def test_multiply_many_quaternions():
    first_matrix = np.empty((3, 4))
    second_matrix = np.empty_like(first_matrix)
    result_matrix = np.empty_like(first_matrix)

    first_matrix[0, :] = lbs.quat_rotation_x(theta_rad=np.pi / 3.0)
    first_matrix[1, :] = lbs.quat_rotation_y(theta_rad=-np.pi / 4.0)
    first_matrix[2, :] = lbs.quat_rotation_z(theta_rad=np.pi / 5.0)

    second_matrix[0, :] = lbs.quat_rotation_y(theta_rad=-np.pi / 6.0)
    second_matrix[1, :] = lbs.quat_rotation_z(theta_rad=np.pi / 7.0)
    second_matrix[2, :] = lbs.quat_rotation_x(theta_rad=-np.pi / 8.0)

    lbs.multiply_quaternions_list_x_list(
        array_a=first_matrix, array_b=second_matrix, result=result_matrix
    )

    expected_quaternion = np.empty(4)

    for i in range(first_matrix.shape[0]):
        expected_quaternion[:] = first_matrix[i, :]
        lbs.quat_right_multiply(expected_quaternion, *second_matrix[i, :])
        np.testing.assert_allclose(
            actual=result_matrix[i, :], desired=expected_quaternion
        )


def test_multiply_many_quaternions_by_one_quaternion():
    first_matrix = np.empty((3, 4))
    second_matrix = np.empty_like(first_matrix)
    result_matrix = np.empty_like(first_matrix)

    first_matrix[0, :] = lbs.quat_rotation_x(theta_rad=np.pi / 3.0)
    first_matrix[1, :] = lbs.quat_rotation_y(theta_rad=-np.pi / 4.0)
    first_matrix[2, :] = lbs.quat_rotation_z(theta_rad=np.pi / 5.0)

    second_matrix[0, :] = lbs.quat_rotation_y(theta_rad=-np.pi / 6.0)
    second_matrix[1, :] = lbs.quat_rotation_z(theta_rad=np.pi / 7.0)
    second_matrix[2, :] = lbs.quat_rotation_x(theta_rad=-np.pi / 8.0)

    # First test: just use the first entry in `second_matrix` and test the case list × one
    lbs.multiply_quaternions_list_x_one(
        array_a=first_matrix, single_b=second_matrix[0, :], result=result_matrix
    )

    expected_quaternion = np.empty(4)

    for i in range(first_matrix.shape[0]):
        expected_quaternion[:] = first_matrix[i, :]
        lbs.quat_right_multiply(expected_quaternion, *second_matrix[0, :])
        np.testing.assert_allclose(
            actual=result_matrix[i, :], desired=expected_quaternion
        )

    # Second test: use the first entry in `first_matrix` and test the case one × list
    lbs.multiply_quaternions_one_x_list(
        single_a=first_matrix[0, :], array_b=second_matrix, result=result_matrix
    )

    expected_quaternion = np.empty(4)

    for i in range(first_matrix.shape[0]):
        expected_quaternion[:] = first_matrix[0, :]
        lbs.quat_right_multiply(expected_quaternion, *second_matrix[i, :])
        np.testing.assert_allclose(
            actual=result_matrix[i, :], desired=expected_quaternion
        )


def test_normalize_quaternions():
    quats = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 3.0, 2.0],
        ]
    )

    lbs.normalize_quaternions(quats)

    for i in range(quats.shape[0]):
        cur_quat = quats[i, :]
        np.testing.assert_almost_equal(actual=np.dot(cur_quat, cur_quat), desired=1.0)


def test_quat_rotations():
    vec_matrix = np.array([x, y, z])

    theta_rad = np.pi / 2
    theta_rad_array = np.array([0.0, np.pi / 2, np.pi / 3])

    quat = lbs.quat_rotation(theta_rad, *z)
    nptest.assert_allclose(quat, np.array(lbs.quat_rotation_z(theta_rad)))

    quat_matrix = lbs.quat_rotation_brdcast(theta_rad, vec_matrix)
    expected_quat = np.array(
        [
            np.array(lbs.quat_rotation_x(theta_rad)),
            np.array(lbs.quat_rotation_y(theta_rad)),
            np.array(lbs.quat_rotation_z(theta_rad)),
        ]
    )
    nptest.assert_allclose(quat_matrix, expected_quat)

    quat_matrix = lbs.quat_rotation_x_brdcast(theta_rad_array)
    expected_quat = np.array(
        [np.array(lbs.quat_rotation_x(theta)) for theta in theta_rad_array]
    )
    nptest.assert_allclose(quat_matrix, expected_quat)

    quat_matrix = lbs.quat_rotation_y_brdcast(theta_rad_array)
    expected_quat = np.array(
        [np.array(lbs.quat_rotation_y(theta)) for theta in theta_rad_array]
    )
    nptest.assert_allclose(quat_matrix, expected_quat)

    quat_matrix = lbs.quat_rotation_z_brdcast(theta_rad_array)
    expected_quat = np.array(
        [np.array(lbs.quat_rotation_z(theta)) for theta in theta_rad_array]
    )
    nptest.assert_allclose(quat_matrix, expected_quat)
