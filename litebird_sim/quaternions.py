# -*- encoding: utf-8 -*-

import numpy as np
from numba import njit


@njit
def quat_rotation_x(theta_rad):
    """Return a quaternion representing a rotation around the x axis

    The angle `theta_rad` must be expressed in radians. The return
    value is the quaternion, using the order ``(v_x, v_y, v_z, w)``;
    it is returned as a 4-element tuple.

    The fact that the result is a tuple instead of a NumPy array is
    because of speed: it helps in preventing unnecessary allocations
    in performance-critical code.

    See also :func:`quat_rotation_y` and :func:`quat_rotation_z`
    """
    return (np.sin(theta_rad / 2), 0.0, 0.0, np.cos(theta_rad / 2))


@njit
def quat_rotation_y(theta_rad):
    """Return a quaternion representing a rotation around the y axis

    See also :func:`quat_rotation_x` and :func:`quat_rotation_z`
    """
    return (0.0, np.sin(theta_rad / 2), 0.0, np.cos(theta_rad / 2))


@njit
def quat_rotation_z(theta_rad):
    """Return a quaternion representing a rotation around the y axis

    See also :func:`quat_rotation_x` and :func:`quat_rotation_y`
    """
    return (0.0, 0.0, np.sin(theta_rad / 2), np.cos(theta_rad / 2))


@njit
def quat_right_multiply(result, other_v1, other_v2, other_v3, other_w):
    """Perform a multiplication between two quaternions

    This function implements the computation :math:`r = r \times q`,
    where `r` is the parameter `result` (a 3-element NumPy array) and
    `q` is the set of parameters `other_v1`, `other_v2`, `other_v3`,
    `other_w`. The reason why the elements of quaternion `q` are
    passed one by one is efficiency: in this way, the caller does not
    have to allocate a numpy.array for simple quaternions (like the
    ones returned by :func:`quat_rotation_x`, :func:`quat_rotation_y`,
    :func:`quat_rotation_z`).

    It's easy to use NumPy quaternions for `q` as well::

        import numpy as np
        r = np.array([1.0, 2.0, 3.0, 4.0])
        q = np.array([0.1, 0.2, 0.3, 0.4])
        quat_right_multiply(r, *q)  # Unpack "q"
        print("Result:", r)

    See also :func:`quat_left_multiply` for the computation :math:`r =
    q \times r`.

    """

    v1 = (
        result[3] * other_v1
        + result[0] * other_w
        + result[1] * other_v3
        - result[2] * other_v2
    )
    v2 = (
        result[3] * other_v2
        - result[0] * other_v3
        + result[1] * other_w
        + result[2] * other_v1
    )
    v3 = (
        result[3] * other_v3
        + result[0] * other_v2
        - result[1] * other_v1
        + result[2] * other_w
    )
    w = (
        result[3] * other_w
        - result[0] * other_v1
        - result[1] * other_v2
        - result[2] * other_v3
    )

    result[0] = v1
    result[1] = v2
    result[2] = v3
    result[3] = w


@njit
def quat_left_multiply(result, other_v1, other_v2, other_v3, other_w):
    """Perform a multiplication between two quaternions

    This function implements the computation :math:`r = q \\times r`;
    see also :func:`quat_right_multiply` for the computation :math:`r
    = r\\times q`.

    It's easy to use NumPy quaternions for `q` as well::

        import numpy as np
        r = np.array([1.0, 2.0, 3.0, 4.0])
        q = np.array([0.1, 0.2, 0.3, 0.4])
        quat_right_multiply(r, *q)  # Unpack "q"
        print("Result:", r)

    """

    v1 = (
        other_w * result[0]
        + other_v1 * result[3]
        + other_v2 * result[2]
        - other_v3 * result[1]
    )
    v2 = (
        other_w * result[1]
        - other_v1 * result[2]
        + other_v2 * result[3]
        + other_v3 * result[0]
    )
    v3 = (
        other_w * result[2]
        + other_v1 * result[1]
        - other_v2 * result[0]
        + other_v3 * result[3]
    )
    w = (
        other_w * result[3]
        - other_v1 * result[0]
        - other_v2 * result[1]
        - other_v3 * result[2]
    )

    result[0] = v1
    result[1] = v2
    result[2] = v3
    result[3] = w


@njit
def _cross(result, v0, v1, v2, w0, w1, w2):
    result[0] = v1 * w2 - v2 * w1
    result[1] = v2 * w0 - v0 * w2
    result[2] = v0 * w1 - v1 * w0


@njit
def rotate_vector(result, vx, vy, vz, w, vect):
    """Rotate a vector using a quaternion

    Applies a rotation, encoded through the quaternion `vx, vy, vz,
    vw`, to the vector `vect` (a 3-element NumPy array), storing the
    result in `result` (again a 3-element array).

    The formula to rotate a vector `v` by a quaternion `(q_v, w)` is
    the following: :math:`v' = v + 2q_v ⨯ (q_v ⨯ v + w v)`, where
    `q_v` is the vector `(vx, vy, vz)`.

    """

    # In the code below the term within the parentheses has already
    # been expanded (it's just basic algebra), and the call to _cross
    # computes the external cross product.

    _cross(
        result,
        vx,
        vy,
        vz,
        vy * vect[2] - vz * vect[1] + w * vect[0],
        -vx * vect[2] + vz * vect[0] + w * vect[1],
        vx * vect[1] - vy * vect[0] + w * vect[2],
    )
    for i in (0, 1, 2):
        result[i] = vect[i] + 2.0 * result[i]


@njit
def rotate_x_vector(result, vx, vy, vz, w):
    """Rotate the x vector using the quaternion (vx, vy, vz, w)

    This function is equivalent to ``rotate_vector(result, vx, vy, vz,
    w, [1, 0, 0])``, but it's faster.

    """
    # The same as rotate_vector, but it's faster
    result[0] = 1.0 - 2 * (vy * vy + vz * vz)
    result[1] = 2 * (vx * vy + w * vz)
    result[2] = 2 * (vx * vz - w * vy)


@njit
def rotate_y_vector(result, vx, vy, vz, w):
    """Rotate the x vector using the quaternion (vx, vy, vz, w)

    This function is equivalent to ``rotate_vector(result, vx, vy, vz,
    w, [0, 1, 0])``, but it's faster.

    """
    # The same as rotate_vector, but it's faster
    result[0] = 2 * (vx * vy - w * vz)
    result[1] = 1.0 - 2 * (vx * vx + vz * vz)
    result[2] = 2 * (w * vx + vy * vz)


@njit
def rotate_z_vector(result, vx, vy, vz, w):
    """Rotate the x vector using the quaternion (vx, vy, vz, w)

    This function is equivalent to ``rotate_vector(result, vx, vy, vz,
    w, [0, 0, 1])``, but it's faster.

    """
    # The same as rotate_vector, but it's faster
    result[0] = 2 * (w * vy + vx * vz)
    result[1] = 2 * (vy * vz - w * vx)
    result[2] = 1.0 - 2 * (vx * vx + vy * vy)
