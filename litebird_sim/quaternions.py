# -*- encoding: utf-8 -*-

import numpy as np
import numpy.typing as npt
from numba import njit, prange

x = np.array([1.0, 0.0, 0.0])
y = np.array([0.0, 1.0, 0.0])
z = np.array([0.0, 0.0, 1.0])


@njit
def quat_rotation_x(theta_rad):
    """Return a quaternion representing a rotation around the x axis

    Prototype::

        quat_rotation_x(theta_rad: float) -> Tuple[float, float, float, float]

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

    Prototype::

        quat_rotation_y(theta_rad: float) -> Tuple[float, float, float, float]

    See also :func:`quat_rotation_x` and :func:`quat_rotation_z`
    """
    return (0.0, np.sin(theta_rad / 2), 0.0, np.cos(theta_rad / 2))


@njit
def quat_rotation_z(theta_rad):
    """Return a quaternion representing a rotation around the y axis

    Prototype::

        quat_rotation_z(theta_rad: float) -> Tuple[float, float, float, float]

    See also :func:`quat_rotation_x` and :func:`quat_rotation_y`
    """
    return (0.0, 0.0, np.sin(theta_rad / 2), np.cos(theta_rad / 2))


@njit
def quat_right_multiply(result, other_v1, other_v2, other_v3, other_w):
    """Perform a multiplication between two quaternions

    Prototype::

        quat_right_multiply(
            result: numpy.array[3],
            other_v1: float,
            other_v2: float,
            other_v3: float,
            other_w: float,
        )

    This function implements the computation :math:`r = r \\times q`,
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

    Prototype::

        quat_left_multiply(
            result: numpy.array[3],
            other_v1: float,
            other_v2: float,
            other_v3: float,
            other_w: float,
        )

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

    Prototype::

        rotate_vector(
            result: numpy.array[3],
            vx: float,
            vy: float,
            vz: float,
            w: float,
            vect: numpy.array[3],
        )

    Applies a rotation, encoded through the quaternion `vx, vy, vz,
    vw`, to the vector `vect` (a 3-element NumPy array), storing the
    result in `result` (again a 3-element array).

    *Note:* do not pass the same variable to `vect` and `result`!

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

    Prototype::

        rotate_x_vector(
            result: numpy.array[3],
            vx: float,
            vy: float,
            vz: float,
            w: float,
        )

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

    Prototype::

        rotate_y_vector(
            result: numpy.array[3],
            vx: float,
            vy: float,
            vz: float,
            w: float,
        )

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

    Prototype::

        rotate_z_vector(
            result: numpy.array[3],
            vx: float,
            vy: float,
            vz: float,
            w: float,
        )

    This function is equivalent to ``rotate_vector(result, vx, vy, vz,
    w, [0, 0, 1])``, but it's faster.

    """
    # The same as rotate_vector, but it's faster
    result[0] = 2 * (w * vy + vx * vz)
    result[1] = 2 * (vy * vz - w * vx)
    result[2] = 1.0 - 2 * (vx * vx + vy * vy)


@njit
def all_rotate_vectors(result_matrix, quat_matrix, vec_matrix):
    """Rotate a set of vectors using quaternions

    Prototype::

        all_rotate_vectors(
            result_matrix: numpy.array[N, 3],
            quat_matrix: numpy.array[N, 4],
            vec_matrix: numpy.array[N, 3],
        )

    Assuming that `result_matrix` and `vec_matrix` are two NumPy
    arrays with shape ``(N, 3)`` and `quat_matrix` with shape ``(N,
    4)``, apply :func:`rotate_vector` to each row.

    """
    for rowidx in range(result_matrix.shape[0]):
        vx, vy, vz, w = quat_matrix[rowidx, :]
        rotate_vector(result_matrix[rowidx, :], vx, vy, vz, w, vec_matrix[rowidx, :])


@njit
def all_rotate_x_vectors(result_matrix, quat_matrix):
    """Rotate the vector ``[1, 0, 0]`` using quaternions

    Prototype::

        all_rotate_x_vectors(
            result_matrix: numpy.array[N, 3],
            quat_matrix: numpy.array[N, 4],
        )

    Assuming that `result_matrix` is a NumPy array with shape ``(N,
    3)`` and `quat_matrix` with shape ``(N, 4)``, apply
    :func:`rotate_x_vector` to each row.

    """
    for rowidx in range(result_matrix.shape[0]):
        rotate_x_vector(
            result_matrix[rowidx, :],
            quat_matrix[rowidx, 0],
            quat_matrix[rowidx, 1],
            quat_matrix[rowidx, 2],
            quat_matrix[rowidx, 3],
        )


@njit
def all_rotate_y_vectors(result_matrix, quat_matrix):
    """Rotate the vector ``[0, 1, 0]`` using quaternions

    Prototype::

        all_rotate_y_vectors(
            result_matrix: numpy.array[N, 3],
            quat_matrix: numpy.array[N, 4],
        )

    Assuming that `result_matrix` is a NumPy array with shape ``(N,
    3)`` and `quat_matrix` with shape ``(N, 4)``, apply
    :func:`rotate_y_vector` to each row.

    """
    for rowidx in range(result_matrix.shape[0]):
        rotate_y_vector(
            result_matrix[rowidx, :],
            quat_matrix[rowidx, 0],
            quat_matrix[rowidx, 1],
            quat_matrix[rowidx, 2],
            quat_matrix[rowidx, 3],
        )


@njit
def all_rotate_z_vectors(result_matrix, quat_matrix):
    """Rotate the vector ``[0, 0, 1]`` using quaternions

    Prototype::

        all_rotate_z_vectors(
            result_matrix: numpy.array[N, 3],
            quat_matrix: numpy.array[N, 4],
        )

    Assuming that `result_matrix` is a NumPy array with shape ``(N,
    3)`` and `quat_matrix` with shape ``(N, 4)``, apply
    :func:`rotate_z_vector` to each row.

    """

    for rowidx in range(result_matrix.shape[0]):
        rotate_z_vector(
            result_matrix[rowidx, :],
            quat_matrix[rowidx, 0],
            quat_matrix[rowidx, 1],
            quat_matrix[rowidx, 2],
            quat_matrix[rowidx, 3],
        )


@njit(parallel=True)
def multiply_quaternions_list_x_list(
    array_a: npt.NDArray, array_b: npt.NDArray, result: npt.NDArray
) -> None:
    """Multiply two sets of quaternions together

    All the matrices must have the same shape ``(N, 4)``. The result of ``a × b`` is saved
    into `result`."""

    num = array_a.shape[0]
    for i in prange(num):
        a0, a1, a2, a3 = array_a[i, :]
        b0, b1, b2, b3 = array_b[i, :]
        result[i, 0] = a3 * b0 - a2 * b1 + a1 * b2 + a0 * b3
        result[i, 1] = a2 * b0 + a3 * b1 - a0 * b2 + a1 * b3
        result[i, 2] = -a1 * b0 + a0 * b1 + a3 * b2 + a2 * b3
        result[i, 3] = -a0 * b0 - a1 * b1 - a2 * b2 + a3 * b3


@njit(parallel=True)
def multiply_quaternions_list_x_one(
    array_a: npt.NDArray, single_b: npt.NDArray, result: npt.NDArray
) -> None:
    """Multiply a matrix of quaternions by one quaternion: `array_a × single_b`.

    The result of ``a × b`` is saved into `result`."""

    num = array_a.shape[0]
    b0, b1, b2, b3 = single_b[:]
    for i in prange(num):
        a0, a1, a2, a3 = array_a[i, :]
        result[i, 0] = a3 * b0 - a2 * b1 + a1 * b2 + a0 * b3
        result[i, 1] = a2 * b0 + a3 * b1 - a0 * b2 + a1 * b3
        result[i, 2] = -a1 * b0 + a0 * b1 + a3 * b2 + a2 * b3
        result[i, 3] = -a0 * b0 - a1 * b1 - a2 * b2 + a3 * b3


@njit(parallel=True)
def multiply_quaternions_one_x_list(
    single_a: npt.NDArray, array_b: npt.NDArray, result: npt.NDArray
) -> None:
    """Multiply one quaternion by a matrix of quaternions: `single_a × array_b`.

    The result of ``a × b`` is saved into `result`."""

    num = array_b.shape[0]
    a0, a1, a2, a3 = single_a[:]
    for i in prange(num):
        b0, b1, b2, b3 = array_b[i, :]
        result[i, 0] = a3 * b0 - a2 * b1 + a1 * b2 + a0 * b3
        result[i, 1] = a2 * b0 + a3 * b1 - a0 * b2 + a1 * b3
        result[i, 2] = -a1 * b0 + a0 * b1 + a3 * b2 + a2 * b3
        result[i, 3] = -a0 * b0 - a1 * b1 - a2 * b2 + a3 * b3


@njit(parallel=True)
def normalize_quaternions(quat_matrix: npt.NDArray) -> None:
    """Normalize all the quaternions in a matrix with shape (N, 4)

    All the quaternions are normalized in place.
    """
    num, _ = quat_matrix.shape
    for i in prange(num):
        v1, v2, v3, w = quat_matrix[i, :]
        norm = np.sqrt(v1**2 + v2**2 + v3**2 + w**2)
        quat_matrix[i, 0] /= norm
        quat_matrix[i, 1] /= norm
        quat_matrix[i, 2] /= norm
        quat_matrix[i, 3] /= norm


@njit
def quat_rotation(theta_rad, x, y, z):
    """This function rotates a quaternion by theta_rad about a specific axis.
    Args:
        theta_rad (float): The angle to rotate the quaternion by in radians.
        x (float): The x element of vector to rotate the quaternion about.
        y (float): The y element of vector to rotate the quaternion about.
        z (float): The z element of vector to rotate the quaternion about.
    """
    s = np.sin(theta_rad / 2)
    return (x * s, y * s, z * s, np.cos(theta_rad / 2))


def quat_rotation_brdcast(theta_rad, vec_matrix):
    """This function rotates a quaternion by theta_rad about a specific axis.
    Args:
        theta_rad (float): The angle to rotate the quaternion by in radians.
        vec_matrix (numpy.array[N, 3]): The vectors to rotate the quaternion about.
    """
    s = np.sin(theta_rad / 2)
    return np.hstack(
        [vec_matrix * s, np.cos(theta_rad / 2) * np.ones((vec_matrix.shape[0], 1))]
    )


def quat_rotation_x_brdcast(theta_rad_array):
    """Return a quaternion representing a rotation around the x axis

    Prototype::

        quat_rotation_x(theta_rad: np.ndarray) -> np.ndarray

    The angles `theta_rad` must be expressed in radians. The return
    value is the quaternion, using the order ``(v_x, v_y, v_z, w)``;
    it is returned as a 2D array with shape (n, 4), where n is the number of angles.

    See also :func:`quat_rotation_y_brdcast` and :func:`quat_rotation_z_brdcast`
    """
    sin_theta_over_2 = np.sin(theta_rad_array / 2)
    cos_theta_over_2 = np.cos(theta_rad_array / 2)
    return np.stack(
        [
            sin_theta_over_2,
            np.zeros_like(theta_rad_array),
            np.zeros_like(theta_rad_array),
            cos_theta_over_2,
        ],
        axis=-1,
    )


def quat_rotation_y_brdcast(theta_rad_array):
    """Return a quaternion representing a rotation around the y axis

    Prototype::

        quat_rotation_y(theta_rad: np.ndarray) -> np.ndarray

    The angles `theta_rad` must be expressed in radians. The return
    value is the quaternion, using the order ``(v_x, v_y, v_z, w)``;
    it is returned as a 2D array with shape (n, 4), where n is the number of angles.

    See also :func:`quat_rotation_x_brdcast` and :func:`quat_rotation_z_brdcast`
    """
    sin_theta_over_2 = np.sin(theta_rad_array / 2)
    cos_theta_over_2 = np.cos(theta_rad_array / 2)
    return np.stack(
        [
            np.zeros_like(theta_rad_array),
            sin_theta_over_2,
            np.zeros_like(theta_rad_array),
            cos_theta_over_2,
        ],
        axis=-1,
    )


def quat_rotation_z_brdcast(theta_rad_array):
    """Return a quaternion representing a rotation around the z axis

    Prototype::

        quat_rotation_z(theta_rad: np.ndarray) -> np.ndarray

    The angles `theta_rad` must be expressed in radians. The return
    value is the quaternion, using the order ``(v_x, v_y, v_z, w)``;
    it is returned as a 2D array with shape (n, 4), where n is the number of angles.

    See also :func:`quat_rotation_x_brdcast` and :func:`quat_rotation_y_brdcast`
    """
    sin_theta_over_2 = np.sin(theta_rad_array / 2)
    cos_theta_over_2 = np.cos(theta_rad_array / 2)
    return np.stack(
        [
            np.zeros_like(theta_rad_array),
            np.zeros_like(theta_rad_array),
            sin_theta_over_2,
            cos_theta_over_2,
        ],
        axis=-1,
    )
