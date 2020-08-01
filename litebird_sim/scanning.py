# -*- encoding: utf-8 -*-

from astropy.coordinates import ICRS, get_body_barycentric, BarycentricMeanEcliptic
from astropy.time import Time
import astropy.units as u
from numba import njit
import numpy as np

YEARLY_OMEGA_SPIN_HZ = 2 * np.pi / (1.0 * u.year).to(u.s).value


@njit
def qrotation_x(theta):
    return (np.sin(theta / 2), 0.0, 0.0, np.cos(theta / 2))


@njit
def qrotation_y(theta):
    return (0.0, np.sin(theta / 2), 0.0, np.cos(theta / 2))


@njit
def qrotation_z(theta):
    return (0.0, 0.0, np.sin(theta / 2), np.cos(theta / 2))


@njit
def quat_multiply(result, other_v1, other_v2, other_v3, other_w):
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
def _cross(result, v0, v1, v2, w0, w1, w2):
    result[0] = v1 * w2 - v2 * w1
    result[1] = v2 * w0 - v0 * w2
    result[2] = v0 * w1 - v1 * w0


@njit
def rotate_vector(result, vx, vy, vz, w, vect):
    # This implements the formula
    #
    #    v' = v + 2q_v ⨯ (q_v ⨯ v + w v)
    #
    # where q_v is the vector part of the quaternion, i.e., [vx vy
    # vz], w the scalar part, v the vector to rotate.
    #
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
def all_rotate_vector(result_matrix, quat_matrix, vect):
    for row in range(result_matrix.shape[0]):
        vx = quat_matrix[row, 0]
        vy = quat_matrix[row, 1]
        vz = quat_matrix[row, 2]
        w = quat_matrix[row, 3]

        rotate_vector(result_matrix[row, :], vx, vy, vz, w, vect)


@njit
def rotate_z_vector(result, vx, vy, vz, w):
    # The same as rotate_vector, but it's faster
    result[0] = 2 * (w * vy + vx * vz)
    result[1] = 2 * (vy * vz - w * vx)
    result[2] = 1.0 - 2 * (vx * vx + vy * vy)


@njit
def all_rotate_z_vector(result_matrix, quat_matrix):
    for row in range(result_matrix.shape[0]):
        vx = quat_matrix[row, 0]
        vy = quat_matrix[row, 1]
        vz = quat_matrix[row, 2]
        w = quat_matrix[row, 3]

        rotate_z_vector(result_matrix[row, :], vx, vy, vz, w)


@njit
def boresight_to_sun_earth_axis(
    result,
    spin_sun_angle_rad,
    spin_boresight_angle_rad,
    precession_rate_hz,
    spin_rate_hz,
    time_s,
):
    result[:] = qrotation_y(spin_boresight_angle_rad)
    quat_multiply(result, *qrotation_z(2 * np.pi * spin_rate_hz * time_s))
    quat_multiply(result, *qrotation_y(np.pi / 2 - spin_sun_angle_rad))
    quat_multiply(result, *qrotation_x(2 * np.pi * precession_rate_hz * time_s))


@njit
def boresight_to_ecliptic(
    result,
    sun_earth_angle_rad,
    spin_sun_angle_rad,
    spin_boresight_angle_rad,
    precession_rate_hz,
    spin_rate_hz,
    time_s,
):
    boresight_to_sun_earth_axis(
        result=result,
        spin_sun_angle_rad=spin_sun_angle_rad,
        spin_boresight_angle_rad=spin_boresight_angle_rad,
        precession_rate_hz=precession_rate_hz,
        spin_rate_hz=spin_rate_hz,
        time_s=time_s,
    )
    quat_multiply(result, *qrotation_z(sun_earth_angle_rad))


@njit
def all_boresight_to_ecliptic(
    result_matrix,
    sun_earth_angles_rad,
    spin_sun_angle_rad,
    spin_boresight_angle_rad,
    precession_rate_hz,
    spin_rate_hz,
    time_vector_s,
):
    for row in range(result_matrix.shape[0]):
        boresight_to_ecliptic(
            result=result_matrix[row, :],
            sun_earth_angle_rad=sun_earth_angles_rad[row],
            spin_sun_angle_rad=spin_sun_angle_rad,
            spin_boresight_angle_rad=spin_boresight_angle_rad,
            precession_rate_hz=precession_rate_hz,
            spin_rate_hz=spin_rate_hz,
            time_s=time_vector_s[row],
        )


def calculate_sun_earth_angles_rad(time_vector, use_mjd=False):
    if use_mjd:
        pos = get_body_barycentric("earth", time_vector)
        coord = ICRS(pos).transform_to(BarycentricMeanEcliptic).cartesian
        return np.arctan2(coord.x.value, coord.y.value)
    else:
        return YEARLY_OMEGA_SPIN_HZ * time_vector


class ScanningStrategy:
    def __init__(
        self,
        spin_sun_angle_deg,
        spin_boresight_angle_deg,
        precession_period_min,
        spin_rate_rpm,
        start_time=Time("2027-01-01", scale="tdb"),
    ):
        self.spin_sun_angle_rad = np.deg2rad(spin_sun_angle_deg)
        self.spin_boresight_angle_rad = np.deg2rad(spin_boresight_angle_deg)
        self.precession_rate_hz = 1.0 / (60.0 * precession_period_min)
        self.spin_rate_hz = spin_rate_rpm * 2 * np.pi / 60.0
        self.start_time = start_time

    def all_boresight_to_ecliptic(
        self, result_matrix, sun_earth_angles_rad, time_vector_s
    ):
        assert result_matrix.shape == (len(time_vector_s), 4)
        assert len(sun_earth_angles_rad) == len(time_vector_s)

        all_boresight_to_ecliptic(
            result_matrix=result_matrix,
            sun_earth_angles_rad=sun_earth_angles_rad,
            spin_sun_angle_rad=self.spin_sun_angle_rad,
            spin_boresight_angle_rad=self.spin_boresight_angle_rad,
            precession_rate_hz=self.precession_rate_hz,
            spin_rate_hz=self.spin_rate_hz,
            time_vector_s=time_vector_s,
        )
