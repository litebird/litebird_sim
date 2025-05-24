# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union, List, Optional
from uuid import UUID

from astropy.coordinates import ICRS, get_body_barycentric
import astropy.time
import astropy.units as u
from numba import njit, prange
import numpy as np
import numpy.typing as npt

from ducc0.pointingprovider import PointingProvider as DuccPointingProvider

from .coordinates import DEFAULT_COORDINATE_SYSTEM
from .imo import Imo

from .quaternions import (
    quat_rotation_x,
    quat_rotation_y,
    quat_rotation_z,
    quat_left_multiply,
    quat_right_multiply,
    rotate_x_vector,
    rotate_z_vector,
    multiply_quaternions_list_x_list,
    multiply_quaternions_list_x_one,
    multiply_quaternions_one_x_list,
    normalize_quaternions,
)

YEARLY_OMEGA_SPIN_HZ = 2 * np.pi / (1.0 * u.year).to(u.s).value


@njit
def _clip_sincos(x):
    # np.clip would be perfect here, but it is slightly inefficient to use here
    # because `x` might either be a float64 or a float32, and we would need to
    # define the −1 and 1 constant accordingly. Better to resort to a combination
    # of calls to min/max, as they are efficiently optimized by LLVM.
    return min(max(x, -1), 1)


@njit
def orientation_angle(theta_rad, phi_rad, ordir):
    """Compute the orientation of a detector at a given point on the sky

    Prototype::

        orientation_angle(
            theta_rad: float,
            phi_rad: float,
            ordir: numpy.array[3],
        )

    This function returns the orientation (in radians) with respect
    to the North Pole of the celestial sphere for the point at
    coordinates `theta_rad` (colatitude, in radians) and `phi_rad`
    (longitude, in radians), assuming that `ordir` is a 3-element
    NumPy array representing a normalized vector which departs from
    the point on the celestial sphere and is aligned with the
    orientation direction.

    """
    # We want here to associate a orientation with a specific
    # direction in the sky P = (θ, ϕ) and a orientation direction,
    # which is a vector of length one starting from P. To compute the
    # orientation angle with respect to a fixed frame on the
    # celestial sphere, we need first to derive the two vectors
    # pointing towards North and East.
    #
    # Consider the following (ugly) figure, which shows how the North
    # is computed for the point at P:
    #
    #                 z axis ^    North direction
    #                        ▪.   .
    #                       .■▪   ..
    #                        |    .+
    #                  ......+.++..+
    #               ....     |   ...+...
    #            ...         |    ..+   ..
    #          ...           |      ▪     ..
    #         ..             |      .+      ..
    #        +               |       +       ..
    #       +                |---    +▪. P    ..
    #      ..                |   \ ...+        +
    #      +                 | θ  .. ..         .
    #     ..                 |  ..    .         .
    #     +                  |..      ..        .
    #  ---+----------------.+▪--------.+--------+--> y axis
    #     .              ... |        ..        .
    #      .          ...    |        .         +
    #      +        ...      |        .        ..
    #       .    ...         |        .        +
    #       .....            |       ..       +
    #      ++++              |       +       .
    #     L    ..            |      +      ..
    #   x axis  ...          |     ..    ..
    #             ...        |    ..  ...
    #                ......  | ..+....
    #                      ..+...
    #
    # Since the vector v is
    #
    #   v = [sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)]
    #
    # then North is -dv/dθ and East is dv/dϕ, which leads to
    #
    #   North = [-cos(θ) * cos(ϕ), -cos(θ) * sin(ϕ), sin(θ)]
    #   East  = [-sin(ϕ), cos(ϕ), 0]
    #
    # To compute the orientation angle, we're just looking at the dot
    # product between "ordir" and these two directions. We use
    # `clip_sincos` to prevent problems from values that are slightly
    # outside the allowed range [-1,1] because of numerical roundoff
    # errors.

    cos_psi = _clip_sincos(-np.sin(phi_rad) * ordir[0] + np.cos(phi_rad) * ordir[1])
    sin_psi = _clip_sincos(
        (-np.cos(theta_rad) * np.cos(phi_rad) * ordir[0])
        + (-np.cos(theta_rad) * np.sin(phi_rad) * ordir[1])
        + (np.sin(theta_rad) * ordir[2])
    )
    return np.arctan2(sin_psi, cos_psi)


@njit
def compute_pointing_and_orientation(result, quaternion):
    """Store in "result" the pointing direction and orientation angle.

    Prototype::

        compute_pointing_and_orientation(
            result: numpy.array[3],
            quaternion: numpy.array[4],
        )

    The function assumes that `quaternion` encodes a rotation which
    transforms the z axis into the direction of a beam in the sky,
    i.e., it assumes that the beam points towards z in its own
    reference frame and that `quaternion` transforms the reference
    frame to celestial coordinates.

    The variable `result` is used to save the result of the
    computation, and it should be a 3-element NumPy array. On exit,
    its values will be:

    - ``result[0]``: the colatitude of the sky direction, in radians

    - ``result[1]``: the longitude of the sky direction, in radians

    - ``result[2]``: the orientation angle (assuming that in the beam
      reference frame points towards x), measured with respect to the
      North and East directions in the celestial sphere

    This function does *not* support broadcasting; use
    :func:`all_compute_pointing_and_orientation` if you need to
    transform several quaternions at once.

    Example::

        import numpy as np
        result = np.empty(3)
        compute_pointing_and_orientation(result, np.array([
            0.0, np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2,
        ])

    """

    vx, vy, vz, w = quaternion

    # Dirty trick: as "result" is a vector of three floats (θ, φ, ψ),
    # we're reusing it over and over again to compute intermediate
    # vectors before the final result. First, we use it to compute the
    # (x, y, z) pointing direction
    rotate_z_vector(result, vx, vy, vz, w)
    theta_pointing = np.arctan2(np.sqrt(result[0] ** 2 + result[1] ** 2), result[2])
    phi_pointing = np.arctan2(result[1], result[0])

    # Now reuse "result" to compute the orientation
    rotate_x_vector(result, vx, vy, vz, w)

    # Compute the orientation angle
    orientation = orientation_angle(
        theta_rad=theta_pointing, phi_rad=phi_pointing, ordir=result
    )

    # Finally, set "result" to the true result of the computation
    result[0] = theta_pointing
    result[1] = phi_pointing
    result[2] = orientation


@njit(parallel=True)
def all_compute_pointing_and_orientation(result_matrix, quat_matrix):
    """Repeatedly apply :func:`compute_pointing_and_orientation`

    Prototype::

        all_compute_pointing_and_orientation(
            result_matrix: numpy.ndarray[N, 3],
            quat_matrix: numpy.ndarray[N, 4],
        )

    Assuming that `result_matrix` is a (N, 3) matrix and `quat_matrix` a (N, 4)
    matrix, iterate over all the N samples and apply
    :func:`compute_pointing_and_orientation` to every item.

    """

    n_samples = result_matrix.shape[0]

    for sample_idx in prange(n_samples):
        compute_pointing_and_orientation(
            result_matrix[sample_idx, :],
            quat_matrix[sample_idx, :],
        )


@njit
def spin_to_ecliptic(
    result,
    sun_earth_angle_rad,
    spin_sun_angle_rad,
    precession_rate_hz,
    spin_rate_hz,
    time_s,
):
    """Compute a quaternion with the spin-axis-to-Ecliptic rotation

    Prototype::

        spin_to_ecliptic(
            result: numpy.array[4],
            sun_earth_angle_rad: float,
            spin_sun_angle_rad: float,
            precession_rate_hz: float,
            spin_rate_hz: float,
            time_s: float,
        )

    This function computes the (normalized) quaternion that encodes
    the rotation which transforms the frame of reference of the
    spacecraft's spin axis into the Ecliptic frame of reference. The
    result is saved in the parameter `result`, which must be a
    4-element NumPy array; the order of the elements of the quaternion
    is `(vx, vy, vz, w)`.

    The function computes the quaternion as the following sequence of
    rotations:

    1. A rotation around the `z` axis by the angle :math:`2π ν t`,
       with `ν` being the parameter `spin_rate_hz` and `t` the
       parameter `time_s` (this rotation accounts for the rotation of
       the spacecraft around the spin axis)

    2. A rotation around the `y` axis by the angle :math:`π/2 -
       \\alpha`, with `ɑ` being the parameter `spin_sun_angle_rad`
       (this accounts for the inclination of the spin axis with
       respect to the Ecliptic plane)

    3. A rotation around the `x` axis by the angle :math:`2π ν t`,
       with `ν` being the parameter `precession_rate_hz` and `t` the
       parameter `time_s` (this rotation accounts for the rotation of
       the spin axis because of the precessional motion)

    4. A rotation around the `z` axis by the angle
       `sun_earth_angle_rad` (this accounts for the yearly revolution
       of the spacecraft around the Sun)

    Args:

       `sun_earth_angle_rad` (float): Angle between the x axis and the
          Sun-Earth direction on the xy Ecliptic plane (in radians)

       `spin_sun_angle_rad` (float): Angle between the spin axis of
          the spacecraft and the Sun-Earth direction (in radians);
          this angle is sometimes called `ɑ`

       `precession_rate_hz` (float): The frequency of rotations around
          the precession axis (in rotations/sec)

       `spin_rate_hz` (float): The frequency of rotations around the
          spin axis (in rotations/sec)

       `time_s` (float): the time when to compute the quaternion

    """
    result[:] = quat_rotation_z(2 * np.pi * spin_rate_hz * time_s)
    quat_left_multiply(result, *quat_rotation_y(np.pi / 2 - spin_sun_angle_rad))
    quat_left_multiply(
        result, *quat_rotation_x(2 * np.pi * precession_rate_hz * time_s)
    )
    quat_left_multiply(result, *quat_rotation_z(sun_earth_angle_rad))


@njit(parallel=True)
def all_spin_to_ecliptic(
    result_matrix,
    sun_earth_angles_rad,
    spin_sun_angle_rad,
    precession_rate_hz,
    spin_rate_hz,
    time_vector_s,
):
    """Apply :func:`spin_to_ecliptic` to each row of a matrix

    Prototype::

        all_spin_to_ecliptic(
            result_matrix: numpy.array[N, 4],
            sun_earth_angle_rad: float,
            spin_sun_angle_rad: float,
            precession_rate_hz: float,
            spin_rate_hz: float,
            time_vector_s: numpy.array[N],
        )

    This function extends :func:`spin_to_ecliptic` to work with
    the vector of times `time_vector_s`; all the other parameters must
    still be float as in `spin_to_ecliptic`; the variable
    `result_matrix` must be a matrix of shape ``(len(time_vector_s),
    4)``.

    """
    for row in prange(result_matrix.shape[0]):
        spin_to_ecliptic(
            result=result_matrix[row, :],
            sun_earth_angle_rad=sun_earth_angles_rad[row],
            spin_sun_angle_rad=spin_sun_angle_rad,
            precession_rate_hz=precession_rate_hz,
            spin_rate_hz=spin_rate_hz,
            time_s=time_vector_s[row],
        )


def calculate_sun_earth_angles_rad(time_vector):
    """Compute the angle between the x axis and the Earth

    This function computes the angle on the plane of the Ecliptic
    (assuming to be the xy plane) between the Sun-Earth direction and
    the x axis. Depending on the type of the parameter `time_vector`,
    the result is computed differently:

    - If `time_vector` is a ``astropy.time.Time`` object, the angle is
      computed using the Barycentric Mean Ecliptic reference frame and
      the Ephemerides tables provided by AstroPy (slow but accurate)

    - Otherwise, `time_vector` is assumed to be a NumPy array of
      floats, and a simple circular motion with constant angular
      velocity is assumed. The angular velocity is
      ``YEARLY_OMEGA_SPIN_HZ``, which is equal to :math:`2π/T`, with T
      being the average duration of one year in seconds, and it is
      assumed that at time `t = 0` the angle is zero.

    """
    # This is the geometry of the problem: the Ecliptic plane is
    # supposed to be on the xy plane, with the Sun in the origin, and
    # we're looking for the angle θ in the figure:
    #
    #             ┌──────────────────────────────┐
    #           1 │⠀⠀⠀⠀⠀⠀⠀⢀⡠⠤⠒⠊⠉⠉⠉⡏⠉⠉⠉⠒⠤⢄⡀⠀⠀⠀⠀⠀⠀⠀│
    #             │⠀⠀⠀⠀⠀⡠⠊⠁⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠈⠓⢄⠀⠀⠀⠀⠀│
    #             │⠀⠀⠀⡰⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢆⠀⠀⠀│
    #             │⠀⢀⠜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢲⡀⠀│
    #             │⢀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠊⠁⢱⠀│
    #             │⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⢀⡠⠒⠉⠀⠀⠀⠀⠀⢇│
    #             │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⣀⠔⠊⠁⠀⠀θ⠀⠀⠀⠀⠀⢸│
    #   y [AU]    │⡧⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⡷⠭⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢼│
    #             │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇Sun⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│
    #             │⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜│
    #             │⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁│
    #             │⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡴⠁⠀│
    #             │⠀⠀⠀⠱⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠜⠀⠀⠀│
    #             │⠀⠀⠀⠀⠀⠑⢄⡀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⢀⡤⠒⠁⠀⠀⠀⠀│
    #          -1 │⠀⠀⠀⠀⠀⠀⠀⠈⠑⠢⠤⣀⣀⣀⣀⣇⣀⣀⡠⠤⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀│
    #             └──────────────────────────────┘
    #             -1                             1
    #                         x [AU]
    #
    # where of course θ points towards the Earth. The value of the
    # angle depends whether we're using proper MJD dates (in this
    # case, we need ephemeridis tables) or not: in the latter case we
    # just assume that at time 0 the angle θ is zero, and that the
    # Earth follows a uniform circular motion around the Sun with
    # frequency ω = 2πν = YEARLY_OMEGA_SPIN_HZ.

    if isinstance(time_vector, astropy.time.Time):
        pos = get_body_barycentric("earth", time_vector)
        coord = ICRS(pos).transform_to(DEFAULT_COORDINATE_SYSTEM).cartesian
        return np.arctan2(coord.y.value, coord.x.value)
    else:
        return YEARLY_OMEGA_SPIN_HZ * time_vector


class RotQuaternion:
    """A matrix of quaternions sampled uniformly over time that encode rotations

    This class is used to hold quaternions that represent the
    rotation from the reference frame of the LiteBIRD spin axis
    to the Ecliptic reference frame. All the quaternions are normalized,
    i.e., they encode a rotation in 3D space.

    The class has the following members:

    - ``quats`` is a NumPy array of shape ``(N × 4)``, containing the
      ``N`` (normalized) quaternions. All the quaternions are guaranteed
      to be normalized.

    - ``start_time`` is either a floating-point number or an
      ``astropy.time.Time`` object. It can be ``None`` if and only
      if there is just *one* quaternion in ``quats``.

    - ``sampling_rate_hz`` is the sampling frequency of the
      quaternions, in Hertz. It can be ``None`` if and only
      if there is just *one* quaternion in ``quats``.

    """

    def __init__(
        self,
        quats: Union[npt.ArrayLike, "RotQuaternion", None] = None,
        start_time: Optional[Union[float, astropy.time.Time]] = None,
        sampling_rate_hz: Optional[float] = None,
    ):
        """
        Create a new instance of a time-dependent quaternion

        If both `start_time` and `sampling_freq_hz` are ``None``, the quaternion
        is assumed to be constant in time.

        :param quats: Either a 4-element NumPy array, or another instance
            of :class:`TimeDependentQuaternion`. Its default value is the
            identity rotation
        :param start_time: the start time, either a floating point number
            or an ``astropy.time.Time`` object
        :param sampling_rate_hz: the sampling frequency
        """

        if quats is None:
            quats = np.array([[0.0, 0.0, 0.0, 1.0]])

        if isinstance(quats, RotQuaternion):
            self.quats = quats.quats
            self.start_time = quats.start_time
            self.sampling_rate_hz = quats.sampling_rate_hz
        else:
            # Make sure that `quats` is always a 2D array with shape (N, 4),
            # even if it was originally a 1D array with shape (4,)
            self.quats = quats.reshape(-1, 4)

            if self.quats.shape[0] > 1:
                assert start_time is not None, (
                    "You must specify start_time if the quaternion is not constant"
                )
                assert sampling_rate_hz is not None, (
                    "You must specify sampling_rate_hz if the quaternion is not constant"
                )

            self.start_time = start_time
            self.sampling_rate_hz = sampling_rate_hz

        # Normalize the quaternions
        normalize_quaternions(self.quats)

    def nbytes(self):
        "Return the number of bytes allocated for the quaternions"
        return self.quats.nbytes

    def __mul__(self, other: "RotQuaternion") -> "RotQuaternion":
        """Multiply two time-dependent quaternions together

        The two operands must have the same value for `start_time`
        and `pointing_freq_hz`, and they must contain the same number
        of quaternions.
        """

        if (self.start_time is not None) and (other.start_time is not None):
            # Both time-dependent quaternions are actual lists, so we must
            # first ensure that they have the same starting time and the
            # same sampling frequency
            assert self.quats.shape == other.quats.shape, (
                f"quaternions have different shapes: {self.quats.shape} vs {other.quats.shape}"
            )
            assert isinstance(self.start_time, type(other.start_time)), (
                f"start_time must have the same type: {type(self.start_time)} vs {type(other.start_time)}"
            )
            if isinstance(self.start_time, float):
                assert np.isclose(self.start_time, other.start_time), (
                    f"start_time must be the same value: {self.start_time} vs {other.start_time}"
                )
            else:
                assert self.start_time == other.start_time, (
                    f"start_time must be the same value: {self.start_time} vs {other.start_time}"
                )
            assert np.isclose(self.sampling_rate_hz, other.sampling_rate_hz), (
                f"sampling_rate_hz must be the same value: {self.sampling_rate_hz} vs {other.sampling_rate_hz}"
            )

            result_start_time = self.start_time
            result_sampling_freq_hz = self.sampling_rate_hz
            result_quats = np.empty_like(self.quats)

            # This is the most expensive case: we must multiply each pair of
            # quaternions one by one
            multiply_quaternions_list_x_list(
                array_a=self.quats, array_b=other.quats, result=result_quats
            )
        elif (self.start_time is not None) and (other.start_time is None):
            # The first quaternion (`a`) is time-dependent, but the other one (`b`) is constant
            result_start_time = self.start_time
            result_sampling_freq_hz = self.sampling_rate_hz
            result_quats = np.empty_like(self.quats)
            multiply_quaternions_list_x_one(
                array_a=self.quats, single_b=other.quats[0, :], result=result_quats
            )
        elif (self.start_time is None) and (other.start_time is not None):
            # The first quaternion (`a`) is constant, but the other one (`b`) is time-dependent
            result_start_time = other.start_time
            result_sampling_freq_hz = other.sampling_rate_hz
            result_quats = np.empty_like(other.quats)
            multiply_quaternions_one_x_list(
                single_a=self.quats[0, :], array_b=other.quats, result=result_quats
            )
        else:
            # Both quaternions are constant in time
            result_start_time = None
            result_sampling_freq_hz = None
            result_quats = np.copy(self.quats)
            quat_right_multiply(result_quats[0, :], *other.quats[0, :])

        return RotQuaternion(
            start_time=result_start_time,
            sampling_rate_hz=result_sampling_freq_hz,
            quats=result_quats,
        )

    def conj(self):
        return RotQuaternion(
            start_time=self.start_time,
            sampling_rate_hz=self.sampling_rate_hz,
            quats=np.array(
                [
                    -self.quats[:, 0],
                    -self.quats[:, 1],
                    -self.quats[:, 2],
                    self.quats[:, 3],
                ]
            ),
        )

    def slerp(
        self,
        start_time: Union[float, astropy.time.Time],
        sampling_rate_hz: float,
        nsamples: int,
    ):
        """Oversample the quaternion using a “slerp” operation

        This method combines the spin-axis-to-Ecliptic quaternions in
        ``self.quat`` with two additional rotations (`detector_quat`,
        `bore2spin_quat`), representing the transformation from the
        reference frame of a detector to the boresight reference frame
        and the transformation from the boresight to the spin
        reference frame. The result is a quaternion that directly
        transforms the reference frame of the detector to Ecliptic
        coordinates.

        Usually, the parameter `detector_quat` is read from the IMO,
        and the parameter `bore2spin_quat` is calculated through the
        class :class:`.InstrumentInfo`, which has the field
        ``bore2spin_quat``. If all you have is the angle β (in
        radians) between the boresight and the spin axis, just pass
        ``quat_rotation_y(β)`` here.

        As this kind of quaternion is used to compute the pointings of
        a detector, which are used in map-making, it applies a «slerp»
        operation on the quaternion, oversampling them to the sampling
        frequency of the detector, expressed through the parameter
        `sampling_rate_hz`.

        The parameters `time0` and `nsamples` specify which is the
        time interval that needs to be covered by the quaternions
        computed by this method. The type of the parameter `time0`
        must match that of `self.start_time`.

        """
        assert self.quats.shape[0] > 1, (
            "having only one quaternion is still unsupported"
        )

        if isinstance(self.start_time, astropy.time.Time):
            assert isinstance(start_time, astropy.time.Time), (
                "you must pass an astropy.time.Time object to time0 here, as "
                "TimeDependentQuaternion.start_time = {}"
            ).format(self.start_time)

            time_skip_s = (start_time - self.start_time).sec
        else:
            time_skip_s = start_time - self.start_time

        pp = DuccPointingProvider(0.0, self.sampling_rate_hz, self.quats)
        return pp.get_rotated_quaternions(
            time_skip_s,
            sampling_rate_hz,
            np.array([0.0, 0.0, 0.0, 1.0]),
            nsamples,
            rot_left=False,
        )

    def is_close_to(self, other: "RotQuaternion") -> bool:
        if self.quats.shape != other.quats.shape:
            return False

        if not isinstance(self.start_time, type(other.start_time)):
            return False

        if isinstance(self.start_time, float):
            if not np.isclose(self.start_time, other.start_time):
                return False
        elif self.start_time is not None:
            if self.start_time != other.start_time:
                return False

        if not isinstance(self.sampling_rate_hz, type(other.sampling_rate_hz)):
            # This catches the case where one variable is None but the other is not
            return False

        if (self.sampling_rate_hz is not None) and not np.isclose(
            self.sampling_rate_hz, other.sampling_rate_hz
        ):
            return False

        return np.allclose(self.quats, other.quats)


# This is an Abstract Base Class (ABC)
class ScanningStrategy(ABC):
    """A class that simulate a scanning strategy

    This is an abstract base class; you should probably use
    :class:`SpinningScanningStrategy`, unless you are interested in
    simulating other kinds of scanning strategies. If this is the
    case, refer to the documentation.

    """

    @abstractmethod
    def generate_spin2ecl_quaternions(
        self,
        start_time: Union[float, astropy.time.Time],
        time_span_s: float,
        delta_time_s: float,
    ) -> RotQuaternion:
        """Generate the quaternions for spin-axis-to-Ecliptic rotations

        This method simulates the scanning strategy of the spacecraft
        assuming that the mission begins at some time `start_time` and
        lasts for `time_span_s` seconds. The purpose of the function
        is to compute the orientation of the spacecraft once every
        `delta_time_s` seconds for the whole duration of the mission;
        the orientation is expressed as a quaternion that encodes the
        rotation from the reference frame of the spacecraft's spin
        axis (aligned with the y axis) to the reference frame of the
        Ecliptic Coordinate System.

        The function returns a :class:`TimeDependentQuaternion`
        object that fully covers the time interval between
        `start_time` and `start_time + time_span_s`: this means that
        an additional quaternion *after* the time ``t_end = start_time
        + time_span_s`` might be appended.

        Args:

            start_time (Union[float, astropy.time.Time]): start time
                of the simulation. If it is a floating-point number,
                it is arbitrary and can usually be set to 0.0;
                otherwise, it must be a ``astropy.time.Time`` object,
                and in this case a more precise computation of the
                orientation of the spacecraft is used. Depending on
                the duration of the simulation, the second case can be
                a few orders of magnitude slower: it should be used
                only when the simulation needs to track the position
                of moving objects (e.g., planets).

            time_span_s (float): interval of time that needs to be
                simulated, in seconds. These seconds are added to
                `start_time`, and their meaning depends on its type:
                if `start_time` is a float, you should consider the
                duration as a sidereal time, but if it's a
                `astropy.time.Time` time, `time_span_s` is expressed
                as a Solar time.

            delta_time_s (float): for efficiency, quaternions are not
                sampled at the same sample rate as the scientific
                data, but at a much lower rate. The default should be
                good for all the most common cases, but you can tune
                it with this parameter.

        """
        pass

    @staticmethod
    def optimal_num_of_quaternions(time_span_s: float, delta_time_s: float) -> int:
        """Return the number of quaternions to compute

        Given a time span and a time interval between consecutive
        quaternions, this static method computes how many quaternions
        are needed to properly cover the time span.
        """
        num_of_quaternions = int(time_span_s / delta_time_s) + 1
        if delta_time_s * (num_of_quaternions - 1) < time_span_s:
            num_of_quaternions += 1

        return num_of_quaternions

    @staticmethod
    def get_times(
        start_time: Union[float, astropy.time.Time],
        delta_time_s: float,
        num_of_quaternions: int,
    ):
        """Return a vector of equally-separated times

        Depending on the type of the parameter `start_time` (either a
        ``float`` or a ``astropy.time.Time`` instance), return a
        vector of times that mark the instant when a quaternion must
        be computed by the class.

        The class returns a 2-element tuple, containing (1) the time
        expressed using the same type as `start_time` (either
        ``float`` or ``astropy.time.Time``), and (2) a vector
        containing the time measured in seconds. The latter is useful
        when your scanning strategy depends on the time for the
        computation of angles and rotation, e.g., if you need to
        compute :math:`2\\pi\\nu t`.

        """
        if isinstance(start_time, astropy.time.Time):
            delta_time = astropy.time.TimeDelta(
                np.arange(num_of_quaternions) * delta_time_s, format="sec", scale="tdb"
            )
            time_s = delta_time.sec
            time = start_time + delta_time
        else:
            time_s = start_time + np.arange(num_of_quaternions) * delta_time_s
            time = time_s

        return time, time_s


class SpinningScanningStrategy(ScanningStrategy):
    """A class containing the parameters of the sky scanning strategy

    This class is used to hold together the parameters that define the
    nominal scanning strategy of the LiteBIRD spacecraft. It's a
    simple scanning strategy that closely matches the ones proposed
    for other CMB experiments from space like CORE and Pico: a
    spinning motion of the spacecraft around some axis, composed with
    a precession motion around the Sun-Earth-spacecraft axis (assuming
    that the spacecraft flies around the L_2 point of the Sun-Earth
    system).

    The constructor accepts the following parameters:

    - `spin_sun_angle_rad`: angle between the spin axis and the
      Sun-LiteBIRD direction (floating-point number, in radians)

    - `precession_rate_hz`: the period of the precession rotation
      (floating-point number, in minutes)

    - `spin_rate_hz`: the number of rotations per minute (RPM) around
      the spin axis (floating-point number)

    - `start_time`: an ``astropy.time.Time`` object representing the
      start of the observation. It's currently unused, but it is meant
      to represent the time when the rotation starts (i.e., the angle
      ωt is zero).

    These fields are available once the object has been initialized.

    You can create an instance of this class using the class method
    :meth:`.from_imo`, which reads the
    parameters from the IMO.

    """

    def __init__(
        self,
        spin_sun_angle_rad,
        precession_rate_hz,
        spin_rate_hz,
        start_time=astropy.time.Time("2027-01-01", scale="tdb"),
    ):
        self.spin_sun_angle_rad = spin_sun_angle_rad
        self.precession_rate_hz = precession_rate_hz
        self.spin_rate_hz = spin_rate_hz
        self.start_time = start_time

    def all_spin_to_ecliptic(self, result_matrix, sun_earth_angles_rad, time_vector_s):
        assert result_matrix.shape == (len(time_vector_s), 4)
        assert len(sun_earth_angles_rad) == len(time_vector_s)

        all_spin_to_ecliptic(
            result_matrix=result_matrix,
            sun_earth_angles_rad=sun_earth_angles_rad,
            spin_sun_angle_rad=self.spin_sun_angle_rad,
            precession_rate_hz=self.precession_rate_hz,
            spin_rate_hz=self.spin_rate_hz,
            time_vector_s=time_vector_s,
        )

    def __repr__(self):
        return (
            "SpinningScanningStrategy(spin_sun_angle_rad={spin_sun_angle_rad}, "
            "precession_rate_hz={precession_rate_hz}, "
            "spin_rate_hz={spin_rate_hz}, "
            "start_time={start_time})".format(
                spin_sun_angle_rad=self.spin_sun_angle_rad,
                precession_rate_hz=self.precession_rate_hz,
                spin_rate_hz=self.spin_rate_hz,
                start_time=self.start_time,
            )
        )

    def __str__(self):
        return """Spinning scanning strategy:
    angle between the Sun and the spin axis:       {spin_sun_angle_deg:.1f}°
    rotations around the precession angle:         {precession_rate_hr} rot/hr
    rotations around the spinning axis:            {spin_rate_hr} rot/hr
    start time of the simulation:                  {start_time}""".format(
            spin_sun_angle_deg=np.rad2deg(self.spin_sun_angle_rad),
            precession_rate_hr=3600.0 * self.precession_rate_hz,
            spin_rate_hr=3600.0 * self.spin_rate_hz,
            start_time=self.start_time,
        )

    @staticmethod
    def from_imo(imo: Imo, url: Union[str, UUID]):
        """Read the definition of the scanning strategy from the IMO

        This function returns a :class:`.SpinningScanningStrategy`
        object containing the set of parameters that define the
        scanning strategy of the spacecraft, i.e., the way it observes
        the sky during the nominal mission.

        Args:

            imo (:class:`.Imo`): an instance of the :class:`.Imo` class

            url (str or ``UUID``): a reference to the data file
                containing the definition of the scanning strategy. It can
                be either a string like
                ``/releases/v1.0/satellite/scanning_parameters/`` or a
                UUID.

        Example::

            imo = Imo()
            sstr = SpinningScanningStrategy.from_imo(
                imo=imo,
                url="/releases/v1.0/satellite/scanning_parameters/",
            )
            print(sstr)

        """
        obj = imo.query(url)
        return SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(obj.metadata["spin_sun_angle_deg"]),
            precession_rate_hz=1.0 / (60.0 * obj.metadata["precession_period_min"]),
            spin_rate_hz=obj.metadata["spin_rate_rpm"] / 60.0,
        )

    def generate_spin2ecl_quaternions(
        self,
        start_time: Union[float, astropy.time.Time],
        time_span_s: float,
        delta_time_s: float,
    ) -> RotQuaternion:
        pointing_freq_hz = 1.0 / delta_time_s
        num_of_quaternions = ScanningStrategy.optimal_num_of_quaternions(
            time_span_s=time_span_s, delta_time_s=delta_time_s
        )

        spin2ecliptic_quats = np.empty((num_of_quaternions, 4))
        time, time_s = ScanningStrategy.get_times(
            start_time=start_time,
            delta_time_s=delta_time_s,
            num_of_quaternions=num_of_quaternions,
        )

        sun_earth_angles_rad = calculate_sun_earth_angles_rad(time)

        self.all_spin_to_ecliptic(
            result_matrix=spin2ecliptic_quats,
            sun_earth_angles_rad=sun_earth_angles_rad,
            time_vector_s=time_s,
        )

        return RotQuaternion(
            start_time=start_time,
            sampling_rate_hz=pointing_freq_hz,
            quats=spin2ecliptic_quats,
        )


def get_quaternion_buffer_shape(observations, num_of_detectors=None):
    """Return the shape of the buffer used to hold detector quaternions.

    This function can be used to pre-allocate the buffer used by
    :func:`.get_det2ecl_quaternions` and :func:`.get_ecl2det_quaternions` to
    save the quaternions representing the change of the orientation of the
    detectors with time.

    Here is a typical use::

        import numpy as np
        import litebird_sim as lbs
        observations = lbs.Observation(...)
        bufshape = get_quaternion_buffer_shape(observations, n_detectors)
        quaternions = np.empty(bufshape, dtype=np.float64)
        quats = get_det2ecl_quaternions(
            ...,
            quaternion_buffer=quaternions,
        )

    """

    if not num_of_detectors:
        num_of_detectors = observations.n_detectors

    return (observations.n_samples, num_of_detectors, 4)


def get_det2ecl_quaternions(
    observations,
    spin2ecliptic_quats: RotQuaternion,
    detector_quats: List[RotQuaternion],
    bore2spin_quat: RotQuaternion,
    quaternion_buffer=None,
    dtype=np.float64,
):
    """Return the detector-to-Ecliptic quaternions

    This function returns a ``(D, N, 4)`` tensor containing the
    quaternions that convert a vector in detector's coordinates
    into the frame of reference of the Ecliptic. The number of
    quaternions is equal to the number of samples hold in this
    observation, ``observations.n_samples``.
    Given that the z axis in the frame of reference of a detector
    points along the main beam axis, this means that if you use
    these quaternions to rotate the vector `z = [0, 0, 1]`, you
    will end up with the sequence of vectors pointing towards the
    points in the sky (in Ecliptic coordinates) that are observed
    by the detector.
    This is a low-level function; you should usually call the function
    :func:`.get_pointings`, which wraps this function to compute
    both the pointing direction and the orientation angle.
    See also the method :func:`.get_ecl2det_quaternions`, which
    mirrors this one.
    If you plan to call this function repeatedly, you can save
    some running time by pre-allocating the buffer used to hold
    the quaternions with the parameter `quaternion_buffer`. This
    must be a NumPy floating-point array whose shape can be
    computed using
    :func:`.get_quaternion_buffer_shape`. If you pass
    `quaternion_buffer`, the return value will be a pointer to
    this buffer.
    """

    bufshape = get_quaternion_buffer_shape(observations, len(detector_quats))
    if quaternion_buffer is None:
        quaternion_buffer = np.empty(bufshape, dtype=dtype)
    else:
        assert quaternion_buffer.shape == bufshape, (
            f"error, wrong buffer size: {quaternion_buffer.size} != {bufshape}"
        )

    for idx, detector_quat in enumerate(detector_quats):
        complete_quaternion = spin2ecliptic_quats * bore2spin_quat * detector_quat
        quaternion_buffer[:, idx, :] = complete_quaternion.slerp(
            start_time=observations.start_time,
            sampling_rate_hz=observations.sampling_rate_hz,
            nsamples=observations.n_samples,
        )

    return quaternion_buffer


def get_ecl2det_quaternions(
    observations,
    spin2ecliptic_quats: RotQuaternion,
    detector_quats: RotQuaternion,
    bore2spin_quat: RotQuaternion,
    quaternion_buffer=None,
    dtype=np.float64,
):
    """Return the Ecliptic-to-detector quaternions

    This function returns a ``(D, N, 4)`` matrix containing the ``N``
    quaternions of all the ``D`` detectors
    that convert a vector in Ecliptic coordinates into
    the frame of reference of the detector itself. The number of
    quaternions is equal to the number of samples hold in this
    observation.

    This function is useful when you want to simulate how a point
    source is observed by the detector's beam: if you know the
    Ecliptic coordinates of the point sources, you can easily
    derive the location of the source with respect to the
    reference frame of the detector's beam.
    """

    quats = get_det2ecl_quaternions(
        observations=observations,
        spin2ecliptic_quats=spin2ecliptic_quats,
        detector_quats=detector_quats,
        bore2spin_quat=bore2spin_quat,
        quaternion_buffer=quaternion_buffer,
        dtype=dtype,
    )
    quats[..., 0:3] *= -1  # Apply the quaternion conjugate
    return quats
