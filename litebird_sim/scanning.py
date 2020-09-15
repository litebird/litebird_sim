# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union
from uuid import UUID

from astropy.coordinates import ICRS, get_body_barycentric, BarycentricMeanEcliptic
import astropy.time
import astropy.units as u
from numba import njit
import numpy as np

from ducc0.pointingprovider import PointingProvider

from .imo import Imo

from .quaternions import (
    quat_rotation_x,
    quat_rotation_y,
    quat_rotation_z,
    quat_left_multiply,
    rotate_x_vector,
    rotate_z_vector,
)

YEARLY_OMEGA_SPIN_HZ = 2 * np.pi / (1.0 * u.year).to(u.s).value


@njit
def _clip_sincos(x):
    # Unfortunately, Numba 0.51 does not support np.clip, so we must
    # roll our own version (see
    # https://jcristharif.com/numba-overload.html)
    return min(max(x, -1), 1)


@njit
def polarization_angle(theta_rad, phi_rad, poldir):
    """Compute the polarization angle at a given point on the sky

    Prototype::

        polarization_angle(
            theta_rad: float,
            phi_rad: float,
            poldir: numpy.array[3], 
        )

    This function returns the polarization angle (in radians) with
    respect to the North Pole of the celestial sphere for the point at
    coordinates `theta_rad` (colatitude, in radians) and `phi_rad`
    (longitude, in radians), assuming that `poldir` is a 3-element
    NumPy array representing a normalized vector which departs from
    the point on the celestial sphere and is aligned with the
    polarization direction.

    """
    # We want here to associate a polarization angle with a specific
    # direction in the sky P = (θ, ϕ) and a polarization direction,
    # which is a vector of length one starting from P. To compute the
    # polarization angle with respect to a fixed frame on the
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
    # To compute the polarization angle, we're just looking at the dot
    # product between "poldir" and these two directions. We use
    # `clip_sincos` to prevent problems from values that are slightly
    # outside the allowed range [-1,1] because of numerical roundoff
    # errors.

    cos_psi = _clip_sincos(-np.sin(phi_rad) * poldir[0] + np.cos(phi_rad) * poldir[1])
    sin_psi = _clip_sincos(
        (-np.cos(theta_rad) * np.cos(phi_rad) * poldir[0])
        + (-np.cos(theta_rad) * np.sin(phi_rad) * poldir[1])
        + (np.sin(theta_rad) * poldir[2])
    )
    return np.arctan2(sin_psi, cos_psi)


@njit
def compute_pointing_and_polangle(result, quaternion):
    """Store in "result" the pointing direction and polarization angle.

    Prototype::

        compute_pointing_and_polangle(
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

    - ``result[2]``: the polarization angle (assuming that in the beam
      reference frame points towards x), measured with respect to the
      North and East directions in the celestial sphere

    This function does *not* support broadcasting; use
    :func:`all_compute_pointing_and_polangle` if you need to
    transform several quaternions at once.

    Example::

        import numpy as np
        result = np.empty(3)
        compute_pointing_and_polangle(result, np.array([
            0.0, np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2,
        ])

    """

    vx, vy, vz, w = quaternion

    # Dirty trick: as "result" is a vector of three floats (θ, φ, ψ),
    # we're reusing it over and over again to compute intermediate
    # vectors before the final result. First, we use it to compute the
    # (x, y, z) pointing direction
    rotate_z_vector(result, vx, vy, vz, w)
    theta_pointing = np.arccos(result[2])
    phi_pointing = np.arctan2(result[1], result[0])

    # Now reuse "result" to compute the polarization direction
    rotate_x_vector(result, vx, vy, vz, w)

    # Compute the polarization angle
    pol_angle = polarization_angle(
        theta_rad=theta_pointing, phi_rad=phi_pointing, poldir=result
    )

    # Finally, set "result" to the true result of the computation
    result[0] = theta_pointing
    result[1] = phi_pointing
    result[2] = pol_angle


@njit
def all_compute_pointing_and_polangle(result_matrix, quat_matrix):
    """Repeatedly apply :func:`compute_pointing_and_polangle`

    Prototype::

        all_compute_pointing_and_polangle(
            result_matrix: numpy.array[N, 3],
            quat_matrix: numpy.array[N, 4],
        )

    Assuming that `result_matrix` is a (N×3) matrix and `quat_matrix`
    a (N×4) matrix, iterate over all the N rows and apply
    :func:`compute_pointing_and_polangle` to every row.

    """
    assert quat_matrix[..., 0].size == result_matrix[..., 0].size
    result_matrix = result_matrix.reshape(-1, 3)
    quat_matrix = quat_matrix.reshape(-1, 4)

    for row in range(result_matrix.shape[0]):
        compute_pointing_and_polangle(result_matrix[row, :], quat_matrix[row, :])


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


@njit
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
    for row in range(result_matrix.shape[0]):
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
        coord = ICRS(pos).transform_to(BarycentricMeanEcliptic).cartesian
        return np.arctan2(coord.x.value, coord.y.value)
    else:
        return YEARLY_OMEGA_SPIN_HZ * time_vector


class Spin2EclipticQuaternions:
    """A matrix of quaternions sampled uniformly over time

    This class is used to hold quaternions that represent the
    transformation from the reference frame of the LiteBIRD spin axis
    to the Ecliptic reference frame.

    The class has the following members:

    - ``start_time`` is either a floating-point number or an
      ``astropy.time.Time`` object.

    - ``pointing_freq_hz`` is the sampling frequency of the
      quaternions, in Hertz

    - ``quats`` is a NumPy array of shape ``(N × 4)``, containing the
      ``N`` (normalized) quaternions

    """

    def __init__(
        self,
        start_time: Union[float, astropy.time.Time],
        pointing_freq_hz: float,
        quats,
    ):
        self.start_time = start_time
        self.pointing_freq_hz = pointing_freq_hz
        self.quats = quats

    def nbytes(self):
        "Return the number of bytes allocated for the quaternions"
        return self.quats.nbytes

    def get_detector_quats(
        self,
        detector_quat,
        bore2spin_quat,
        time0: Union[float, astropy.time.Time],
        sampling_rate_hz: float,
        nsamples: int,
    ):
        """Return detector-to-Ecliptic quaternions

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
        class :class:`.Instrument`, which has the field
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
        assert len(detector_quat) == 4
        assert (
            self.quats.shape[0] > 1
        ), "having only one quaternion is still unsupported"

        if isinstance(self.start_time, astropy.time.Time):
            assert isinstance(time0, astropy.time.Time), (
                "you must pass an astropy.time.Time object to time0 here, as "
                "Spin2EclipticQuaternions.start_time = {}"
            ).format(self.start_time)

            time_skip_s = (time0 - self.start_time).sec
        else:
            time_skip_s = time0 - self.start_time

        det2spin_quat = np.copy(detector_quat)
        quat_left_multiply(det2spin_quat, *bore2spin_quat)

        pp = PointingProvider(0.0, self.pointing_freq_hz, self.quats)
        # TODO: use the "right" (as opposed to "left") form of this
        # call, once https://github.com/litebird/ducc/issues/3 is
        # solved.
        return pp.get_rotated_quaternions(
            time_skip_s, sampling_rate_hz, det2spin_quat, nsamples, rot_left=False,
        )


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
    ) -> Spin2EclipticQuaternions:
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

        The function returns a :class:`Spin2EclipticQuaternions`
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

    - `spin_sun_angle_deg`: angle between the spin axis and the
      Sun-LiteBIRD direction (floating-point number, in degrees)

    - `precession_period_min`: the period of the precession rotation
      (floating-point number, in minutes)

    - `spin_rate_rpm`: the number of rotations per minute (RPM) around
      the spin axis (floating-point number)

    - `start_time`: an ``astropy.time.Time`` object representing the
      start of the observation. It's currently unused, but it is meant
      to represent the time when the rotation starts (i.e., the angle
      ωt is zero).

    Once the object is created, the following fields are available:

    - `spin_sun_angle_rad`: the same as `spin_sun_angle_deg`, but in
      radians

    - `precession_rate_hz`: the frequency of the precession rotation,
      in Hertz, or zero if no precession occurs (i.e.,
      `precession_period_min` is zero)

    - `spin_rate_hz`: the frequency of the spin rotation, in Hertz

    - `start_time`: see above

    You can create an instance of this class using the class method
    :meth:`.from_imo`, which reads the
    parameters from the IMO.

    """

    def __init__(
        self,
        spin_sun_angle_deg,
        precession_period_min,
        spin_rate_rpm,
        start_time=astropy.time.Time("2027-01-01", scale="tdb"),
    ):
        self.spin_sun_angle_rad = np.deg2rad(spin_sun_angle_deg)
        if precession_period_min > 0:
            self.precession_rate_hz = 1.0 / (60.0 * precession_period_min)
        else:
            self.precession_rate_hz = 0.0
        self.spin_rate_hz = spin_rate_rpm / 60.0
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
    rotations around the spinning axis:            {spin_rate_min} rot/hr
    start time of the simulation:                  {start_time}""".format(
            spin_sun_angle_deg=np.rad2deg(self.spin_sun_angle_rad),
            precession_rate_hr=3600.0 * self.precession_rate_hz,
            spin_rate_min=3600.0 * self.spin_rate_hz,
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
            spin_sun_angle_deg=obj.metadata["spin_sun_angle_deg"],
            precession_period_min=obj.metadata["precession_period_min"],
            spin_rate_rpm=obj.metadata["spin_rate_rpm"],
        )

    def generate_spin2ecl_quaternions(
        self,
        start_time: Union[float, astropy.time.Time],
        time_span_s: float,
        delta_time_s: float,
    ) -> Spin2EclipticQuaternions:
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

        return Spin2EclipticQuaternions(
            start_time=start_time,
            pointing_freq_hz=pointing_freq_hz,
            quats=spin2ecliptic_quats,
        )
