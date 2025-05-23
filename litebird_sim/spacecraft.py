# -*- encoding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, List, Tuple

import astropy
from astropy.coordinates import ICRS, get_body_barycentric_posvel, SkyCoord
import astropy.time

from numba import njit
import numpy as np

from litebird_sim import constants as c
from .coordinates import DEFAULT_COORDINATE_SYSTEM, DEFAULT_TIME_SCALE
from .observations import Observation


def cycles_per_year_to_rad_per_s(x: float) -> float:
    """Convert an angular speed from cycles/yr to rad/s"""
    return x * 2 * np.pi / (365 * 86400)


def get_ecliptic_vec(vec):
    "Convert a coordinate in a XYZ vector expressed in the Ecliptic rest frame"
    return ICRS(vec).transform_to(DEFAULT_COORDINATE_SYSTEM).cartesian.get_xyz()


def compute_l2_pos_and_vel(
    time0: astropy.time.Time, earth_l2_distance_km: float = c.EARTH_L2_DISTANCE_KM
):
    """
    Compute the position and velocity of the L2 Sun-Earth point at a given time.

    The L2 point is not calculated using Lagrange's equations; instead, its
    distance from the Earth must be provided as the parameter `earth_l2_distance_km`.
    The default value is a reasonable estimate. The L2 point is assumed to lie along
    the line that connects the Solar System Barycenter with the Earth's gravitational
    center.

    The return value is a 2-tuple containing two NumPy arrays:

    1. A 3D array containing the XYZ components of the vector specifying the position
       of the L2 point, in km
    2. A 3D array containing the XYZ components of the velocity vector of the L2 point,
       in km/s

    The two vectors are always roughly perpendicular, but they are not exactly due to
    the irregular motion of the Earth (caused by gravitational interactions with other
    Solar System bodies, like the Moon and Jupiter).
    """
    # Use
    #
    #    solar_system_ephemeris.set("builtin")
    #
    # to make AstroPy use the built-in ephemeris table

    earth_pos, earth_vel = get_body_barycentric_posvel("earth", time0)
    earth_pos = ICRS(earth_pos).transform_to(DEFAULT_COORDINATE_SYSTEM).cartesian
    earth_vel = ICRS(earth_vel).transform_to(DEFAULT_COORDINATE_SYSTEM).cartesian

    fudge_factor = earth_l2_distance_km / earth_pos.norm().to("km").value

    # Move from Earth to L2
    l2_pos = earth_pos * (1.0 + fudge_factor)
    l2_vel = earth_vel * (1.0 + fudge_factor)

    return (
        l2_pos.xyz.to("km").value.transpose(),
        l2_vel.xyz.to("km/s").value.transpose(),
    )


@njit
def compute_lissajous_pos_and_vel(
    time0,
    earth_angle_rad,
    earth_ang_speed_rad_s,
    radius1_km,
    radius2_km,
    ang_speed1_rad_s,
    ang_speed2_rad_s,
    phase_rad,
):
    """Compute the position and velocity of the spacecraft assuming a Lissajous orbit

    The position and velocity are calculated in a reference frame centered on the L2
    point, whose axes are aligned with the Solar System Barycenter. This means that
    the position and velocity of the spacecraft with respect to the Solar System
    Barycenter itself can be calculated by summing the result of this function with
    the result of a call to :func:`.compute_l2_pos_and_vel`.
    """

    φ1 = ang_speed1_rad_s * time0
    φ2 = ang_speed2_rad_s * time0
    φ_earth = earth_ang_speed_rad_s * time0

    pos_x, pos_y, pos_z = (
        -radius1_km * np.sin(earth_angle_rad) * np.cos(φ1),
        radius1_km * np.cos(earth_angle_rad) * np.cos(φ1),
        radius2_km * np.sin(φ2 + phase_rad),
    )

    # This is the analytical derivative of the position (see above)
    cos1, sin1 = np.cos(φ1), np.sin(φ1)
    cos_earth, sin_earth = np.cos(φ_earth), np.sin(φ_earth)
    vel_x, vel_y, vel_z = (
        -radius1_km
        * (
            earth_ang_speed_rad_s * cos1 * cos_earth
            - ang_speed1_rad_s * sin1 * sin_earth
        ),
        -radius1_km
        * (
            earth_ang_speed_rad_s * cos1 * sin_earth
            + ang_speed1_rad_s * sin1 * cos_earth
        ),
        ang_speed2_rad_s * radius2_km * np.cos(φ2 + phase_rad),
    )

    return pos_x, pos_y, pos_z, vel_x, vel_y, vel_z


@njit
def sum_lissajous_pos_and_vel(
    pos,
    vel,
    start_time_s,
    end_time_s,
    radius1_km,
    radius2_km,
    ang_speed1_rad_s,
    ang_speed2_rad_s,
    phase_rad,
):
    """Add the position and velocity of a Lissajous orbit to some positions/velocities

    The `pos` and `vel` arrays must have the same shape (N×3, with N being the number
    of position-velocity pairs); the 3D position and velocity of the Lissajous orbit
    will be added to these arrays.

    The `times` array must be an array of *float* values, typically starting from zero,
    and they must measure the time in seconds. All the other parameters are usually
    taken from a :class:`.SpacecraftOrbit` object.

    This function has no return value, as the result is stored in `pos` and `vel`."""

    # This code might look weird if you have never used Numba. However, it is extremely
    # efficient, as it completely avoids memory allocations.

    n_elements = pos.shape[
        0
    ]  # With Numba 0.52, you cannot use len(…) on a NumPy array yet
    time_s = start_time_s
    delta_time_s = (end_time_s - start_time_s) / n_elements

    for idx in range(n_elements):
        l2_pos_x, l2_pos_y, l2_pos_z = pos[idx, :]
        l2_vel_x, l2_vel_y, l2_vel_z = vel[idx, :]

        # The parameter `earth_angle_rad` is the angular position of the Earth (in
        # radians) on the Ecliptic plane with respect to the Vernal Equinox. If we
        # denote with r = (x, y) the projected position of the L2 point on the
        # Ecliptic plane, then the angle is just atan(y/x).
        earth_angle_rad = np.arctan2(l2_pos_y, l2_pos_x)

        # `earth_ang_speed_rad_s` is the angular speed of the Earth on the Ecliptic
        # plane (in rad/s). We calculate it by projecting the motion of L2 on the
        # Ecliptic plane and determining the tangential component of its velocity.
        # The vector perpendicular to it and directed counterclockwise is
        #
        #           | 0   −1 |   | x |   | −y |
        #     e⟂ =  |        | · |   | = |    |,
        #           | 1    0 |   | y |   |  x |
        #
        # which must be normalized, so that e⟂ = (−y, x) / √(x² + y²). Then, the
        # component of the velocity vector aligned along this vector is v·e⟂.
        # However, we are interested in the *angular* speed, so we must use the
        # relation v·e⟂ = ω r, which leads to
        #
        #     ω = v·e⟂ / r
        #       = (v_x, v_y) · (−y / √(x² + y²), x / √(x² + y²)) / √(x² + y²) =
        #       = (−v_x · y + v_y · x) / (x² + y²)
        radius_squared = l2_pos_x**2 + l2_pos_y**2
        earth_ang_speed_rad_s = (
            -l2_vel_x * l2_pos_y + l2_vel_y * l2_pos_x
        ) / radius_squared

        posx, posy, posz, velx, vely, velz = compute_lissajous_pos_and_vel(
            time0=time_s,
            earth_angle_rad=earth_angle_rad,
            earth_ang_speed_rad_s=earth_ang_speed_rad_s,
            radius1_km=radius1_km,
            radius2_km=radius2_km,
            ang_speed1_rad_s=ang_speed1_rad_s,
            ang_speed2_rad_s=ang_speed2_rad_s,
            phase_rad=phase_rad,
        )
        pos[idx][0] += posx
        pos[idx][1] += posy
        pos[idx][2] += posz

        vel[idx][0] += velx
        vel[idx][1] += vely
        vel[idx][2] += velz

        time_s += delta_time_s


@dataclass
class SpacecraftOrbit:
    """A dataclass describing the orbit of the LiteBIRD spacecraft

    This structure has the following fields:

    - `start_time`: Date and time when the spacecraft starts its nominal orbit
    - `earth_l2_distance_km`: distance between the Earth's barycenter and the L2
      point, in km
    - `radius1_km`: first radius describing the Lissajous orbit followed by the
      spacecraft, in km
    - `radius2_km`: second radius describing the Lissajous orbit followed by the
      spacecraft, in km
    - `ang_speed1_rad_s`: first angular speed of the Lissajous orbit, in rad/s
    - `ang_speed2_rad_s`: second angular speed of the Lissajous orbit, in rad/s
    - `phase_rad`: phase difference between the two periodic motions in the Lissajous
      orbit, in radians
    - `solar_velocity_km_s`: velocity of the Sun as estimated from Planck 2018 Solar
      dipole (see arxiv: 1807.06207)
    - `solar_velocity_gal_lat_rad`: galactic latitude direction of the Planck 2018
      Solar dipole
    - `solar_velocity_gal_lon_rad`: galactic longitude direction of the Planck 2018
      Solar dipole

    The default values are the nominal numbers of the orbit followed by WMAP, described
    in Cavaluzzi, Fink & Coyle (2008).
    """

    start_time: astropy.time.Time
    earth_l2_distance_km: float = c.EARTH_L2_DISTANCE_KM
    radius1_km: float = 244_450.0
    radius2_km: float = 137_388.0
    ang_speed1_rad_s: float = cycles_per_year_to_rad_per_s(2.02104)
    ang_speed2_rad_s: float = cycles_per_year_to_rad_per_s(1.98507)
    phase_rad: float = np.deg2rad(-47.944)
    solar_velocity_km_s: float = c.SOLAR_VELOCITY_KM_S
    solar_velocity_gal_lat_rad: float = c.SOLAR_VELOCITY_GAL_LAT_RAD
    solar_velocity_gal_lon_rad: float = c.SOLAR_VELOCITY_GAL_LON_RAD

    solar_velocity_ecl_xyz_km_s = (
        SkyCoord(
            solar_velocity_gal_lon_rad,
            solar_velocity_gal_lat_rad,
            unit="rad",
            frame="galactic",
        )
        .transform_to(DEFAULT_COORDINATE_SYSTEM)
        .cartesian.get_xyz()
        .value
        * solar_velocity_km_s
    )


class SpacecraftPositionAndVelocity:
    """Encode the position/velocity of the spacecraft with respect to the Solar System

    This class contains information that characterize the motion of the spacecraft.
    It is mainly useful to simulate the so-called CMB «orbital dipole» and to properly
    check the visibility of the Sun, the Moon and inner planets.  The coordinate
    system used by this class is the standard Barycentric Ecliptic reference frame.

    The fields of this class are the following:

    - ``orbit``: a :class:`.SpacecraftOrbit` object used to compute the positions and
                 velocities in this object;

    - ``start_time``: the time when the nominal orbit started (a ``astropy.time.Time``
                      object);

    - ``time_span_s``: the time span covered by this object, in seconds;

    - ``positions_km``: a ``N×3`` matrix, representing a list of ``N`` XYZ vectors
                        encoding the position of the spacecraft in the Barycentric
                        Ecliptic reference frame (in kilometers);

    - ``velocities_km_s``: a ``N×3`` matrix, representing a list of ``N`` XYZ vectors
                           encoding the linear velocity of the spacecraft in the
                           Barycentric Ecliptic reference frame (in km/s).

    """

    def __init__(
        self,
        orbit: SpacecraftOrbit,
        start_time: astropy.time.Time,
        time_span_s: float,
        positions_km=None,
        velocities_km_s=None,
    ):
        self.orbit = orbit
        self.start_time = start_time
        self.time_span_s = time_span_s
        self.positions_km = positions_km
        self.velocities_km_s = velocities_km_s

    def __str__(self):
        return (
            "SpacecraftPositionAndVelocity(start_time={0}, "
            "time_span_s={1}, nsamples={2})"
        ).format(self.start_time, self.time_span_s, len(self.positions_km))

    def __repr__(self):
        return str(self)

    def compute_velocities(
        self, time0: astropy.time.Time, delta_time_s: float, num_of_samples: int
    ):
        """Perform a linear interpolation to sample the satellite velocity in time

        Return a N×3 array containing a set of `num_of_samples` 3D vectors with the
        velocity of the spacecraft (in km/s) computed every `delta_time_s` seconds
        starting from time `time0`.
        """
        delta_start_s = (time0 - self.start_time).sec
        t = delta_start_s + np.linspace(
            start=0.0,
            stop=delta_time_s * num_of_samples,
            endpoint=False,
            num=num_of_samples,
        )
        tp = np.linspace(
            start=0.0, stop=self.time_span_s, num=self.velocities_km_s.shape[0]
        )

        velocities = np.empty((num_of_samples, 3))
        velocities[:, 0] = np.interp(x=t, xp=tp, fp=self.velocities_km_s[:, 0])
        velocities[:, 1] = np.interp(x=t, xp=tp, fp=self.velocities_km_s[:, 1])
        velocities[:, 2] = np.interp(x=t, xp=tp, fp=self.velocities_km_s[:, 2])

        return velocities


def compute_start_and_span_for_obs(
    observations: Union[Observation, List[Observation]],
) -> Tuple[astropy.time.Time, float]:
    """
    Compute the start time and the overall duration in seconds of a set of observations.

    The code returns the earliest start time of the observations in `observations` as
    well as their overall time span. Gaps between observations are neglected.
    """

    if isinstance(observations, Observation):
        obs_list = [observations]
    else:
        obs_list = observations

    start_time, end_time = None, None
    for cur_obs in obs_list:
        assert isinstance(cur_obs.start_time, astropy.time.Time), (
            "You must use astropy.time.Time in Observation objects"
        )

        cur_start_time = cur_obs.start_time
        if (start_time is None) or (cur_start_time < start_time):
            start_time = cur_start_time

        cur_end_time = cur_obs.start_time + cur_obs.get_time_span()
        if (end_time is None) or (end_time > cur_end_time):
            end_time = cur_end_time

    time_span_s = (end_time - start_time).to("s").value

    return start_time, time_span_s


def spacecraft_pos_and_vel(
    orbit: SpacecraftOrbit,
    observations: Union[Observation, List[Observation], None] = None,
    start_time: Union[astropy.time.Time, None] = None,
    time_span_s: Union[float, None] = None,
    delta_time_s: float = 86400.0,
) -> SpacecraftPositionAndVelocity:
    """Compute the position and velocity of the L2 point within some time span

    This function computes the XYZ position and velocity of the second Sun-Earth
    Lagrangean point (L2) over a time span specified either by a
    :class:`.Observation` object/list of objects, or by an explicit pair of values
    `start_time` (an ``astropy.time.Time`` object) and `time_span_s` (length in
    seconds). The position is specified in the standard Barycentric Ecliptic
    reference frame.

    The position of the L2 point is computed starting from the position of the
    Earth and moving away along the anti-Sun direction by a number of kilometers
    equal to `earth_l2_distance_km`.

    The result is an object of type :class:`.SpacecraftPositionAndVelocity`.

    If SpacecraftOrbit.solar_velocity_km_s > 0 also the Sun velocity in the rest
    frame of the CMB is added to the total velocity of the spacecraft.
    """
    assert observations or (start_time and time_span_s), (
        "You must either provide a Observation or start_time/time_span_s"
    )

    if observations:
        # The caller either provided an observation or a list of observations.
        # Let's compute the overall time span
        start_time, time_span_s = compute_start_and_span_for_obs(observations)

    # We are going to compute the position of the L2 point at N times. The value N
    # is chosen such that the spacing between two consecutive times is never longer
    # than ~1 day
    times = astropy.time.TimeDelta(
        np.linspace(
            start=0.0,
            stop=time_span_s,
            num=int(np.ceil(time_span_s / delta_time_s)) + 1,
        ),
        format="sec",
        scale=DEFAULT_TIME_SCALE,
    )

    pos, vel = compute_l2_pos_and_vel(
        time0=start_time + times, earth_l2_distance_km=orbit.earth_l2_distance_km
    )

    if orbit.radius1_km > 0 or orbit.radius2_km > 0:
        sum_lissajous_pos_and_vel(
            pos=pos,
            vel=vel,
            start_time_s=(start_time - orbit.start_time).to("s").value,
            end_time_s=time_span_s,
            radius1_km=orbit.radius1_km,
            radius2_km=orbit.radius2_km,
            ang_speed1_rad_s=orbit.ang_speed1_rad_s,
            ang_speed2_rad_s=orbit.ang_speed2_rad_s,
            phase_rad=orbit.phase_rad,
        )

    if orbit.solar_velocity_km_s > 0:
        vel += orbit.solar_velocity_ecl_xyz_km_s[np.newaxis, :]

    return SpacecraftPositionAndVelocity(
        orbit=orbit,
        start_time=start_time,
        time_span_s=time_span_s,
        positions_km=pos,
        velocities_km_s=vel,
    )
