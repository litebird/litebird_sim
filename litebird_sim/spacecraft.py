# -*- encoding: utf-8 -*-

from dataclasses import dataclass

import astropy
from astropy.coordinates import (
    ICRS,
    get_body_barycentric_posvel,
)
from numba import njit
import numpy as np

from .coordinates import DEFAULT_COORDINATE_SYSTEM

EARTH_L2_DISTANCE_KM = 1_496_509.30522


def cycles_per_year_to_rad_per_s(x: float) -> float:
    """Convert an angular speed from cycles/yr to rad/s"""
    return x * 2 * np.pi / (365 * 86400)


def get_ecliptic_vec(vec):
    "Convert a coordinate in a XYZ vector expressed in the Ecliptic rest frame"
    return ICRS(vec).transform_to(DEFAULT_COORDINATE_SYSTEM).cartesian.get_xyz()


def compute_l2_pos_and_vel(
    time0: astropy.time.Time, earth_l2_distance_km: float = EARTH_L2_DISTANCE_KM
):
    """
    Compute the position and velocity of the L2 Sun-Earth point at a given time, specified as a AstroPy time.

    The L2 point is not calculated using Lagrange's equations; instead, its distance from the Earth must be
    provided as the parameter `earth_l2_distance_km`. The default value is a reasonable estimate. The L2 point
    is assumed to lie along the line that connects the Solar System Barycenter with the Earth's gravitational center.

    The return value is a 2-tuple containing two NumPy arrays:

    1. A 3D array containing the XYZ components of the vector specifying the position of the L2 point, in km
    2. A 3D array containing the XYZ components of the velocity vector of the L2 point, in km/s

    The two vectors are always roughly perpendicular, but they are not exactly due to the irregular motion of the
    Earth (caused by gravitational interactions with other solar system bodies, like the Moon and Jupiter).
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

    return l2_pos.xyz.to("km").value, l2_vel.xyz.to("km/s").value


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

    The position and velocity are calculated in a reference frame centered on the L2 point, whose axes
    are aligned with the Solar System Barycenter. This means that the position and velocity of the spacecraft
    with respect to the Solar System Barycenter itself can be calculated by summing the result of this function
    with the result of a call to :func:`.compute_l2_pos_and_vel`.
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


@dataclass
class SpacecraftOrbit:
    """A dataclass describing the orbit of the LiteBIRD spacecraft

    This structure has the following fields:

    - `earth_l2_distance_km`: distance between the Earth's barycenter and the L2 point, in km
    - `radius1_km`: first radius describing the Lissajous orbit followed by the spacecraft, in km
    - `radius2_km`: second radius describing the Lissajous orbit followed by the spacecraft, in km
    - `ang_speed1_rad_s`: first angular speed of the Lissajous orbit, in rad/s
    - `ang_speed2_rad_s`: second angular speed of the Lissajous orbit, in rad/s
    - `phase_rad`: phase difference between the two periodic motions in the Lissajous orbit, in radians.

    The default values are the nominal numbers of the orbit followed by WMAP, described in Cavaluzzi,
    Fink & Coyle (2008).
    """

    earth_l2_distance_km: float = EARTH_L2_DISTANCE_KM
    radius1_km: float = 244_450.0
    radius2_km: float = 137_388.0
    ang_speed1_rad_s: float = cycles_per_year_to_rad_per_s(2.021_04)
    ang_speed2_rad_s: float = cycles_per_year_to_rad_per_s(1.985_07)
    phase_rad: float = np.deg2rad(-47.944)
