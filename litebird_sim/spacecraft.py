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


def compute_l2_pos_and_vel(time0: astropy.time.Time, earth_l2_distance_km: float = EARTH_L2_DISTANCE_KM):
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
