# -*- encoding: utf-8 -*-
from numba import njit
import numpy as np

from astropy.coordinates import BarycentricMeanEcliptic
from astropy.coordinates import SkyCoord

from enum import Enum

"""The coordinate system used by the framework"""
DEFAULT_COORDINATE_SYSTEM = BarycentricMeanEcliptic()

"""The time scale used by the framework"""
DEFAULT_TIME_SCALE = "tdb"

ECL_TO_GAL_ROT_MATRIX = (
    SkyCoord(
        x=[1.0, 0.0, 0.0],
        y=[0.0, 1.0, 0.0],
        z=[0.0, 0.0, 1.0],
        frame=DEFAULT_COORDINATE_SYSTEM.name,
        representation_type="cartesian",
    )
    .transform_to("galactic")
    .data.to_cartesian()
    .get_xyz()
    .value
)

NORTH_POLE_VEC = np.tensordot(ECL_TO_GAL_ROT_MATRIX, [0.0, 0.0, 1.0], axes=(1, 0))

"""
The coordinate system used to express pointing angles.
"""
CoordinateSystem = Enum("CoordinateSystem", ["Ecliptic", "Galactic"])


def ang2vec(theta, phi):
    """Transform a direction theta,phi to a unit vector.

    Parameters
    ----------
    theta : float, scalar or array-like
      The angle theta (scalar or shape (N,))
    phi : float, scalar or array-like
      The angle phi (scalar or shape (N,)).

    Returns
    -------
    vec : array
      The vector(s) corresponding to given angles, shape is (3,) or (3, N).

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L657
    """
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    vec = np.empty((3, ct.size), np.float64)
    vec[0, :] = st * cp
    vec[1, :] = st * sp
    vec[2, :] = ct
    return vec.squeeze()


def vec2ang(vx, vy, vz):
    """Transform a vector to angle given by theta,phi.

    Parameters
    ----------
    vx : float, scalar or array-like
      The x component of the vector (scalar or shape (N,))
    vy : float, scalar or array-like, optional
      The y component of the vector (scalar or shape (N,))
    vz : float, scalar or array-like, optional
      The z component of the vector (scalar or shape (N,))

    Returns
    -------
    angles : float, array
      The angles in radiants in an array of shape (2, N)

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L610
    """

    ang = np.empty((2, vx.size))
    ang[0, :] = np.arctan2(np.sqrt(vx**2 + vy**2), vz)
    ang[1, :] = np.arctan2(vy, vx)
    return ang.squeeze()


@njit
def _ang2galvec_one_sample(theta, phi):
    """Transform a direction (theta, phi) in Ecliptic coordinates to
    a unit vector in Galactic coordinates.

    Parameters
    ----------
    theta : float, scalar
      The angle θ (colatitude) in Ecliptic coordinates
    phi : float, scalar
      The angle φ (longitude) in Ecliptic coordinates

    Returns
    -------
    vx, vy, vz : float
      A tuple of three floats

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L657
    """

    rotmatr = ECL_TO_GAL_ROT_MATRIX
    st = np.sin(theta)
    vx, vy, vz = st * np.cos(phi), st * np.sin(phi), np.cos(theta)

    return (
        rotmatr[0, 0] * vx + rotmatr[0, 1] * vy + rotmatr[0, 2] * vz,
        rotmatr[1, 0] * vx + rotmatr[1, 1] * vy + rotmatr[1, 2] * vz,
        rotmatr[2, 0] * vx + rotmatr[2, 1] * vy + rotmatr[2, 2] * vz,
    )


@njit
def _vec2ang_for_one_sample(vx, vy, vz):
    """Transform a vector to angle given by theta,phi.

    Parameters
    ----------
    vx : float, scalar
      The x component of the vector (scalar)
    vy : float, scalar
      The y component of the vector (scalar))
    vz : float, scalar
      The z component of the vector (scalar)

    Returns
    -------
    theta, phi : float
      A tuple containing the value of the colatitude and of the longitude

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L610
    """

    return (np.arctan2(np.sqrt(vx**2 + vy**2), vz), np.arctan2(vy, vx))


@njit
def _rotate_coordinates_e2g_for_one_sample(pointings_ecl):
    """
    Parameters
    ----------
    pointings_ecl : array
      ``(2)`` array containing the colatitude and the longitude in ecliptic
      coordinates

    Returns
    -------
    pointings_gal : 2-tuple of floats
      The value of θ and φ (colatitude and longitude) in Galactic coordinates

    """

    vec = (_ang2galvec_one_sample(pointings_ecl[0], pointings_ecl[1]),)
    return _vec2ang_for_one_sample(vec[0], vec[1], vec[2])


@njit
def _rotate_coordinates_and_pol_e2g_for_one_sample(pointings_ecl, pol_angle_ecl):
    """Rotate the angles theta,phi and psi from ecliptic to galactic coordinates

    Parameters
    ----------
    pointings_ecl : array
      ``(2)`` array containing the colatitude and the longitude in ecliptic
      coordinates

    pol_angle_ecl : array
      polarization angle (in radians) in ecliptic coordinates

    Returns
    -------
    pointings_gal : 2-tuple of floats
      The value of θ and φ (colatitude and longitude) in Galactic coordinates

    pol_angle_gal: float
      polarization angle (in radians) in galactic coordinates
    """

    vec = _ang2galvec_one_sample(pointings_ecl[0], pointings_ecl[1])
    pointings_gal = _vec2ang_for_one_sample(vec[0], vec[1], vec[2])

    sinalpha = NORTH_POLE_VEC[0] * vec[1] - NORTH_POLE_VEC[1] * vec[0]
    cosalpha = NORTH_POLE_VEC[2] - vec[2] * (
        NORTH_POLE_VEC[0] * vec[0]
        + NORTH_POLE_VEC[1] * vec[1]
        + NORTH_POLE_VEC[2] * vec[2]
    )
    pol_angle_gal = pol_angle_ecl + np.arctan2(sinalpha, cosalpha)

    return pointings_gal, pol_angle_gal


@njit
def _rotate_coordinates_e2g_for_all(pointings_ecl, pointings_gal):
    """
    Parameters
    ----------
    pointings_ecl : array
      ``(N × 2)`` array containing the colatitude and the longitude in ecliptic
      coordinates

    pointings_gal : array
      ``(N × 2)`` array containing the colatitude and the longitude in galactic
      coordinates
    """

    for i in range(len(pointings_ecl[:, 0])):
        cur_theta, cur_phi = _rotate_coordinates_e2g_for_one_sample(pointings_ecl[i, :])
        pointings_gal[i, 0] = cur_theta
        pointings_gal[i, 1] = cur_phi


@njit
def _rotate_coordinates_and_pol_e2g_for_all(
    pointings_ecl, pol_angle_ecl, pointings_gal, pol_angle_gal
):
    """
    Parameters
    ----------
    pointings_ecl : array
      ``(N × 2)`` array containing the colatitude and the longitude in ecliptic
      coordinates

    pol_angle_ecl : array
      ``(N)`` polarization angle (in radians) in ecliptic coordinates

    pointings_gal : array
      ``(N × 2)`` array containing the colatitude and the longitude in galactic
      coordinates

    pol_angle_gal: array
      ``(N)`` polarization angle (in radians) in gal actic coordinates
    """

    for i in range(len(pointings_ecl[:, 0])):
        (
            cur_direction,
            cur_pol_angle,
        ) = _rotate_coordinates_and_pol_e2g_for_one_sample(
            pointings_ecl[i, :], pol_angle_ecl[i]
        )

        pointings_gal[i, 0] = cur_direction[0]
        pointings_gal[i, 1] = cur_direction[1]
        pol_angle_gal[i] = cur_pol_angle


def rotate_coordinates_e2g(pointings_ecl, pol_angle_ecl=None):
    """Rotate the angles theta,phi and psi from ecliptic to galactic coordinates

    Parameters
    ----------
    pointings_ecl : array
      ``(N × 2)`` array containing the colatitude and the longitude in ecliptic
      coordinates

    pol_angle_ecl : array
      ``(N)`` polarization angle (in radians) in ecliptic coordinates, otional

    Returns
    -------
    pointings_gal : array
      ``(N × 2)`` array containing the colatitude and the longitude in galactic
      coordinates

    pol_angle_gal: array
      ``(N)`` polarization angle (in radians) in gal actic coordinates, returned
      if pol_angle_ecl is passed

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L578
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L537
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L357
    """

    pointings_gal = np.empty_like(pointings_ecl)

    if type(pol_angle_ecl) is np.ndarray:
        pol_angle_gal = np.empty_like(pol_angle_ecl)
        _rotate_coordinates_and_pol_e2g_for_all(
            pointings_ecl, pol_angle_ecl, pointings_gal, pol_angle_gal
        )
        return pointings_gal, pol_angle_gal
    else:
        _rotate_coordinates_e2g_for_all(pointings_ecl, pointings_gal)
        return pointings_gal
