# -*- encoding: utf-8 -*-
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
      ``(N)`` polarization angle (in radians) in galactic coordinates, returned
      if pol_angle_ecl is passed

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L578
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L537
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L357
    """

    pointings_gal = np.empty_like(pointings_ecl)

    vec = np.tensordot(
        ECL_TO_GAL_ROT_MATRIX,
        ang2vec(pointings_ecl[:, 0], pointings_ecl[:, 1]),
        axes=(1, 0),
    )
    north_pole = np.tensordot(ECL_TO_GAL_ROT_MATRIX, [0.0, 0.0, 1.0], axes=(1, 0))
    pointings_gal[:, 0], pointings_gal[:, 1] = vec2ang(vec[0], vec[1], vec[2])

    if type(pol_angle_ecl) is np.ndarray:
        pol_angle_gal = np.empty_like(pol_angle_ecl)

        sinalpha = north_pole[0] * vec[1] - north_pole[1] * vec[0]
        cosalpha = north_pole[2] - vec[2] * np.dot(north_pole, vec)
        pol_angle_gal = pol_angle_ecl + np.arctan2(sinalpha, cosalpha)

        return pointings_gal, pol_angle_gal

    else:
        return pointings_gal
