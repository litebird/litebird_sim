# -*- encoding: utf-8 -*-
from enum import Enum
from typing import Tuple

import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic
from astropy.coordinates import SkyCoord
from numba import njit

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

_COORD_SYS_TO_HEALPIX = {
    CoordinateSystem.Ecliptic: "ECLIPTIC",
    CoordinateSystem.Galactic: "GALACTIC",
}


def coord_sys_to_healpix_string(coordsys: CoordinateSystem) -> str:
    """Convert the value of a :class:`.CoordinateSystem` instance into a string

    The string is suitable to be saved in a FITS file containing a Healpix map"""
    return _COORD_SYS_TO_HEALPIX[coordsys]


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
    """Transform a vector (or many vectors) to angle given by theta,phi.

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
      The angles in radians in an array of shape (2, N)

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L610
    """

    ang = np.empty((2, vx.size))
    ang[0, :] = np.arctan2(np.sqrt(vx**2 + vy**2), vz)
    ang[1, :] = np.arctan2(vy, vx)
    return ang.squeeze()


@njit
def _ang2galvec_one_sample(
    theta_rad: float, phi_rad: float
) -> Tuple[float, float, float]:
    """Transform a direction (theta, phi) in Ecliptic coordinates to
    a unit vector in Galactic coordinates.

    Parameters
    ----------
    theta_rad : float, scalar
      The angle θ (colatitude) in Ecliptic coordinates
    phi_rad : float, scalar
      The angle φ (longitude) in Ecliptic coordinates

    Returns
    -------
    vx, vy, vz : float
      A tuple of three floats representing the (x, y, z) components
      of the vector

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L657
    """

    rotmatr = ECL_TO_GAL_ROT_MATRIX
    st = np.sin(theta_rad)
    vx, vy, vz = st * np.cos(phi_rad), st * np.sin(phi_rad), np.cos(theta_rad)

    return (
        rotmatr[0, 0] * vx + rotmatr[0, 1] * vy + rotmatr[0, 2] * vz,
        rotmatr[1, 0] * vx + rotmatr[1, 1] * vy + rotmatr[1, 2] * vz,
        rotmatr[2, 0] * vx + rotmatr[2, 1] * vy + rotmatr[2, 2] * vz,
    )


@njit
def _vec2ang_for_one_sample(vx: float, vy: float, vz: float) -> Tuple[float, float]:
    """Transform a vector to angle given by (θ,φ).

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
      A tuple containing the value of the colatitude and of the longitude,
      in radians

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L610
    """

    return np.arctan2(np.sqrt(vx**2 + vy**2), vz), np.arctan2(vy, vx)


@njit
def _rotate_coordinates_and_pol_e2g_for_one_sample(
    theta_ecl_rad: float, phi_ecl_rad: float, psi_ecl_rad: float
) -> Tuple[float, float, float]:
    """Rotate the angles theta,phi and psi from ecliptic to galactic coordinates

    Parameters
    ----------
    theta_ecl_rad : float
      Colatitude in Ecliptic coordinates (in radians)

    phi_ecl_rad : float
      Longitude in Ecliptic coordinates (in radians)

    psi_ecl_rad : float
      Orientation in Ecliptic coordinates (in radians)

    Returns
    -------
    theta_gal_rad, phi_gal_rad, psi_gal_rad : 3-tuple of floats
      The value of θ, φ, ψ (colatitude, longitude, orientation) in Galactic
      coordinates, in radians
    """

    # Rotate the direction θ,φ
    x, y, z = _ang2galvec_one_sample(theta_rad=theta_ecl_rad, phi_rad=phi_ecl_rad)
    theta_gal_rad, phi_gal_rad = _vec2ang_for_one_sample(x, y, z)

    # Rotate the orientation ψ
    sinalpha = NORTH_POLE_VEC[0] * y - NORTH_POLE_VEC[1] * x
    cosalpha = NORTH_POLE_VEC[2] - z * (
        NORTH_POLE_VEC[0] * x + NORTH_POLE_VEC[1] * y + NORTH_POLE_VEC[2] * z
    )
    psi_gal_rad = psi_ecl_rad + np.arctan2(sinalpha, cosalpha)

    return theta_gal_rad, phi_gal_rad, psi_gal_rad


@njit
def _rotate_coordinates_and_pol_e2g_for_all(
    input_pointings_ecl_rad: np.ndarray,
    output_pointings_gal_rad: np.ndarray,
) -> None:
    """
    Parameters
    ----------
    input_pointings_ecl_rad : array
      ``(N × 3)`` array containing the colatitude, longitude, orientation in Ecliptic
      coordinates (in radians)

    output_pointings_gal_rad : array
      ``(N × 3)`` output array that will contain the colatitude, longitude, orientation
      in Galactic coordinates (in radians)
    """

    n_samples = input_pointings_ecl_rad.shape[0]
    for i in range(n_samples):
        cur_theta_ecl_rad, cur_phi_ecl_rad, cur_psi_ecl_rad = input_pointings_ecl_rad[
            i, :
        ]

        (
            cur_theta_gal_rad,
            cur_phi_gal_rad,
            cur_psi_gal_rad,
        ) = _rotate_coordinates_and_pol_e2g_for_one_sample(
            cur_theta_ecl_rad,
            cur_phi_ecl_rad,
            cur_psi_ecl_rad,
        )

        output_pointings_gal_rad[i, 0] = cur_theta_gal_rad
        output_pointings_gal_rad[i, 1] = cur_phi_gal_rad
        output_pointings_gal_rad[i, 2] = cur_psi_gal_rad


def rotate_coordinates_e2g(pointings_ecl_rad: np.ndarray) -> np.ndarray:
    """Rotate the angles theta,phi and psi from ecliptic to galactic coordinates

    Parameters
    ----------
    pointings_ecl_rad : array
      ``(N × 3)`` array containing the colatitude and the longitude in ecliptic
      coordinates

    Returns
    -------
    pointings_gal_rad : array
      ``(N × 3)`` array containing the colatitude and the longitude in galactic
      coordinates

    See Also
    --------
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L578
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L537
    https://github.com/healpy/healpy/blob/main/healpy/rotator.py#L357
    """

    pointings_gal_rad = np.empty_like(pointings_ecl_rad)

    _rotate_coordinates_and_pol_e2g_for_all(
        input_pointings_ecl_rad=pointings_ecl_rad,
        output_pointings_gal_rad=pointings_gal_rad,
    )
    return pointings_gal_rad
