# -*- encoding: utf-8 -*-

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    SkyCoord,
    BarycentricMeanEcliptic,
    Galactic,
    CartesianRepresentation,
)


def astropy_ecl_to_gal(
    theta: np.ndarray, phi: np.ndarray, psi: np.ndarray
) -> np.ndarray:
    """Convert coordinates from Ecliptic to Galactic using AstroPy"""

    coords_ecl = SkyCoord(
        lon=phi * u.rad,
        lat=(np.pi / 2 - theta) * u.rad,
        frame=BarycentricMeanEcliptic(),
    )
    coords_gal = coords_ecl.transform_to(Galactic())

    gal_north = SkyCoord(
        l=0 * u.deg,
        b=90 * u.deg,
        frame=Galactic(),
    )
    delta_psi = coords_ecl.position_angle(gal_north)

    return np.stack(
        [
            np.pi / 2 - coords_gal.b.rad,
            coords_gal.l.rad,
            psi + delta_psi.rad,
        ],
        axis=-1,
    )


def astropy_ecl_to_gal_matrix():
    """
    Computes the 3x3 rotation matrix from Barycentric Mean Ecliptic (J2000)
    to Galactic coordinates using AstroPy's high-precision frames.
    """

    # Create the identity matrix representing the Ecliptic axes (X, Y, Z)
    identity_axes = CartesianRepresentation(np.eye(3))

    # Wrap them in the Ecliptic frame
    c_ecl = SkyCoord(identity_axes, frame=BarycentricMeanEcliptic())

    # Transform these axes to the Galactic frame
    c_gal = c_ecl.transform_to(Galactic())

    # The resulting cartesian coordinates form the rotation matrix.
    # We transpose to match Healpy's 'passive' rotation expectation.
    return c_gal.cartesian.xyz.value.T
