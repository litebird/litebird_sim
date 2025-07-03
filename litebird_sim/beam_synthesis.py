# -*- encoding: utf-8 -*-

from typing import Optional, Union, List

import numpy as np
from numpy import sqrt, exp, sin, cos, log
from scipy.special import iv as bessel_i

from .spherical_harmonics import SphericalHarmonics
from .observations import Observation

from .detectors import FreqChannelInfo

from .constants import ARCMIN_TO_RAD


def alm_index(lmax: int, ell: int, m: int) -> int:
    """
    Return the index of an a_ℓm coefficient in an array.

    Parameters
    ----------
    lmax : int
        ℓ_max of the expansion.
    ell : int
        The degree ℓ of the coefficient.
    m : int
        The order m of the coefficient.

    Returns
    -------
    int
        The index of the a_ℓm coefficient in the array.
    """
    return m * (2 * lmax + 1 - m) // 2 + ell


def allocate_alm(
    lmax: int, mmax: Optional[int] = None, nstokes: int = 3, dtype=np.complex128
) -> SphericalHarmonics:
    """
    Allocate an array to store spherical harmonics coefficients.

    Parameters
    ----------
    lmax : int
        The maximum degree ℓ_max.
    mmax : int, optional
        The maximum order m_max. If None, it is set equal to `lmax`.
    nstokes : int, default=3
        The number of Stokes parameters.
    dtype : data-type, default=np.complex128
        The data type of the array (should be a complex type).

    Returns
    -------
    SphericalHarmonics
        A SphericalHarmonics instance initialized with zeros.
    """
    nalm = SphericalHarmonics.num_of_alm_from_lmax(lmax, mmax)
    return SphericalHarmonics(
        values=np.zeros((nstokes, nalm), dtype=dtype),
        lmax=lmax,
        mmax=mmax if mmax else lmax,
    )


def gauss_beam_to_alm(
    lmax: int,
    mmax: int,
    fwhm_rad: float,
    ellipticity: Optional[float] = 1.0,
    psi_ell_rad: Optional[float] = 0.0,
    psi_pol_rad: Union[float, None] = 0.0,
    cross_polar_leakage: Optional[float] = 0.0,
) -> SphericalHarmonics:
    """
    Compute spherical harmonics coefficients a_ℓm representing a Gaussian beam.
    The code is taken from Planck LevelS, see
    https://github.com/zonca/planck-levelS/blob/master/Beam/gaussbeampol_main.f90

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree ℓ_max.
    mmax : int
        Maximum spherical harmonic order m_max.
    fwhm_rad : float
        Full width at half maximum (FWHM) of the beam in radians. Defined as fwhm = sqrt(fwhm_max*fwhm_min)
    ellipticity : float, optional, default=1.0
        Beam ellipticity. Defined as fwhm_max/fwhm_min Default is 1 (circular beam).
    psi_ell_rad : float, optional, default=0.0
        Orientation of the beam's major axis wrt the x-axis(radians).
    psi_pol_rad : float, optional, default=0.0
        Polarization angle of the beam wrt the x-axis. If None, only the intensity (I) is computed.
    cross_polar_leakage : float, optional, default=0.0
        Cross-polar leakage factor (pure number).

    Returns
    -------
    SphericalHarmonics
        The spherical harmonics coefficients representing the beam.
    """

    is_elliptical = False if ellipticity == 1.0 else True

    is_polarized = psi_pol_rad is not None

    if is_polarized:
        num_stokes = 3
    else:
        num_stokes = 1

    alm = allocate_alm(lmax, mmax, nstokes=num_stokes)

    if not is_elliptical:
        # Circular beam
        sigma_squared = fwhm_rad**2 / (8 * log(2))
        for ell in range(lmax + 1):
            alm.values[0, alm_index(lmax, ell, 0)] = sqrt(
                (2 * ell + 1) / (4 * np.pi)
            ) * exp(-0.5 * sigma_squared * ell * (ell + 1))

        if is_polarized:
            f1 = cos(2 * psi_pol_rad) - sin(2 * psi_pol_rad) * 1.0j
            for ell in range(2, lmax + 1):
                value = (
                    np.sqrt((2 * ell + 1) / (32 * np.pi))
                    * exp(-0.5 * sigma_squared * ell * (ell + 1))
                    * f1
                )
                alm.values[1, alm_index(lmax, ell, 2)] = value
                alm.values[2, alm_index(lmax, ell, 2)] = value * 1.0j
    else:
        # Elliptical beam
        e_squared = 1.0 - 1.0 / ellipticity**2
        sigma_x_squared = fwhm_rad**2 * ellipticity / (log(2) * 8)

        # I component
        for ell in range(lmax + 1):
            tmp = ell * (ell + 1) * sigma_x_squared
            for m in range(0, min(ell, mmax) + 1, 2):
                alm.values[0, alm_index(lmax, ell, m)] = (
                    np.sqrt((2 * ell + 1) / (4 * np.pi))
                    * exp(-0.5 * tmp * (1.0 - e_squared / 2))
                    * bessel_i(m // 2, 0.25 * tmp * e_squared)
                )

        if is_polarized:
            # Do G and C components

            rho = psi_pol_rad - psi_ell_rad
            f1 = cos(2 * rho) - sin(2 * rho) * 1j
            f2 = cos(2 * rho) + sin(2 * rho) * 1j

            for ell in range(2, lmax + 1):
                tmp = ell * (ell + 1) * sigma_x_squared
                tmp2 = 0.25 * tmp * e_squared

                # m = 0
                value = (
                    np.sqrt((2 * ell + 1) / (8 * np.pi))
                    * exp(-tmp * (1.0 - e_squared / 2) / 2)
                    * bessel_i(1, tmp2)
                )

                alm.values[1, alm_index(lmax, ell, 0)] = value * cos(2 * rho)
                alm.values[2, alm_index(lmax, ell, 0)] = value * sin(2 * rho)

                # m = 2, 4, …

                for m in range(2, min(ell, mmax) + 1, 2):
                    value = np.sqrt((2 * ell + 1) / (8 * (4 * np.pi))) * exp(
                        -0.5 * tmp * (1.0 - 0.5 * e_squared)
                    )
                    b1 = f1 * bessel_i((m - 2) // 2, tmp2)
                    b2 = f2 * bessel_i((m + 2) // 2, tmp2)
                    alm.values[1, alm_index(lmax, ell, m)] = value * (b1 + b2)
                    alm.values[2, alm_index(lmax, ell, m)] = value * (b1 - b2) * 1j

        # Rotate the multipoles through angle psi_ell about the z-axis, so
        # the beam is in the right orientation (only need this for even m).

        for m in range(0, mmax + 1, 2):
            f1 = cos(m * psi_ell_rad) - sin(m * psi_ell_rad) * 1j
            for n in range(num_stokes):
                for ell in range(m, lmax + 1):
                    alm.values[n, alm_index(lmax, ell, m)] *= f1

    # Adjust multipoles for cross-polar leakage
    alm.values[0, :] *= 1.0 + cross_polar_leakage

    if num_stokes > 1:
        for n in (1, 2):
            alm.values[n, :] *= 1.0 - cross_polar_leakage

    # Adjust the normalization
    if num_stokes > 1:
        for n in (1, 2):
            alm.values[n, :] *= -sqrt(2.0)

    return alm


def generate_gauss_beam_alms(
    observation: Observation,
    lmax: int,
    mmax: Optional[int] = None,
    channels: Union[FreqChannelInfo, List[FreqChannelInfo], None] = None,
    store_in_observation: Optional[bool] = False,
):
    """
    Generate Gaussian beam spherical harmonics coefficients for each detector in
    the given Observation

    This function computes the blms for a 2D Gaussian beam, accounting for
    detector-specific parameters such as beam width (FWHM), ellipticity,
    and polarization orientation. Optionally, the results can be stored
    directly in the `Observation` object.

    Parameters
    ----------
    observation : Observation
        Observation object containing detector parameters.
    lmax : int
        Maximum spherical harmonic degree ℓ_max.
    mmax : int, optional
        Maximum spherical harmonic order m_max. If None, it defaults to `lmax`.
    channels : FreqChannelInfo or list of FreqChannelInfo, optional
        Frequency channels to use in the simulation. If None, it uses the detectors
        from the current observations.
    store_in_observation : bool, optional
        If True, the computed blms will be stored in the `blms` attribute of
        the observation object.

    Returns
    -------
    dict
        Dictionary mapping detector names to SphericalHarmonics objects.
    """
    mmax_val = mmax or lmax  # Use mmax if provided, else default to lmax

    blms = {}

    if channels is None:
        # Use detectors from observations
        for detector_idx, det_name in enumerate(observation.name):
            fwhm_rad = observation.fwhm_arcmin[detector_idx] * ARCMIN_TO_RAD

            blms[det_name] = gauss_beam_to_alm(
                lmax=lmax,
                mmax=mmax_val,
                fwhm_rad=fwhm_rad,
                ellipticity=observation.ellipticity[detector_idx],
                psi_ell_rad=observation.psi_rad[detector_idx],
                psi_pol_rad=observation.pol_angle_rad[detector_idx],
                cross_polar_leakage=0,
            )
    else:
        # Use explicitly provided frequency channels
        channel_list = [channels] if isinstance(channels, FreqChannelInfo) else channels

        for channel in channel_list:
            fwhm_rad = channel.fwhm_arcmin * ARCMIN_TO_RAD
            blms[channel.channel] = gauss_beam_to_alm(
                lmax=lmax,
                mmax=mmax_val,
                fwhm_rad=fwhm_rad,
                ellipticity=channel.ellipticity,
                psi_ell_rad=0,
                psi_pol_rad=0,
                cross_polar_leakage=0,
            )

    if store_in_observation:
        observation.blms = blms

    return blms
