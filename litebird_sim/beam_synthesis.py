# -*- encoding: utf-8 -*-

from typing import Optional

import numpy as np
import numpy.typing as npt
from numpy import sqrt, exp, sin, cos, log
from scipy.special import iv as bessel_i


def alm_size(lmax: int, mmax: Optional[int] = None) -> int:
    """Given a value for ℓ_max and m_max, return the size of the array a_ℓm

    If `mmax` is not provided, it is set equal to `lmax`
    """
    if mmax is None:
        mmax = lmax
    else:
        assert mmax >= 0
        assert mmax <= lmax

    return mmax * (2 * lmax + 1 - mmax) // 2 + lmax + 1


def alm_index(lmax: int, ell: int, m: int) -> int:
    "Return the index of an a_ℓm coefficient in an array"

    return m * (2 * lmax + 1 - m) // 2 + ell


def allocate_alm(
    lmax: int, mmax: Optional[int] = None, nstokes: int = 3, dtype=np.complex128
) -> npt.NDArray:
    """
    Allocate a NumPy array that will hold a set of a_ℓm coefficients

    :param lmax: The maximum value for ℓ
    :param mmax: The maximum value for m. If ``None``, it is set to `lmax`
    :param nstokes: The number of Stokes parameters to consider
    :param dtype: The data type. It should be a complex type.
    :return: A newly-allocate NumPy array
    """
    nalm = alm_size(lmax, mmax)
    return np.zeros((nstokes, nalm), dtype=dtype)


def gauss_beam_to_alm(
    lmax: int,
    mmax: int,
    fwhm_min_rad: float,
    fwhm_max_rad: Optional[float],
    psi_ell_rad: float,
    psi_pol_rad: float,
    cross_polar_leakage: float,
    alm: Optional[npt.NDArray] = None,
) -> npt.NDArray:
    """
    Return an array of spherical harmonics a_ℓm that represents a Gaussian beam

    :param lmax: The maximum value for ℓ
    :param mmax: The maximum range for m; usually this is equal to ``lmax``
    :param fwhm_min_rad: The FHWM along the minor axis
    :param fwhm_max_rad: The FWHM along the major axis. Set this to ``None`` to
      assume a circular beam, i.e., ``fwhm_max_rad == fwhm_min_rad``
    :param psi_ell_rad: The inclination of the major axis of the ellipse with
      respect to the x-axis
    :param psi_pol_rad: The polarization of the beam with respect to the x-axis
    :param cross_polar_leakage: The cross-polar leakage (pure number)
    :param alm: If provided, it is the buffer that will contain the a_ℓm coefficients.
      It should be a NumPy 1D array of complex values allocated using
      :ref:`.allocate_alm`.
    :return:
      The array containing the a_ℓm values. If the `alm` parameter was not ``None``,
      this is the array that is returned.
    """

    if alm is None:
        alm = allocate_alm(lmax, mmax)

    if fwhm_max_rad is None:
        fwhm_rad = fwhm_min_rad
        ellipticity = 1.0
        is_elliptical = False
    else:
        fwhm_rad = np.sqrt(fwhm_max_rad * fwhm_min_rad)
        ellipticity = fwhm_max_rad / fwhm_min_rad
        is_elliptical = ellipticity != 1.0

    assert (
        len(alm.shape) == 2
    ), "The a_ℓm array should be a 2D array with size (N_STOKES, SIZE)"
    assert (
        alm_size(lmax, mmax) == alm.shape[1]
    ), f"Wrong size of the a_ℓm array ({alm.shape})"

    num_stokes = alm.shape[0]
    is_polarized = not np.isnan(psi_pol_rad)

    if not is_elliptical:
        # Circular beam
        sigma_squared = fwhm_rad**2 / (8 * log(2))
        for ell in range(lmax + 1):
            alm[0, alm_index(lmax, ell, 0)] = sqrt((2 * ell + 1) / (4 * np.pi)) * exp(
                -0.5 * sigma_squared * ell**2
            )

        if num_stokes > 1 and is_polarized:
            f1 = cos(2 * psi_pol_rad) - sin(2 * psi_pol_rad) * 1.0j
            for ell in range(2, lmax + 1):
                value = (
                    np.sqrt((2 * ell + 1) / (32 * np.pi))
                    * exp(-sigma_squared * ell**2 / 2)
                    * f1
                )
                alm[1, alm_index(lmax, ell, 2)] = value
                alm[2, alm_index(lmax, ell, 2)] = value * 1.0j
    else:
        # Elliptical beam
        e_squared = 1.0 - 1.0 / ellipticity**2
        sigma_x_squared = fwhm_rad**2 * ellipticity / (log(2) * 8)

        # I component
        for ell in range(lmax + 1):
            tmp = ell**2 * sigma_x_squared
            for m in range(0, min(ell, mmax) + 1, 2):
                alm[0, alm_index(lmax, ell, m)] = (
                    np.sqrt((2 * ell + 1) / (4 * np.pi))
                    * exp(-0.5 * tmp * (1.0 - e_squared / 2))
                    * bessel_i(m // 2, 0.25 * tmp * e_squared)
                )

        if num_stokes > 1 and is_polarized:
            # Do G and C components

            rho = psi_pol_rad - psi_ell_rad
            f1 = cos(2 * rho) - sin(2 * rho) * 1j
            f2 = cos(2 * rho) + sin(2 * rho) * 1j

            for ell in range(2, lmax + 1):
                tmp = ell**2 * sigma_x_squared
                tmp2 = 0.25 * tmp * e_squared

                # m = 0
                value = (
                    np.sqrt((2 * ell + 1) / (8 * np.pi))
                    * exp(-tmp * (1.0 - e_squared / 2) / 2)
                    * bessel_i(1, tmp2)
                )

                alm[1, alm_index(lmax, ell, 0)] = value * cos(2 * rho)
                alm[2, alm_index(lmax, ell, 0)] = value * sin(2 * rho)

                # m = 2, 4, …

                for m in range(2, min(ell, mmax) + 1, 2):
                    value = np.sqrt((2 * ell + 1) / (8 * (4 * np.pi))) * exp(
                        -0.5 * tmp * (1.0 - 0.5 * e_squared)
                    )
                    b1 = f1 * bessel_i((m - 2) // 2, tmp2)
                    b2 = f2 * bessel_i((m + 2) // 2, tmp2)
                    alm[1, alm_index(lmax, ell, m)] = value * (b1 + b2)
                    alm[2, alm_index(lmax, ell, m)] = value * (b1 - b2) * 1j

        # Rotate the multipoles through angle psi_ell about the z-axis, so
        # the beam is in the right orientation (only need this for even m).

        for m in range(0, mmax + 1, 2):
            f1 = cos(m * psi_ell_rad) - sin(m * psi_ell_rad) * 1j
            for n in range(num_stokes):
                for ell in range(m, lmax + 1):
                    alm[n, alm_index(lmax, ell, m)] *= f1

    # Adjust multipoles for cross-polar leakage
    alm[0, :] *= (1.0 + cross_polar_leakage) / 2

    if num_stokes > 1:
        for n in (1, 2):
            alm[n, :] *= (1.0 - cross_polar_leakage) / 2

    # Adjust the normalization
    if num_stokes > 1:
        for n in (1, 2):
            alm[n, :] *= -sqrt(2.0)

    return alm
