import numpy as np
from scipy.special import iv as bessel_i

from .maps_and_harmonics import SphericalHarmonics
from .observations import Observation
from .detectors import FreqChannelInfo
from .constants import ARCMIN_TO_RAD


def gauss_beam_to_alm(
    lmax: int,
    mmax: int,
    fwhm_rad: float,
    ellipticity: float = 1.0,
    psi_ell_rad: float = 0.0,
    psi_pol_rad: float = 0.0,
    cross_polar_leakage: float = 0.0,
) -> SphericalHarmonics:
    """
    Compute spherical harmonics coefficients a_ℓm representing a Gaussian beam.

    The implementation is vectorized for performance and based on the physics
    described in Planck LevelS:
    https://github.com/zonca/planck-levelS/blob/master/Beam/gaussbeampol_main.f90

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree ℓ_max.
    mmax : int
        Maximum spherical harmonic order m_max.
    fwhm_rad : float
        Full width at half maximum (FWHM) of the beam in radians.
        Defined as fwhm = sqrt(fwhm_max*fwhm_min).
    ellipticity : float, optional, default=1.0
        Beam ellipticity. Defined as fwhm_max/fwhm_min. Default is 1 (circular beam).
    psi_ell_rad : float, optional, default=0.0
        Orientation of the beam's major axis wrt the x-axis (radians).
    psi_pol_rad : float, optional, default=0.0
        Polarization angle of the beam wrt the x-axis.
        If None, only the intensity (I) is computed (nstokes=1).
    cross_polar_leakage : float, optional, default=0.0
        Cross-polar leakage factor (pure number).

    Returns
    -------
    SphericalHarmonics
        The spherical harmonics coefficients representing the beam.
    """
    is_elliptical = ellipticity != 1.0
    is_polarized = psi_pol_rad is not None
    num_stokes = 3 if is_polarized else 1

    # Use the factory method from SphericalHarmonics
    alm = SphericalHarmonics.zeros(lmax=lmax, mmax=mmax, nstokes=num_stokes)

    # Common parameters
    sigma_squared = fwhm_rad**2 / (8 * np.log(2))

    # ---------------------------------------------------------
    # Circular Beam
    # ---------------------------------------------------------
    if not is_elliptical:
        # Intensity (T) component: only m=0 terms are non-zero for circular Gaussian
        ell = np.arange(lmax + 1)
        # Index for m=0 is just ell in standard healpix ordering (first block)
        idx_m0 = ell

        beam_profile = np.sqrt((2 * ell + 1) / (4 * np.pi)) * np.exp(
            -0.5 * sigma_squared * ell * (ell + 1)
        )

        alm.values[0, idx_m0] = beam_profile

        if is_polarized:
            # Pol components: only m=2 terms are non-zero
            # We need ell >= 2
            ell_p = np.arange(2, lmax + 1)
            f1 = np.cos(2 * psi_pol_rad) - np.sin(2 * psi_pol_rad) * 1.0j

            beam_pol = (
                np.sqrt((2 * ell_p + 1) / (32 * np.pi))
                * np.exp(-0.5 * sigma_squared * ell_p * (ell_p + 1))
                * f1
            )

            # Get indices for m=2
            idx_m2 = SphericalHarmonics.get_index(lmax, ell_p, 2)

            alm.values[1, idx_m2] = beam_pol
            alm.values[2, idx_m2] = beam_pol * 1.0j

    # ---------------------------------------------------------
    # Elliptical Beam
    # ---------------------------------------------------------
    else:
        e_squared = 1.0 - 1.0 / ellipticity**2
        sigma_x_squared = fwhm_rad**2 * ellipticity / (np.log(2) * 8)

        # Precompute polarization rotation factors if needed
        if is_polarized:
            rho = psi_pol_rad - psi_ell_rad
            cos_2rho = np.cos(2 * rho)
            sin_2rho = np.sin(2 * rho)
            f1_pol = cos_2rho - sin_2rho * 1j
            f2_pol = cos_2rho + sin_2rho * 1j

        # Loop over m (step 2). For each m, we vectorize over ell.
        for m in range(0, mmax + 1, 2):
            ell = np.arange(m, lmax + 1)

            # Compute common term depending on ell
            tmp = ell * (ell + 1) * sigma_x_squared

            # Slice for the current m block
            idx_start = SphericalHarmonics.get_index(lmax, m, m)
            s_slice = slice(idx_start, idx_start + len(ell))

            # --- Intensity (I) ---
            val_I = (
                np.sqrt((2 * ell + 1) / (4 * np.pi))
                * np.exp(-0.5 * tmp * (1.0 - e_squared / 2))
                * bessel_i(m // 2, 0.25 * tmp * e_squared)
            )

            alm.values[0, s_slice] = val_I

            # --- Polarization (Q, U) ---
            if is_polarized:
                # Mask for ell >= 2 (only relevant if m < 2, i.e., m=0 case)
                if m < 2:
                    mask = ell >= 2
                    ell_sub = ell[mask]
                    tmp_sub = tmp[mask]
                else:
                    mask = slice(None)
                    ell_sub = ell
                    tmp_sub = tmp

                tmp2 = 0.25 * tmp_sub * e_squared

                if m == 0:
                    # Special case m=0, uses bessel_i(1, ...)
                    val_pol = (
                        np.sqrt((2 * ell_sub + 1) / (8 * np.pi))
                        * np.exp(-tmp_sub * (1.0 - e_squared / 2) / 2)
                        * bessel_i(1, tmp2)
                    )

                    # Map mask back to the alm array slice
                    alm_view_1 = alm.values[1, s_slice]
                    alm_view_2 = alm.values[2, s_slice]

                    alm_view_1[mask] = val_pol * cos_2rho
                    alm_view_2[mask] = val_pol * sin_2rho

                else:
                    # General case m >= 2
                    prefactor = np.sqrt((2 * ell_sub + 1) / (32 * np.pi)) * np.exp(
                        -0.5 * tmp_sub * (1.0 - 0.5 * e_squared)
                    )

                    b1 = f1_pol * bessel_i((m - 2) // 2, tmp2)
                    b2 = f2_pol * bessel_i((m + 2) // 2, tmp2)

                    alm.values[1, s_slice] = prefactor * (b1 + b2)
                    alm.values[2, s_slice] = prefactor * (b1 - b2) * 1.0j

        # Rotate the multipoles through angle psi_ell about the z-axis
        if psi_ell_rad != 0.0:
            for m in range(0, mmax + 1, 2):
                if m == 0:
                    continue
                f_rot = np.cos(m * psi_ell_rad) - np.sin(m * psi_ell_rad) * 1j

                idx_start = SphericalHarmonics.get_index(lmax, m, m)
                idx_end = SphericalHarmonics.get_index(lmax, lmax, m) + 1
                alm.values[:, idx_start:idx_end] *= f_rot

    # ---------------------------------------------------------
    # Adjustments
    # ---------------------------------------------------------

    # Adjust for cross-polar leakage
    if cross_polar_leakage != 0.0:
        alm.values[0, :] *= 1.0 + cross_polar_leakage
        if num_stokes > 1:
            alm.values[1:, :] *= 1.0 - cross_polar_leakage

    # Adjust normalization for Pol
    if num_stokes > 1:
        alm.values[1:, :] *= -np.sqrt(2.0)

    return alm


def generate_gauss_beam_alms(
    observation: Observation,
    lmax: int,
    mmax: int | None = None,
    channels: FreqChannelInfo | list[FreqChannelInfo] | None = None,
    store_in_observation: bool = False,
) -> dict[str, SphericalHarmonics]:
    """
    Generate Gaussian beam spherical harmonics coefficients for each detector.

    This function computes the blms for a 2D Gaussian beam, accounting for
    detector-specific parameters such as beam width (FWHM), ellipticity,
    and polarization orientation.

    Parameters
    ----------
    observation : Observation
        Observation object containing detector parameters (names, fwhm, etc.).
    lmax : int
        Maximum spherical harmonic degree ℓ_max.
    mmax : int, optional
        Maximum spherical harmonic order m_max. If None, it defaults to `lmax`.
    channels : FreqChannelInfo or list of FreqChannelInfo, optional
        Frequency channels to use. If None, uses detectors from the observation.
    store_in_observation : bool, default=False
        If True, the computed blms will be stored in `observation.blms`.

    Returns
    -------
    dict
        Dictionary mapping detector names (str) to SphericalHarmonics objects.
    """
    mmax_val = mmax if mmax is not None else lmax
    blms = {}

    assert observation.name is not None
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
                cross_polar_leakage=0.0,
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
                psi_ell_rad=channel.psi_rad,
                psi_pol_rad=0.0,  # Channels usually treated as unrotated/unpolarized basis here
                cross_polar_leakage=0.0,
            )

    if store_in_observation:
        observation.blms = blms

    return blms


def gauss_bl(lmax: int, fwhm_rad: float, pol: bool = True) -> np.ndarray:
    """
    Compute the Gaussian beam transfer function b_l analytically (Pure NumPy).

    This implementation computes the beam window function including the
    polarization correction factors (Challinor et al 2000, astro-ph/0008228).

    It returns components for T, E, B (excluding Stokes V).

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree ℓ_max.
    fwhm_rad : float
        Full width at half maximum (FWHM) of the beam in radians.
    pol : bool, optional
        If True, returns an array of shape (3, lmax+1) containing:
          - Index 0: Temperature beam b_l^T
          - Index 1: E-mode polarization beam b_l^E
          - Index 2: B-mode polarization beam b_l^B
        If False, returns a 1D array of shape (lmax+1,) (Temperature only).

    Returns
    -------
    np.ndarray
        Array containing the beam transfer function b_l.
    """
    if fwhm_rad < 0:
        raise ValueError("FWHM must be non-negative.")

    # Create ell array
    ell = np.arange(lmax + 1, dtype=np.float64)

    # Handle trivial case (delta function)
    if fwhm_rad == 0.0:
        base_bl = np.ones_like(ell)
        sigma2 = 0.0
    else:
        # Convert FWHM to sigma: sigma = FWHM / sqrt(8 * ln(2))
        sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
        sigma2 = sigma**2

        # Analytic Gaussian beam: exp(-0.5 * l(l+1) * sigma^2)
        base_bl = np.exp(-0.5 * ell * (ell + 1) * sigma2)

        # Avoid numerical underflow (consistent with healpy behavior)
        base_bl[base_bl < 1e-30] = 0.0

    if not pol:
        return base_bl

    # Polarization factors for [T, E, B]
    # T -> 1.0
    # E, B -> exp(2 * sigma^2) (due to spin-2 nature of polarization)
    pol_factors = np.exp([0.0, 2.0 * sigma2, 2.0 * sigma2])

    # Broadcast to shape (3, lmax+1)
    # base_bl[None, :] is (1, L+1)
    # pol_factors[:, None] is (3, 1)
    return base_bl[None, :] * pol_factors[:, None]
