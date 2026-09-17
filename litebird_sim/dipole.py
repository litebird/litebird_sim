from dataclasses import dataclass, field
from enum import IntEnum
from typing import cast

from numba import njit, prange
import numpy as np

from .maps_and_harmonics import SphericalHarmonics
from .observations import Observation
from .spacecraft import SpacecraftPositionAndVelocity
from .observation_utilities import for_each_observation_with_pointings

# Updated imports to match the new constants.py structure
from .constants import C_LIGHT_KM_OVER_S, H_OVER_K_B, T_CMB_K


# We use a IntEnum class so that comparisons are much faster than with strings
class DipoleType(IntEnum):
    """Doppler-shift model used when adding the CMB dipole to TOD.

    Every member selects a Doppler-shift approximation for the pencil-beam
    (non-convolved) path. Beam convolution is enabled separately by passing
    beam S-parameters (``s_params`` in :func:`add_dipole`, or ``beam_alms`` in
    :func:`add_dipole_to_observations`); convolution supports the
    polynomial-expansion models (``LINEAR``, ``QUADRATIC_EXACT``,
    ``CUBIC_EXACT``, ``QUADRATIC_FROM_LIN_T`` and ``CUBIC_FROM_LIN_T``) — the
    total-formula models ``TOTAL_EXACT`` and ``TOTAL_FROM_LIN_T`` are not
    supported under convolution.
    """

    LINEAR = 0
    r"""Linear approximation in β using thermodynamic units:

    .. math:: \Delta T(\vec\beta, \hat n) = T_0 \vec\beta\cdot\hat n

    """

    QUADRATIC_EXACT = 1
    r"""Second-order approximation in β using thermodynamic units:

    .. math:: \Delta T(\vec\beta, \hat n) = T_0\left(\vec\beta\cdot\hat n +
              \bigl(\vec\beta\cdot\hat n\bigr)^2\right)

    """

    TOTAL_EXACT = 2
    r"""Exact formula in true thermodynamic units:

    .. math:: \frac{T_0}{\gamma \bigl(1 - \vec\beta \cdot \hat n\bigr)}
    """

    CUBIC_EXACT = 3
    r"""Third-order approximation in β using thermodynamic units:

    .. math::

       \Delta T = T_0\left(\vec\beta\cdot\hat n +
       \bigl(\vec\beta\cdot\hat n\bigr)^2 +
       \bigl(\vec\beta\cdot\hat n\bigr)^3\right)

    Monopole terms (from :math:`\sqrt{1-\beta^2}`) are omitted.
    """

    QUADRATIC_FROM_LIN_T = 4
    r"""Second-order approximation in β using linearized units:

    .. math:: \Delta_2 T(\nu) = T_0 \left(\vec\beta\cdot\hat n + q(x)
              \bigl(\vec\beta\cdot\hat n\bigr)^2\right)
    """

    CUBIC_FROM_LIN_T = 5
    r"""Third-order approximation in β using linearized units:

    .. math::

       \Delta T(\nu) = T_0\left(\vec\beta\cdot\hat n +
       q(x)\bigl(\vec\beta\cdot\hat n\bigr)^2 +
       r(x)\bigl(\vec\beta\cdot\hat n\bigr)^3\right)

    where :math:`r(x) = \frac{x^2(e^{2x}+4e^x+1)}{6(e^x-1)^2}`.
    """

    TOTAL_FROM_LIN_T = 6
    r"""Full formula in linearized units (the most widely used):

    .. math::

       \Delta T = \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}\left(T_0 /
       \gamma\bigl(1 - \vec\beta\cdot\hat n\bigr)\right)}{\mathrm{BB}(T_0)}
       - 1\right) = \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}\bigl(\nu
       \gamma(1-\vec\beta\cdot\hat n), T_0\bigr)}{\bigl(\gamma(1-
       \vec\beta\cdot\hat n)\bigr)^3\mathrm{BB}(t_0)}\right).
    """


@dataclass
class BeamSParams:
    r"""Beam-weighted S-parameters for full-4π dipole convolution.

    Stores the beam moment integrals computed from a full 4π beam map
    in the beam frame (boresight along the z-axis):

    .. math::

       S_i     &= \int B(\hat n)\, \hat n_i\, d\Omega \\
       S_{ij}  &= \int B(\hat n)\, \hat n_i\, \hat n_j\, d\Omega \\
       S_{ijk} &= \int B(\hat n)\, \hat n_i\, \hat n_j\, \hat n_k\, d\Omega

    The beam map must be normalised so that
    :math:`\int B(\hat n)\, d\Omega = 1`.

    Fields are ``s_vec`` (shape ``(3,)``), containing the dipole
    moments :math:`(S_x, S_y, S_z)`; ``s_mat`` (shape ``(3, 3)``),
    containing the quadrupole moments :math:`S_{ij}`; and ``s_ten``
    (shape ``(3, 3, 3)``), containing the octupole moments
    :math:`S_{ijk}`.
    """

    s_vec: np.ndarray  # shape (3,)
    s_mat: np.ndarray  # shape (3, 3)
    s_ten: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3, 3))
    )  # shape (3, 3, 3)

    @classmethod
    def from_beam_alm(cls, beam_alm: SphericalHarmonics) -> "BeamSParams":
        r"""Compute S-parameters from full-4π beam harmonic coefficients.

        The beam alms must be in the beam frame with the boresight centred
        on the north pole. It should be normalised so that its integral over
        the sphere equals unity:

        .. math:: \sum_p B_p \cdot \frac{4\pi}{N_\mathrm{pix}} = 1

        Parameters
        ----------
        beam_alm:
            :class:`SphericalHarmonics` object containing beam alms ``B_{\ell m}``.
            If a polarized object (``nstokes=3``) is provided, only the
            temperature component is used.

        Returns
        -------
        BeamSParams
            A :class:`BeamSParams` instance with ``s_vec`` (shape 3),
            ``s_mat`` (shape 3×3), and ``s_ten`` (shape 3×3×3) computed
            using exact harmonic identities (up to :math:`\ell=3` modes),
            without synthesizing a HEALPix map.
        """
        if not isinstance(beam_alm, SphericalHarmonics):
            raise TypeError("beam_alm must be a SphericalHarmonics instance")

        if beam_alm.nfreqs is not None:
            raise ValueError(
                "beam alms must be single-frequency (nfreqs=None) for dipole beam convolution"
            )

        if beam_alm.nstokes == 1:
            alm = beam_alm
        elif beam_alm.nstokes == 3:
            # Use the temperature component for scalar beam convolution.
            alm = SphericalHarmonics(
                values=np.ascontiguousarray(beam_alm.values[0:1]),
                lmax=beam_alm.lmax,
                mmax=beam_alm.mmax,
                units=beam_alm.units,
                coordinates=beam_alm.coordinates,
            )
        else:
            raise ValueError(
                f"beam alms must have nstokes=1 or 3, got {beam_alm.nstokes}"
            )

        a00 = alm.get_coeff(0, 0)

        a10 = alm.get_coeff(1, 0)
        a11 = alm.get_coeff(1, 1)

        a20: int | float | complex = alm.get_coeff(2, 0)
        a21 = alm.get_coeff(2, 1)
        a22: int | float | complex = alm.get_coeff(2, 2)

        a30: int | float | complex = alm.get_coeff(3, 0)
        a31 = alm.get_coeff(3, 1)
        a32: int | float | complex = alm.get_coeff(3, 2)
        a33: int | float | complex = alm.get_coeff(3, 3)

        # For real beam maps and m>=0 storage:
        # ∫B Y_{l,-m} dΩ = (-1)^m a_{lm},  ∫B Y_{lm} dΩ = a_{lm}^*.
        norm = np.sqrt(4.0 * np.pi) * np.conj(a00)

        sx = -np.sqrt(8.0 * np.pi / 3.0) * np.real(a11)
        sy = np.sqrt(8.0 * np.pi / 3.0) * np.imag(a11)
        sz = np.sqrt(4.0 * np.pi / 3.0) * np.real(np.conj(a10))
        s_vec = np.array([sx, sy, sz], dtype=np.float64)

        i_zz = norm / 3.0 + (2.0 / 3.0) * np.sqrt(4.0 * np.pi / 5.0) * np.conj(a20)
        i_xx_minus_yy = np.sqrt(8.0 * np.pi / 15.0) * (np.conj(a22) + a22)
        i_xy2 = 1j * np.sqrt(8.0 * np.pi / 15.0) * (a22 - np.conj(a22))
        i_xz = np.sqrt(2.0 * np.pi / 15.0) * (-a21 - np.conj(a21))
        i_yz = 1j * np.sqrt(2.0 * np.pi / 15.0) * (-a21 + np.conj(a21))

        i_xx_plus_yy = norm - i_zz
        i_xx = 0.5 * (i_xx_plus_yy + i_xx_minus_yy)
        i_yy = 0.5 * (i_xx_plus_yy - i_xx_minus_yy)
        i_xy = 0.5 * i_xy2

        s_mat = np.array(
            [
                [np.real(i_xx), np.real(i_xy), np.real(i_xz)],
                [np.real(i_xy), np.real(i_yy), np.real(i_yz)],
                [np.real(i_xz), np.real(i_yz), np.real(i_zz)],
            ],
            dtype=np.float64,
        )

        # --- Octupole moments S_{ijk} = ∫ B n_i n_j n_k dΩ  (i,j,k ∈ {x,y,z}) ---
        # Cubic monomials decompose as  n_i n_j n_k = (ℓ=1 trace part) + (ℓ=3 traceless part),
        # so only ℓ=1 and ℓ=3 alms contribute.  Expanding each monomial in spherical harmonics
        # and using  S_{ijk} = Σ_{ℓm} a_{ℓm} ∫ n_i n_j n_k Y_{ℓm} dΩ  (convention a = ∫ B Y dΩ)
        # yields a linear combination of a_{1m} and a_{3m} coefficients.
        #
        # Normalisation constants derived from the standard relations
        #   n_z             =  k10 · Re(Y_{10}*)            k10 = sqrt(4π/3)
        #   n_x + i n_y     = -k11 · Y_{11}*                k11 = sqrt(8π/3)
        #   Y_{3m} ∝ n^3 traceless pieces:
        #     m=0  (5n_z³ − 3n_z)      /2    k30 = sqrt(  π/ 7)
        #     m=1  (5n_z² − 1)(n_x±in_y)/... k31 = sqrt(  π/21)
        #     m=2  n_z(n_x ± in_y)²         k32 = sqrt(8π/105)
        #     m=3  (n_x ± in_y)³             k33 = sqrt(  π/35)
        k10 = np.sqrt(4.0 * np.pi / 3.0)
        k11 = np.sqrt(8.0 * np.pi / 3.0)
        k30 = np.sqrt(np.pi / 7.0)
        k31 = np.sqrt(np.pi / 21.0)
        k32 = np.sqrt(8.0 * np.pi / 105.0)
        k33 = np.sqrt(np.pi / 35.0)

        # Each formula has an ℓ=1 part (from the trace: n_i n_j n_k = δ_{ij}n_k/5 + ...)
        # and an ℓ=3 part (the symmetric traceless remainder).
        # Consistency check: S_{xxx}+S_{xyy}+S_{xzz} = S_x, and cyclic permutations.

        # --- m=0/1 terms (real n_z axis, involve a10, a30, a31) ---
        s_zzz = (  # ℓ=1: (3/5)n_z;  ℓ=3: (2/5)P_3(n_z)
            (3.0 / 5.0) * k10 * np.real(np.conj(a10))
            + (4.0 / 5.0) * k30 * np.real(np.conj(a30))
        )
        s_xxz = (  # S_{xxz} = S_{yyz} ± (a32 term)
            (1.0 / 5.0) * k10 * np.real(np.conj(a10))
            - (2.0 / 5.0) * k30 * np.real(np.conj(a30))
            + k32 * np.real(a32)
        )
        s_yyz = (
            (1.0 / 5.0) * k10 * np.real(np.conj(a10))
            - (2.0 / 5.0) * k30 * np.real(np.conj(a30))
            - k32 * np.real(a32)
        )
        s_xzz = (  # ℓ=1 and ℓ=3 m=1 contribute
            -(1.0 / 5.0) * k11 * np.real(a11) - (8.0 / 5.0) * k31 * np.real(a31)
        )
        s_yzz = (1.0 / 5.0) * k11 * np.imag(a11) + (8.0 / 5.0) * k31 * np.imag(a31)

        # --- m=2 term (off-diagonal xy plane) ---
        s_xyz = -k32 * np.imag(a32)  # pure ℓ=3, m=2

        # --- m=1/3 terms (in-plane, involve a11, a31, a33) ---
        s_xxx = (  # ℓ=1 + ℓ=3 m=1 + ℓ=3 m=3
            -(3.0 / 5.0) * k11 * np.real(a11)
            + (6.0 / 5.0) * k31 * np.real(a31)
            - 2.0 * k33 * np.real(a33)
        )
        s_yyy = (
            (3.0 / 5.0) * k11 * np.imag(a11)
            - (6.0 / 5.0) * k31 * np.imag(a31)
            - 2.0 * k33 * np.imag(a33)
        )
        s_xxy = (
            (1.0 / 5.0) * k11 * np.imag(a11)
            - (2.0 / 5.0) * k31 * np.imag(a31)
            + 2.0 * k33 * np.imag(a33)
        )
        s_xyy = (
            -(1.0 / 5.0) * k11 * np.real(a11)
            + (2.0 / 5.0) * k31 * np.real(a31)
            + 2.0 * k33 * np.real(a33)
        )

        s_ten = np.zeros((3, 3, 3), dtype=np.float64)
        s_ten[0, 0, 0] = s_xxx
        s_ten[0, 0, 1] = s_ten[0, 1, 0] = s_ten[1, 0, 0] = s_xxy
        s_ten[0, 0, 2] = s_ten[0, 2, 0] = s_ten[2, 0, 0] = s_xxz
        s_ten[0, 1, 1] = s_ten[1, 0, 1] = s_ten[1, 1, 0] = s_xyy
        s_ten[0, 1, 2] = s_ten[0, 2, 1] = s_ten[1, 0, 2] = s_xyz
        s_ten[1, 2, 0] = s_ten[2, 0, 1] = s_ten[2, 1, 0] = s_xyz
        s_ten[0, 2, 2] = s_ten[2, 0, 2] = s_ten[2, 2, 0] = s_xzz
        s_ten[1, 1, 1] = s_yyy
        s_ten[1, 1, 2] = s_ten[1, 2, 1] = s_ten[2, 1, 1] = s_yyz
        s_ten[1, 2, 2] = s_ten[2, 1, 2] = s_ten[2, 2, 1] = s_yzz
        s_ten[2, 2, 2] = s_zzz

        return cls(s_vec=s_vec, s_mat=s_mat, s_ten=s_ten)


@njit
def _compute_f_x(x):
    """Spectral function f(x) = x e^x / (e^x - 1)."""
    return x * np.exp(x) / (np.exp(x) - 1)


@njit
def _compute_q_x(x):
    """Quadrupole spectral weight q(x) = x(e^x+1) / (2(e^x-1))."""
    return 0.5 * x * (np.exp(x) + 1) / (np.exp(x) - 1)


@njit
def _compute_r_x(x):
    """Octupole spectral weight r(x) = x^2(e^{2x}+4e^x+1) / (6(e^x-1)^2)."""
    return x**2 * (np.exp(2 * x) + 4 * np.exp(x) + 1) / (6 * (np.exp(x) - 1) ** 2)


@njit
def _compute_occupation(x):
    """Return occupation number for x"""
    return 1 / (np.expm1(x))


@njit
def _compute_scalar_product(theta, phi, v):
    """Return the scalar (dot) product between a given direction and a velocity"""
    dx, dy, dz = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

    return dx * v[0] + dy * v[1] + dz * v[2]


@njit
def _calculate_beta(theta, phi, v_km_s):
    """Return a 2-tuple containing β·n and β"""
    beta_dot_n = _compute_scalar_product(theta, phi, v_km_s) / C_LIGHT_KM_OVER_S
    beta = np.sqrt(v_km_s[0] ** 2 + v_km_s[1] ** 2 + v_km_s[2] ** 2) / C_LIGHT_KM_OVER_S

    return beta_dot_n, beta


@njit
def _rotate_velocity_to_beam_frame(theta, phi, psi, v_km_s):
    """Rotate a velocity vector from the sky frame into the beam frame.

    The beam frame is defined such that its z-axis points along the boresight
    direction (theta, phi), and its x-axis is rotated by psi from the local
    co-latitude unit vector ê_θ = (cos θ cos φ, cos θ sin φ, -sin θ):

        x̂_beam = cos ψ ê_θ - sin ψ ê_φ
        ŷ_beam = sin ψ ê_θ + cos ψ ê_φ
        ẑ_beam = ê_r = (sin θ cos φ, sin θ sin φ, cos θ)

    Returns a 3-tuple (v_x_beam, v_y_beam, v_z_beam).
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    v0 = v_km_s[0]
    v1 = v_km_s[1]
    v2 = v_km_s[2]

    # ê_θ = (ct*cp, ct*sp, -st),  ê_φ = (-sp, cp, 0)
    vx = (
        (cpsi * ct * cp + spsi * sp) * v0
        + (cpsi * ct * sp - spsi * cp) * v1
        + (-cpsi * st) * v2
    )
    vy = (
        (spsi * ct * cp - cpsi * sp) * v0
        + (spsi * ct * sp + cpsi * cp) * v1
        + (-spsi * st) * v2
    )
    vz = st * cp * v0 + st * sp * v1 + ct * v2

    return vx, vy, vz


@njit
def compute_dipole_for_one_sample_linear(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n = _compute_scalar_product(theta, phi, v_km_s) / C_LIGHT_KM_OVER_S
    return t_cmb_k * beta_dot_n


@njit
def compute_dipole_for_one_sample_quadratic_exact(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n, beta = _calculate_beta(theta, phi, v_km_s)

    # Up to second order in beta, including second order in the expansion of
    # thermodynamic temperature. This is in true temperature, and
    # no boosting induced monopoles are added.
    return t_cmb_k * (beta_dot_n + beta_dot_n**2)


@njit
def compute_dipole_for_one_sample_total_exact(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n, beta = _calculate_beta(theta, phi, v_km_s)
    gamma = 1 / np.sqrt(1 - beta**2)

    return t_cmb_k / gamma / (1 - beta_dot_n) - t_cmb_k


@njit
def compute_dipole_for_one_sample_quadratic_from_lin_t(
    theta, phi, v_km_s, t_cmb_k, q_x
):
    # Up to second order in beta, including second order in the expansion of
    # thermodynamic temperature. This is in linearized thermodynamic temperature.
    # No boosting induced monopoles are added
    beta_dot_n, beta = _calculate_beta(theta, phi, v_km_s)
    return t_cmb_k * (beta_dot_n + q_x * beta_dot_n**2)


@njit
def compute_dipole_for_one_sample_cubic_exact(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n, beta = _calculate_beta(theta, phi, v_km_s)
    # Third order in thermodynamic units; all monopole contributions
    # (from sqrt(1-beta^2)) are omitted, consistent with QUADRATIC_EXACT.
    return t_cmb_k * (beta_dot_n + beta_dot_n**2 + beta_dot_n**3)


@njit
def compute_dipole_for_one_sample_cubic_from_lin_t(
    theta, phi, v_km_s, t_cmb_k, q_x, r_x
):
    # Third order in linearized units: r_x = x^2*(exp(2x)+4*exp(x)+1)/(6*(exp(x)-1)^2)
    # No boosting induced monopoles are added.
    beta_dot_n, beta = _calculate_beta(theta, phi, v_km_s)
    return t_cmb_k * (beta_dot_n + q_x * beta_dot_n**2 + r_x * beta_dot_n**3)


@njit
def compute_dipole_for_one_sample_total_from_lin_t(
    theta, phi, v_km_s, t_cmb_k, nu_hz, f_x, n_x
):
    beta_dot_n, beta = _calculate_beta(theta, phi, v_km_s)
    gamma = 1 / np.sqrt(1 - beta**2)

    x_doppler = H_OVER_K_B * nu_hz * gamma * (1 - beta_dot_n) / t_cmb_k
    n_x_shifted = _compute_occupation(x_doppler)

    return t_cmb_k / f_x * (n_x_shifted / n_x - 1)


@njit
def compute_dipole_for_one_sample_expanded(theta, phi, v_km_s, t_cmb_k, q_x, r_x):
    r"""Polynomial β-expansion for one TOD sample (no beam convolution).

    Computes

    .. math::

       \Delta T = T_0\bigl(\beta\cdot\hat n
                 + q_x (\beta\cdot\hat n)^2
                 + r_x (\beta\cdot\hat n)^3\bigr)

    This is the scalar (delta-function beam) limit of
    :func:`compute_dipole_for_one_sample_convolved`, where
    :math:`S_i = \hat n_i`, :math:`S_{ij} = \hat n_i \hat n_j`, etc.
    The caller selects the approximation order by choosing *q_x* and *r_x*:

    ========================== ============== ==============
    DipoleType                 q_x            r_x
    ========================== ============== ==============
    ``LINEAR``                 ``0``          ``0``
    ``QUADRATIC_EXACT``        ``1``          ``0``
    ``CUBIC_EXACT``            ``1``          ``1``
    ``QUADRATIC_FROM_LIN_T``   ``q(x)``       ``0``
    ``CUBIC_FROM_LIN_T``       ``q(x)``       ``r(x)``
    ========================== ============== ==============
    """
    beta_dot_n, _ = _calculate_beta(theta, phi, v_km_s)
    if q_x == 0.0:
        return t_cmb_k * beta_dot_n
    elif r_x == 0.0:
        return t_cmb_k * (beta_dot_n + q_x * beta_dot_n**2)
    else:
        return t_cmb_k * (beta_dot_n + q_x * beta_dot_n**2 + r_x * beta_dot_n**3)


@njit
def compute_dipole_for_one_sample_total(theta, phi, v_km_s, t_cmb_k, nu_hz, f_x, n_x):
    r"""Exact or full-Planck formula for one TOD sample.

    Selects the formula via *f_x*:

    - ``f_x == 0``  →  thermodynamic exact (``TOTAL_EXACT``)

      .. math:: \Delta T = \frac{T_0}{\gamma(1 - \beta\cdot\hat n)} - T_0

    - ``f_x ≠ 0``  →  full Planck in linearised units (``TOTAL_FROM_LIN_T``)

      .. math:: \Delta T = \frac{T_0}{f(x)}\left(
                \frac{\bar n(x_\mathrm{Doppler})}{\bar n(x)} - 1\right)

    where :math:`x_\mathrm{Doppler} = \frac{h\nu}{k_B T_0}\gamma(1-\beta\cdot\hat n)`.

    For ``TOTAL_EXACT`` pass *nu_hz* = 0, *f_x* = 0, *n_x* = 1 (unused).
    """
    beta_dot_n, beta = _calculate_beta(theta, phi, v_km_s)
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    if f_x == 0.0:
        return t_cmb_k / gamma / (1.0 - beta_dot_n) - t_cmb_k
    else:
        x_doppler = H_OVER_K_B * nu_hz * gamma * (1.0 - beta_dot_n) / t_cmb_k
        n_x_shifted = _compute_occupation(x_doppler)
        return t_cmb_k / f_x * (n_x_shifted / n_x - 1.0)


@njit
def compute_dipole_for_one_sample_convolved(
    theta,
    phi,
    psi,
    v_km_s,
    t_cmb_k,
    q_x,
    r_x,
    s_vec,
    s_mat,
    s_ten,
):
    r"""Compute the beam-convolved dipole+quadrupole+octupole for one TOD sample.

    Implements Eq. (C.5) of the Planck NPIPE paper (arXiv:2007.04997)
    adding the octupole term for the CUBIC_FROM_LIN_T case:

        D̃ = T₀ [ Sᵢ βᵢ  +  q(x) Sᵢⱼ βᵢ βⱼ +  r(x) Sᵢⱼₖ βᵢ βⱼ βₖ ]

    where β is the velocity divided by c expressed in the beam frame
    (obtained by rotating v_km_s with (theta, phi, psi)), and the
    S-parameters are the beam-weighted integrals stored in s_vec / s_mat / s_ten.
    """
    vx, vy, vz = _rotate_velocity_to_beam_frame(theta, phi, psi, v_km_s)
    bx = vx / C_LIGHT_KM_OVER_S
    by = vy / C_LIGHT_KM_OVER_S
    bz = vz / C_LIGHT_KM_OVER_S

    # Dipole term: S_i β_i
    dipole = s_vec[0] * bx + s_vec[1] * by + s_vec[2] * bz

    # Quadrupole term: S_ij β_i β_j  (S is symmetric)
    b = (bx, by, bz)

    if q_x == 0.0:
        return t_cmb_k * dipole
    else:
        quadrupole = 0.0
        for i in range(3):
            for j in range(3):
                quadrupole += s_mat[i, j] * b[i] * b[j]
        if r_x == 0.0:
            return t_cmb_k * (dipole + q_x * quadrupole)
        else:
            octupole = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        octupole += s_ten[i, j, k] * b[i] * b[j] * b[k]
            return t_cmb_k * (dipole + q_x * quadrupole + r_x * octupole)


@njit(parallel=True)
def add_dipole_for_one_detector(
    tod_det,
    theta_phi_det,
    velocity,
    t_cmb_k,
    nu_hz,
    dipole_type: DipoleType,
):
    x = H_OVER_K_B * nu_hz / t_cmb_k
    n_x = _compute_occupation(x)
    f_x = _compute_f_x(x)
    q_x = _compute_q_x(x)
    r_x = _compute_r_x(x)

    if dipole_type == DipoleType.TOTAL_EXACT:
        # Exact thermodynamic formula: f_x=0 signals compute_dipole_for_one_sample_total
        # to use the T₀/γ/(1−β·n̂) branch.
        for i in prange(len(tod_det)):  # type: ignore[not-iterable]
            tod_det[i] += compute_dipole_for_one_sample_total(
                theta=theta_phi_det[i, 0],
                phi=theta_phi_det[i, 1],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
                nu_hz=0.0,
                f_x=0.0,
                n_x=1.0,
            )
    elif dipole_type == DipoleType.TOTAL_FROM_LIN_T:
        for i in prange(len(tod_det)):  # type: ignore[not-iterable]
            tod_det[i] += compute_dipole_for_one_sample_total(
                theta=theta_phi_det[i, 0],
                phi=theta_phi_det[i, 1],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
                nu_hz=nu_hz,
                f_x=f_x,
                n_x=n_x,
            )
    else:
        # All polynomial-expansion types, mapped to (q_eff, r_eff):
        #   LINEAR               → (0,    0)
        #   QUADRATIC_EXACT      → (1,    0)
        #   CUBIC_EXACT          → (1,    1)
        #   QUADRATIC_FROM_LIN_T → (q(x), 0)
        #   CUBIC_FROM_LIN_T     → (q(x), r(x))
        if dipole_type == DipoleType.LINEAR:
            q_eff = 0.0
            r_eff = 0.0
        elif dipole_type == DipoleType.QUADRATIC_EXACT:
            q_eff = 1.0
            r_eff = 0.0
        elif dipole_type == DipoleType.CUBIC_EXACT:
            q_eff = 1.0
            r_eff = 1.0
        elif dipole_type == DipoleType.QUADRATIC_FROM_LIN_T:
            q_eff = q_x
            r_eff = 0.0
        elif dipole_type == DipoleType.CUBIC_FROM_LIN_T:
            q_eff = q_x
            r_eff = r_x
        else:
            print("Dipole Type not implemented!!!")
            q_eff = 0.0
            r_eff = 0.0
        for i in prange(len(tod_det)):  # type: ignore[not-iterable]
            tod_det[i] += compute_dipole_for_one_sample_expanded(
                theta=theta_phi_det[i, 0],
                phi=theta_phi_det[i, 1],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
                q_x=q_eff,
                r_x=r_eff,
            )


@njit(parallel=True)
def add_dipole_for_one_detector_convolved(
    tod_det,
    theta_phi_psi_det,
    velocity,
    t_cmb_k,
    nu_hz,
    dipole_type: DipoleType,
    s_vec,
    s_mat,
    s_ten,
):
    x = H_OVER_K_B * nu_hz / t_cmb_k

    if dipole_type == DipoleType.LINEAR:
        q_x = 0.0
        r_x = 0.0
    elif dipole_type == DipoleType.QUADRATIC_EXACT:
        q_x = 1.0
        r_x = 0.0
    elif dipole_type == DipoleType.CUBIC_EXACT:
        q_x = 1.0
        r_x = 1.0
    elif dipole_type == DipoleType.QUADRATIC_FROM_LIN_T:
        q_x = _compute_q_x(x)
        r_x = 0.0
    elif dipole_type == DipoleType.CUBIC_FROM_LIN_T:
        q_x = _compute_q_x(x)
        r_x = _compute_r_x(x)
    else:
        print("Convolved Dipole Type not implemented!!!")

    for i in prange(len(tod_det)):  # type: ignore[not-iterable]
        tod_det[i] += compute_dipole_for_one_sample_convolved(
            theta=theta_phi_psi_det[i, 0],
            phi=theta_phi_psi_det[i, 1],
            psi=theta_phi_psi_det[i, 2],
            v_km_s=velocity[i],
            t_cmb_k=t_cmb_k,
            q_x=q_x,
            r_x=r_x,
            s_vec=s_vec,
            s_mat=s_mat,
            s_ten=s_ten,
        )


def add_dipole(
    tod,
    pointings,
    velocity,
    t_cmb_k: float,
    frequency_ghz: np.ndarray,  # e.g. central frequency of channel
    dipole_type: DipoleType,
    pointings_dtype=np.float64,
    input_detector_names: list[str] | str | None = None,
    s_params: (BeamSParams | dict[str, BeamSParams] | None) = None,
):
    """Add the CMB dipole contribution to time-ordered data (TOD).

    This array-oriented helper operates on one TOD matrix containing all
    detectors for one observation. By default it evaluates the selected
    :class:`DipoleType` along the detector boresight (pencil beam). If
    ``s_params`` is supplied, it instead evaluates the moment-expanded
    full-4π beam convolution, contracting the beam S-parameters with the
    velocity in the beam frame.

    Parameters
    ----------
    tod:
        2-D time-ordered data array for all detectors (shape ``(n_det, n_samples)``).
    pointings:
        Pointing matrices. If ``s_params`` is given, the matrix must
        include the ``psi`` column (shape ``(n_det, n_samples, 3)``).
        Otherwise, only ``theta`` and ``phi`` are used (shape
        ``(n_det, n_samples, 2)`` or more).
    velocity:
        2-D array of shape ``(n_samples, 3)`` with the velocity in km/s.
    t_cmb_k:
        CMB monopole temperature in kelvin.
    frequency_ghz:
        1-D array with frequency in GHz for each detector.
    dipole_type:
        DipoleType enum specifying the Doppler shift approximation. When
        ``s_params`` is given (beam convolution), the polynomial-expansion
        models are supported (``LINEAR``, ``QUADRATIC_EXACT``, ``CUBIC_EXACT``,
        ``QUADRATIC_FROM_LIN_T``, ``CUBIC_FROM_LIN_T``); the total formulae
        ``TOTAL_EXACT`` and ``TOTAL_FROM_LIN_T`` are not.
    pointings_dtype:
        Data type for pointings if generated on the fly.
    s_params:
        Beam S-parameters (:class:`BeamSParams`). When given, the dipole is
        beam-convolved by moment expansion. Pass a single :class:`BeamSParams`
        to use the same beam for all detectors, or a dictionary keyed by
        detector index strings (``"0"``, ``"1"``, ...) for per-detector beams.
        If ``None`` (default), the pencil-beam dipole is computed.
    """

    n_detectors = tod.shape[0]

    if type(pointings) is np.ndarray:
        assert tod.shape == pointings.shape[0:2]

    assert tod.shape[1] == velocity.shape[0]

    if s_params is not None and dipole_type in (
        DipoleType.TOTAL_EXACT,
        DipoleType.TOTAL_FROM_LIN_T,
    ):
        raise ValueError(
            f"dipole_type={dipole_type.name} is not supported under beam "
            "convolution. Use LINEAR, QUADRATIC_FROM_LIN_T or CUBIC_FROM_LIN_T."
        )

    if s_params is not None:
        if isinstance(s_params, dict):
            assert len(s_params.values()) == n_detectors
            assert all(isinstance(v, BeamSParams) for v in s_params.values())

            assert isinstance(input_detector_names, list)
            assert len(input_detector_names) == n_detectors
            assert all(isinstance(v, str) for v in input_detector_names)
        else:
            assert isinstance(s_params, BeamSParams)

    for detector_idx in range(n_detectors):
        nu_hz = frequency_ghz[detector_idx] * 1e9  # freq in GHz

        if s_params is not None:
            # Beam-convolved (moment-expansion) path: needs the ψ column.
            if isinstance(s_params, dict):
                assert all(isinstance(v, BeamSParams) for v in s_params.values())

                s_params_dict = cast(dict[str, BeamSParams], s_params)
                det_names = cast(list[str], input_detector_names)
                sp = s_params_dict.get(det_names[detector_idx])
                if sp is None:
                    raise ValueError(
                        f"s_params dictionary missing key for detector {detector_idx}"
                    )
            else:
                assert isinstance(s_params, BeamSParams)
                sp = s_params

            if type(pointings) is np.ndarray:
                theta_phi_psi_det = pointings[detector_idx, :, :]
            else:
                theta_phi_psi_det = pointings(
                    detector_idx, pointings_dtype=pointings_dtype
                )[0][:, 0:3]

            add_dipole_for_one_detector_convolved(
                tod_det=tod[detector_idx],
                theta_phi_psi_det=theta_phi_psi_det,
                velocity=velocity,
                t_cmb_k=t_cmb_k,
                nu_hz=nu_hz,
                dipole_type=dipole_type,
                s_vec=sp.s_vec,
                s_mat=sp.s_mat,
                s_ten=sp.s_ten,
            )
        else:
            # Standard (non-convolved) dipole calculation
            if type(pointings) is np.ndarray:
                theta_phi_det = pointings[detector_idx, :, 0:2]
            else:
                theta_phi_det = pointings(
                    detector_idx, pointings_dtype=pointings_dtype
                )[0][:, 0:2]

            add_dipole_for_one_detector(
                tod_det=tod[detector_idx],
                theta_phi_det=theta_phi_det,
                velocity=velocity,
                t_cmb_k=t_cmb_k,
                nu_hz=nu_hz,
                dipole_type=dipole_type,
            )


def add_dipole_to_observations(
    observations: Observation | list[Observation],
    pos_and_vel: SpacecraftPositionAndVelocity,
    pointings: np.ndarray | list[np.ndarray] | None = None,
    t_cmb_k: float = T_CMB_K,
    dipole_type: DipoleType = DipoleType.TOTAL_FROM_LIN_T,
    frequency_ghz: (np.ndarray | None) = None,
    component: str = "tod",
    pointings_dtype=np.float64,
    apply_convolution: bool = False,
    beam_alms: (SphericalHarmonics | dict[str, SphericalHarmonics] | None) = None,
):
    """Add the CMB dipole signal to the time-ordered data (TOD) stored
    in one or more `Observation` objects.

    This is the observation-oriented wrapper around :func:`add_dipole`. It
    interpolates the spacecraft velocity to each observation's sampling and
    obtains pointings either from the explicit ``pointings`` argument, from a
    precomputed ``pointing_matrix`` attribute, or from
    :meth:`Observation.get_pointings`.

    The function supports both pencil-beam and moment-expanded full-4π
    beam-convolved dipole calculations. When ``apply_convolution=True``,
    ``beam_alms`` can be provided explicitly or retrieved from the
    observation's ``blms`` attribute.

    Parameters
    ----------
    observations : Observation | list[Observation]
        One or more Observation objects containing detector data.
    pos_and_vel : SpacecraftPositionAndVelocity
        Spacecraft position and velocity data.
    pointings : np.ndarray | list[np.ndarray] | None, default=None
        Pointing matrices. If None, extracted from observations. Convolved
        calculations require the ``psi`` column, i.e. shape
        ``(n_det, n_samples, 3)``.
    t_cmb_k : float, default=T_CMB_K
        CMB monopole temperature in Kelvin.
    dipole_type : DipoleType, default=TOTAL_FROM_LIN_T
        Type of dipole approximation to use. With ``apply_convolution=True``,
        choose one of the polynomial-expansion models; ``TOTAL_EXACT`` and
        ``TOTAL_FROM_LIN_T`` are not supported by the moment-expanded
        convolution.
    frequency_ghz : np.ndarray | None, default=None
        Frequencies in GHz for each detector. If None, taken from observation.
    component : str, default="tod"
        TOD component name to modify.
    pointings_dtype : dtype, default=np.float64
        Data type for computed pointings.
    apply_convolution : bool, default=False
        If True, apply beam convolution using the provided or observed beam_alms.
    beam_alms : SphericalHarmonics | dict[str, SphericalHarmonics] | None, default=None
        Beam harmonic coefficients. If None and apply_convolution=True,
        attempts to retrieve from observation.blms. If a dictionary, its
        keys should correspond to detector or channel names (consistent
        with :func:`.add_convolved_sky`).
    """
    # For convolved types we keep the full (θ, φ, ψ) columns; otherwise strip to (θ, φ).
    ptg_cols = slice(None) if apply_convolution else slice(0, 2)

    for cur_obs, tod, cur_ptg in for_each_observation_with_pointings(
        observations, pointings, component
    ):
        # Callables (lazy pointings) are forwarded untouched, matching the
        # behaviour of the underlying normalizer. Convolved calculations need
        # the psi column too, hence ptg_cols rather than a hardcoded 0:2.
        if isinstance(cur_ptg, np.ndarray):
            cur_ptg = cur_ptg[:, :, ptg_cols]

        # Resolve the beam S-parameters when convolution is requested.
        # 1. Use the explicitly provided beam_alms if given.
        # 2. Otherwise fall back to the observation's stored blms.
        # When apply_convolution is False, any stored/passed beam is ignored
        # and the pencil-beam dipole is computed.
        cur_s_params = None
        input_detector_names = None
        if apply_convolution:
            cur_beam_alms = beam_alms
            if cur_beam_alms is None:
                try:
                    cur_beam_alms = cur_obs.blms
                except AttributeError:
                    msg = (
                        "apply_convolution=True but beam_alms is None and observation "
                        "has no 'blms' attribute. Provide beam_alms explicitly or store "
                        "them in the observation (e.g., using get_gauss_beam_alms)."
                    )
                    raise AttributeError(msg)

            if isinstance(cur_beam_alms, dict):
                input_detector_names = (
                    list(cur_obs.name)
                    if all(k in cur_beam_alms for k in cur_obs.name)
                    else list(cur_obs.channel)
                    if all(k in cur_beam_alms for k in cur_obs.channel)
                    else None
                )
                if input_detector_names is None:
                    raise ValueError(
                        "Beam blms dictionary keys do not match detector/channel names."
                    )
                beam_alms_dict = cast(dict[str, SphericalHarmonics], cur_beam_alms)
                cur_s_params = {
                    key: BeamSParams.from_beam_alm(alm)
                    for key, alm in beam_alms_dict.items()
                }
            else:
                assert isinstance(cur_beam_alms, SphericalHarmonics), (
                    "Invalid blms format."
                )
                input_detector_names = None
                cur_s_params = BeamSParams.from_beam_alm(cur_beam_alms)

        # Alas, this allocates memory for the velocity vector! At the moment it is the
        # simplest implementation, but in the future we might want to inline the
        # interpolation code within "add_dipole" to save memory
        velocity = pos_and_vel.compute_velocities(
            time0=cur_obs.start_time,
            delta_time_s=cur_obs.get_delta_time().value,
            num_of_samples=tod.shape[1],
        )

        if frequency_ghz is None:
            # Assumes cur_obs has bandcenter_ghz, otherwise user must provide frequency_ghz
            frequency_ghz_arr = cur_obs.bandcenter_ghz
        else:
            frequency_ghz_arr = np.repeat(frequency_ghz, tod.shape[0])

        add_dipole(
            tod=tod,
            pointings=cur_ptg,
            velocity=velocity,
            t_cmb_k=t_cmb_k,
            frequency_ghz=frequency_ghz_arr,
            dipole_type=dipole_type,
            pointings_dtype=pointings_dtype,
            input_detector_names=input_detector_names,
            s_params=cur_s_params,
        )
