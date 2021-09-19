# -*- encoding: utf-8 -*-

from enum import IntEnum

from numba import njit
import numpy as np

from typing import Union, List

from astropy.constants import c as c_light
from astropy.constants import h, k_B

from .observations import Observation
from .spacecraft import SpacecraftPositionAndVelocity

C_LIGHT_KM_S = c_light.value / 1e3
H_OVER_K_B = h.value / k_B.value


# We use a IntEnum class so that comparisons are much faster than with strings
class DipoleType(IntEnum):
    """Approximation for the Doppler shift caused by the motion of the spacecraft"""

    LINEAR = 0
    r"""Linear approximation in β using thermodynamic units:

    .. math:: \Delta T(\vec\beta, \hat n) = T_0 \vec\beta\cdot\hat n

    """ ""

    QUADRATIC_EXACT = 1
    r"""Second-order approximation in β using thermodynamic units:

    .. math:: \Delta T(\vec\beta, \hat n) = T_0\left(\vec\beta\cdot\hat n +
              \bigl(\vec\beta\cdot\hat n\bigr)^2\right)

    """

    TOTAL_EXACT = 2
    r"""Exact formula in true thermodynamic units:

    .. math:: \frac{T_0}{\gamma \bigl(1 - \vec\beta \cdot \hat n\bigr)}
    """

    QUADRATIC_FROM_LIN_T = 3
    r"""Second-order approximation in β using linearized units:

    .. math:: \Delta_2 T(\nu) = T_0 \left(\vec\beta\cdot\hat n + q(x)
              \bigl(\vec\beta\cdot\hat n\bigr)^2\right)
    """

    TOTAL_FROM_LIN_T = 4
    r"""Full formula in linearized units (the most widely used):

    .. math::

       \Delta T = \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}\left(T_0 /
       \gamma\bigl(1 - \vec\beta\cdot\hat n\bigr)\right)}{\mathrm{BB}(T_0)}
       - 1\right) = \frac{T_0}{f(x)} \left(\frac{\mathrm{BB}\bigl(\nu
       \gamma(1-\vec\beta\cdot\hat n), T_0\bigr)}{\bigl(\gamma(1-
       \vec\beta\cdot\hat n)\bigr)^3\mathrm{BB}(t_0)}\right).
    """


@njit
def planck(nu_hz, t_k):
    """Return occupation number at frequency nu_hz and temperature t_k"""
    return 1 / (np.exp(H_OVER_K_B * nu_hz / t_k) - 1)


@njit
def compute_scalar_product(theta, phi, v):
    """Return the scalar (dot) product between a given direction and a velocity"""
    dx, dy, dz = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

    return dx * v[0] + dy * v[1] + dz * v[2]


@njit
def calculate_beta(theta, phi, v_km_s):
    """Return a 2-tuple containing β·n and β"""
    beta_dot_n = compute_scalar_product(theta, phi, v_km_s) / C_LIGHT_KM_S
    beta = np.sqrt(v_km_s[0] ** 2 + v_km_s[1] ** 2 + v_km_s[2] ** 2) / C_LIGHT_KM_S

    return beta_dot_n, beta


@njit
def compute_dipole_for_one_sample_linear(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n = compute_scalar_product(theta, phi, v_km_s) / C_LIGHT_KM_S
    return t_cmb_k * beta_dot_n


@njit
def compute_dipole_for_one_sample_quadratic_exact(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)

    # Up to second order in beta, including second order in the expansion of
    # thermodynamic temperature. This is in true temperature, and
    # no boosting induced monopoles are added.
    return t_cmb_k * (beta_dot_n + beta_dot_n ** 2)


@njit
def compute_dipole_for_one_sample_total_exact(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    gamma = 1 / np.sqrt(1 - beta ** 2)

    return t_cmb_k / gamma / (1 - beta_dot_n) - t_cmb_k


@njit
def compute_dipole_for_one_sample_quadratic_from_lin_t(
    theta, phi, v_km_s, t_cmb_k, q_x
):
    # Up to second order in beta, including second order in the expansion of
    # thermodynamic temperature. This is in linearized thermodynamic temperature.
    # No boosting induced monopoles are added
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    return t_cmb_k * (beta_dot_n + q_x * beta_dot_n ** 2)


@njit
def compute_dipole_for_one_sample_total_from_lin_t(
    theta, phi, v_km_s, t_cmb_k, nu_hz, f_x, planck_t0
):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    gamma = 1 / np.sqrt(1 - beta ** 2)

    planck_t = planck(nu_hz, t_cmb_k / gamma / (1 - beta_dot_n))

    return t_cmb_k / f_x * (planck_t / planck_t0 - 1)


@njit
def add_dipole_for_one_detector(
    tod_det,
    theta_det,
    phi_det,
    velocity,
    t_cmb_k,
    nu_hz,
    f_x,
    q_x,
    dipole_type: DipoleType,
):

    if dipole_type == DipoleType.LINEAR:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_linear(
                theta=theta_det[i], phi=phi_det[i], v_km_s=velocity[i], t_cmb_k=t_cmb_k
            )
    elif dipole_type == DipoleType.QUADRATIC_EXACT:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_quadratic_exact(
                theta=theta_det[i], phi=phi_det[i], v_km_s=velocity[i], t_cmb_k=t_cmb_k
            )
    elif dipole_type == DipoleType.TOTAL_EXACT:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_total_exact(
                theta=theta_det[i], phi=phi_det[i], v_km_s=velocity[i], t_cmb_k=t_cmb_k
            )
    elif dipole_type == DipoleType.QUADRATIC_FROM_LIN_T:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_quadratic_from_lin_t(
                theta=theta_det[i],
                phi=phi_det[i],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
                q_x=q_x,
            )
    elif dipole_type == DipoleType.TOTAL_FROM_LIN_T:
        planck_t0 = planck(nu_hz, t_cmb_k)
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_total_from_lin_t(
                theta=theta_det[i],
                phi=phi_det[i],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
                nu_hz=nu_hz,
                f_x=f_x,
                planck_t0=planck_t0,
            )
    else:
        print("Dipole Type not implemented!!!")


def add_dipole(
    tod,
    pointings,
    velocity,
    t_cmb_k: float,
    frequency_ghz: np.ndarray,  # e.g. central frequency of channel from
    # lbs.FreqChannelInfo.from_imo(url=…, imo=imo).bandcenter_ghz
    # using as url f"/releases/v1.0/satellite/{telescope}/{channel}/channel_info"
    dipole_type: DipoleType,
):
    """Add the CMB dipole to some time-ordered data

    This functions modifies the values in `tod` by adding the contribution of the
    CMB dipole. Use `dipole_type` to specify which kind of approximation to use
    for the dipole component. The `pointings` argument must be a N×3 matrix containing
    the pointing information, where N is the size of the `tod` array. The `velocity`
    argument is usually computed through :func:`.spacecraft_pos_and_vel`. Finally,
    `t_cmb_k` is the temperature of the monopole and `frequency_ghz` is an array
    containing the frequencies of each detector in the TOD."""

    assert tod.shape == pointings.shape[0:2]
    assert tod.shape[1] == velocity.shape[0]

    for detector_idx in range(tod.shape[0]):

        nu_hz = frequency_ghz[detector_idx] * 1e9  # freq in GHz
        # Note that x is a dimensionless parameter
        x = h.value * nu_hz / (k_B.value * t_cmb_k)

        f_x = x * np.exp(x) / (np.exp(x) - 1)

        q_x = 0.5 * x * (np.exp(x) + 1) / (np.exp(x) - 1)

        add_dipole_for_one_detector(
            tod_det=tod[detector_idx],
            theta_det=pointings[detector_idx, :, 0],
            phi_det=pointings[detector_idx, :, 1],
            velocity=velocity,
            t_cmb_k=t_cmb_k,
            nu_hz=nu_hz,
            f_x=f_x,
            q_x=q_x,
            dipole_type=dipole_type,
        )


def add_dipole_to_observations(
    obs: Union[Observation, List[Observation]],
    pointings,
    pos_and_vel: SpacecraftPositionAndVelocity,
    t_cmb_k: float = 2.72548,  # Fixsen 2009 http://arxiv.org/abs/0911.1955
    dipole_type: DipoleType = DipoleType.TOTAL_FROM_LIN_T,
    frequency_ghz: Union[
        np.ndarray, None
    ] = None,  # e.g. central frequency of channel from
):
    """Add the CMB dipole to some time-ordered data

    This is a wrapper around the :func:`.add_dipole` function that applies to the TOD
    stored in `obs`, which can either be one :class:`.Observation` instance or a list
    of observations.
    """

    if isinstance(obs, Observation):
        obs_list = [obs]
    else:
        obs_list = obs

    for cur_obs in obs_list:
        # Alas, this allocates memory for the velocity vector! At the moment it is the
        # simplest implementation, but in the future we might want to inline the
        # interpolation code within "add_dipole" to save memory
        velocity = pos_and_vel.compute_velocities(
            time0=cur_obs.start_time,
            delta_time_s=cur_obs.get_delta_time().value,
            num_of_samples=cur_obs.tod.shape[1],
        )

        if frequency_ghz is None:
            frequency_ghz = cur_obs.bandcenter_ghz
        else:
            frequency_ghz = np.repeat(frequency_ghz, cur_obs.tod.shape[0])

        add_dipole(
            tod=cur_obs.tod,
            pointings=pointings,
            velocity=velocity,
            t_cmb_k=t_cmb_k,
            frequency_ghz=frequency_ghz,
            dipole_type=dipole_type,
        )
