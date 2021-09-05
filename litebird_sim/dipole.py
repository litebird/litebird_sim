# -*- encoding: utf-8 -*-

from enum import IntEnum

from numba import njit
import numpy as np

from typing import Union

from astropy.constants import c as c_light
import astropy
from astropy.constants import h, k_B

from .observations import Observation
from .spacecraft import SpacecraftPositionAndVelocity

C_LIGHT_KM_S = c_light.value / 1e3


# We use a IntEnum class so that comparisons are much faster than with strings
class DipoleType(IntEnum):
    """Kind of calculation to use when estimating the Doppler shift caused by the motion of the spacecraft"""

    # Simple linear approximation (the fastest calculation)
    LINEAR = 0

    # Up to second order in β, including second order in the expansion of the thermodynamic temperature
    QUADRATIC_EXACT = 1

    # Total contribution (the slowest but more accurate formula, correct only in true thermodynamic units) 
    TOTAL_EXACT = 2

    # Up to second order in β, using the linear temperature approximation (linearization of thermodynamic temperature)
    # This is the formula to use if you want the leading frequency dependent term (second order) neglecting the boosting induced monopoles
    QUADRATIC_FROM_LIN_T = 3

    # Total contribution, using the linear temperature approximation (the slowest but more accurate formula)
    # Linear temperature approximation is tipically used in CMB experiments 
    # This is the formula to use if you want the frequency dependent terms at all orders
    TOTAL_FROM_LIN_T = 4

@njit
def planck(nu_hz , t_k, h_over_k_B):
    """Return occupation number at frequency nu_hz and temperature t_k"""
    return 1 / (np.exp(h_over_k_B * nu_hz / t_k)-1)

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

    # up to second order in beta, including second order in the expansion of thermodynamic temperature
    # this is in true temperature
    # no boosting induced monopoles added
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
    # up to second order in beta, including second order in the expansion of thermodynamic temperature
    # this is in linearized thermodynamic temperature
    # no boosting induced monopoles added
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    return t_cmb_k * (beta_dot_n + q_x * beta_dot_n ** 2)


@njit
def compute_dipole_for_one_sample_total_from_lin_t(
    theta, phi, v_km_s, t_cmb_k, nu_hz, f_x , planck_t0, h_over_k_B
):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    gamma = 1 / np.sqrt(1 - beta ** 2)

    planck_t = planck(nu_hz, t_cmb_k / gamma / (1 - beta_dot_n), h_over_k_B)

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
    h_over_k_B,
    dipole_type: DipoleType,
):

    if dipole_type == DipoleType.LINEAR:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_linear(
                theta=theta_det[i],
                phi=phi_det[i],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
            )
    elif dipole_type == DipoleType.QUADRATIC_EXACT:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_quadratic_exact(
                theta=theta_det[i],
                phi=phi_det[i],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
            )
    elif dipole_type == DipoleType.TOTAL_EXACT:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_total_exact(
                theta=theta_det[i],
                phi=phi_det[i],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
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
        planck_t0 = planck(nu_hz , t_cmb_k, h_over_k_B)
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_total_from_lin_t(
                theta=theta_det[i],
                phi=phi_det[i],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
                nu_hz=nu_hz,
                f_x=f_x,
                planck_t0=planck_t0,
                h_over_k_B=h_over_k_B,
            )
    else:
        print('Dipole Type not implemented!!!')


def add_dipole(
    tod,
    pointings,
    velocity,
    t_cmb_k: float,
    frequency_ghz: np.ndarray,  # e.g. central frequency of channel from
    # lbs.FreqChannelInfo.from_imo(url=f"/releases/v1.0/satellite/{telescope}/{channel}/channel_info",imo=imo).bandcenter_ghz
    dipole_type: DipoleType,
):
    """Add dipole to tod 

    """

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
            h_over_k_B=h.value / k_B.value,
            dipole_type=dipole_type,
        )


def add_dipole_to_observation(
    obs: Observation,
    pointings,
    pos_and_vel: SpacecraftPositionAndVelocity,
    t_cmb_k: float,
    dipole_type: DipoleType,
    frequency_ghz: Union[np.ndarray, None] = None,  # e.g. central frequency of channel from
):
    # Alas, this allocates memory for the velocity vector! At the moment it is the simplest implementation, but
    # in the future we might want to inline the interpolation code within "add_dipole" to save memory
    velocity = pos_and_vel.compute_velocities(
        time0=obs.start_time,
        delta_time_s=obs.get_delta_time().value,
        num_of_samples=obs.tod.shape[1],
    )

    if frequency_ghz == None:
        frequency_ghz = obs.bandcenter_ghz
    else:
        frequency_ghz = np.repeat(frequency_ghz,obs.tod.shape[0])

    add_dipole(
        tod=obs.tod,
        pointings=pointings,
        velocity=velocity,
        t_cmb_k=t_cmb_k,
        frequency_ghz=frequency_ghz,
        dipole_type=dipole_type,
    )
