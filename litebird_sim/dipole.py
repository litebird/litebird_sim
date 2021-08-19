# -*- encoding: utf-8 -*-

from enum import IntEnum

from numba import njit
import numpy as np
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
    # Up to second order in β, using the linear temperature approximation (linearization of thermodynamic temperature)
    QUADRATIC_FROM_LIN_T = 1

    # Up to second order in β, including second order in the expansion of the thermodynamic temperature
    QUADRATIC_EXACT = 2

    # Total contribution (the slowest but more accurate formula)
    TOTAL = 3


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
    return t_cmb_k * (1 + beta_dot_n)


@njit
def compute_dipole_for_one_sample_quadratic_from_lin_t(
    theta, phi, v_km_s, t_cmb_k, q_x
):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    return t_cmb_k * (1 + beta_dot_n + q_x * beta_dot_n ** 2 - 0.5 * beta ** 2)


@njit
def compute_dipole_for_one_sample_quadratic_exact(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)

    # up to second order in beta, including second order in the expansion of thermodynamic temperature
    return t_cmb_k * (1 + beta_dot_n + beta_dot_n ** 2 - 0.5 * beta ** 2)


@njit
def compute_dipole_for_one_sample_total(theta, phi, v_km_s, t_cmb_k):
    beta_dot_n, beta = calculate_beta(theta, phi, v_km_s)
    gamma = 1 / np.sqrt(1 - beta ** 2)

    return t_cmb_k / gamma / (1 - beta_dot_n)


@njit
def add_dipole_for_one_detector(
    tod_det,
    theta_det,
    phi_det,
    velocity,
    t_cmb_k,
    q_x,
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
    elif dipole_type == DipoleType.QUADRATIC_FROM_LIN_T:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_quadratic_from_lin_t(
                theta=theta_det[i],
                phi=phi_det[i],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
                q_x=q_x,
            )
    elif dipole_type == DipoleType.QUADRATIC_EXACT:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_quadratic_exact(
                theta=theta_det[i],
                phi=phi_det[i],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
            )
    else:
        for i in range(len(tod_det)):
            tod_det[i] += compute_dipole_for_one_sample_total(
                theta=theta_det[i],
                phi=phi_det[i],
                v_km_s=velocity[i],
                t_cmb_k=t_cmb_k,
            )


def add_dipole(
    tod,
    pointings,
    velocity,
    t_cmb_k: float,
    frequency_ghz: float,  # e.g. central frequency of channel from
    # lbs.FreqChannelInfo.from_imo(url=f"/releases/v1.0/satellite/{telescope}/{channel}/channel_info",imo=imo).bandcenter_ghz
    dipole_type: DipoleType,
):
    nu_hz = frequency_ghz * 1e6  # freq in GHz
    # Note that x is a dimensionless parameter
    x = h.value * nu_hz / (k_B.value * t_cmb_k)

    q_x = 0.5 * x * (np.exp(x) + 1) / (np.exp(x) - 1)

    assert tod.shape == pointings.shape[0:2]
    assert tod.shape[1] == velocity.shape[0]

    for detector_idx in range(tod.shape[0]):
        add_dipole_for_one_detector(
            tod_det=tod[detector_idx],
            theta_det=pointings[detector_idx, :, 0],
            phi_det=pointings[detector_idx, :, 1],
            velocity=velocity,
            t_cmb_k=t_cmb_k,
            q_x=q_x,
            dipole_type=dipole_type,
        )


def add_dipole_to_observation(
    obs: Observation,
    pointings,
    pos_and_vel: SpacecraftPositionAndVelocity,
    t_cmb_k: float,
    frequency_ghz: float,  # e.g. central frequency of channel from
    dipole_type: DipoleType,
):
    # Alas, this allocates memory for the velocity vector! At the moment it is the simplest implementation, but
    # in the future we might want to inline the interpolation code within "add_dipole" to save memory
    velocity = pos_and_vel.compute_velocities(
        time0=obs.start_time,
        delta_time_s=obs.get_delta_time(),
        num_of_samples=obs.tod.shape[1],
    )

    add_dipole(
        tod=obs.tod,
        pointings=pointings,
        velocity=velocity,
        t_cmb_k=t_cmb_k,
        frequency_ghz=frequency_ghz,
        dipole_type=dipole_type,
    )
