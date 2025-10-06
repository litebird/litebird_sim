# -*- encoding: utf-8 -*-
import numpy as np
from numba import njit, prange
from .common import (
    compute_signal_for_one_sample,
    compute_TQUsolver_for_one_sample,
    integrate_inband_TQUsolver_for_one_sample,
    integrate_inband_signal_for_one_sample,
)
from ..hwp_diff_emiss import compute_2f_for_one_sample

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""
- - - - - - METHODS FOR MUELLER TOD COMPUTATIONS - - - - - 
""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""


################################################################
#################### SINGLE FREQUENCY ##########################
################################################################


@njit(parallel=True)
def compute_signal_for_one_detector(
    tod_det,
    m0f,
    m2f,
    m4f,
    rho,
    psi,
    mapT,
    mapQ,
    mapU,
    cos2Xi2Phi,
    sin2Xi2Phi,
    phi,
    apply_non_linearity,
    g_one_over_k,
    add_2f_hwpss,
    amplitude_2f_k,
    phases_2f,
    phases_4f,
):
    """
    Single-frequency case: compute the signal for a single detector,
    looping over (time) samples.
    """

    for i in prange(len(tod_det)):
        Four_rho_phi = 4 * (rho[i] - phi)
        Two_rho_phi = 2 * (rho[i] - phi)
        tod_det[i] += compute_signal_for_one_sample(
            T=mapT[i],
            Q=mapQ[i],
            U=mapU[i],
            mII=m0f[0, 0]
            + m2f[0, 0] * np.cos(Two_rho_phi + phases_2f[0, 0])
            + m4f[0, 0] * np.cos(Four_rho_phi + phases_4f[0, 0]),
            mQI=m0f[1, 0]
            + m2f[1, 0] * np.cos(Two_rho_phi + phases_2f[1, 0])
            + m4f[1, 0] * np.cos(Four_rho_phi + phases_4f[1, 0]),
            mUI=m0f[2, 0]
            + m2f[2, 0] * np.cos(Two_rho_phi + phases_2f[2, 0])
            + m4f[2, 0] * np.cos(Four_rho_phi + phases_4f[2, 0]),
            mIQ=m0f[0, 1]
            + m2f[0, 1] * np.cos(Two_rho_phi + phases_2f[0, 1])
            + m4f[0, 1] * np.cos(Four_rho_phi + phases_4f[0, 1]),
            mIU=m0f[0, 2]
            + m2f[0, 2] * np.cos(Two_rho_phi + phases_2f[0, 2])
            + m4f[0, 2] * np.cos(Four_rho_phi + phases_4f[0, 2]),
            mQQ=m0f[1, 1]
            + m2f[1, 1] * np.cos(Two_rho_phi + phases_2f[1, 1])
            + m4f[1, 1] * np.cos(Four_rho_phi + phases_4f[1, 1]),
            mUU=m0f[2, 2]
            + m2f[2, 2] * np.cos(Two_rho_phi + phases_2f[2, 2])
            + m4f[2, 2] * np.cos(Four_rho_phi + phases_4f[2, 2]),
            mUQ=m0f[2, 1]
            + m2f[2, 1] * np.cos(Two_rho_phi + phases_2f[2, 1])
            + m4f[2, 1] * np.cos(Four_rho_phi + phases_4f[2, 1]),
            mQU=m0f[1, 2]
            + m2f[1, 2] * np.cos(Two_rho_phi + phases_2f[1, 2])
            + m4f[1, 2] * np.cos(Four_rho_phi + phases_4f[1, 2]),
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
            cos2Psi2Phi=np.cos(2 * psi[i] + 2 * phi),
            sin2Psi2Phi=np.sin(2 * psi[i] + 2 * phi),
        )

        if add_2f_hwpss:
            tod_det[i] += compute_2f_for_one_sample(rho[i], amplitude_2f_k)
        if apply_non_linearity:
            tod_det[i] += g_one_over_k * tod_det[i] ** 2


################################################################
#################### BAND INTEGRATION ##########################
################################################################


@njit
def integrate_inband_signal_for_one_detector(
    tod_det,
    freqs,
    band,
    m0f,
    m2f,
    m4f,
    pixel_ind,
    rho,
    psi,
    maps,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
    phases_2f,
    phases_4f,
):
    """
    Multi-frequency case: band integration of the signal for a single detector,
    looping over (time) samples.
    """
    for i in range(len(tod_det)):
        Four_rho_phi = 4 * (rho[i] - phi)
        Two_rho_phi = 2 * (rho[i] - phi)

        tod_det[i] += integrate_inband_signal_for_one_sample(
            T=maps[:, 0, pixel_ind[i]],
            Q=maps[:, 1, pixel_ind[i]],
            U=maps[:, 2, pixel_ind[i]],
            freqs=freqs,
            band=band,
            mII=m0f[0, 0]
            + m2f[0, 0] * np.cos(Two_rho_phi + phases_2f[0, 0])
            + m4f[0, 0] * np.cos(Four_rho_phi + phases_4f[0, 0]),
            mQI=m0f[1, 0]
            + m2f[1, 0] * np.cos(Two_rho_phi + phases_2f[1, 0])
            + m4f[1, 0] * np.cos(Four_rho_phi + phases_4f[1, 0]),
            mUI=m0f[2, 0]
            + m2f[2, 0] * np.cos(Two_rho_phi + phases_2f[2, 0])
            + m4f[2, 0] * np.cos(Four_rho_phi + phases_4f[2, 0]),
            mIQ=m0f[0, 1]
            + m2f[0, 1] * np.cos(Two_rho_phi + phases_2f[0, 1])
            + m4f[0, 1] * np.cos(Four_rho_phi + phases_4f[0, 1]),
            mIU=m0f[0, 2]
            + m2f[0, 2] * np.cos(Two_rho_phi + phases_2f[0, 2])
            + m4f[0, 2] * np.cos(Four_rho_phi + phases_4f[0, 2]),
            mQQ=m0f[1, 1]
            + m2f[1, 1] * np.cos(Two_rho_phi + phases_2f[1, 1])
            + m4f[1, 1] * np.cos(Four_rho_phi + phases_4f[1, 1]),
            mUU=m0f[2, 2]
            + m2f[2, 2] * np.cos(Two_rho_phi + phases_2f[2, 2])
            + m4f[2, 2] * np.cos(Four_rho_phi + phases_4f[2, 2]),
            mUQ=m0f[2, 1]
            + m2f[2, 1] * np.cos(Two_rho_phi + phases_2f[2, 1])
            + m4f[2, 1] * np.cos(Four_rho_phi + phases_4f[2, 1]),
            mQU=m0f[1, 2]
            + m2f[1, 2] * np.cos(Two_rho_phi + phases_2f[1, 2])
            + m4f[1, 2] * np.cos(Four_rho_phi + phases_4f[1, 2]),
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
            cos2Psi2Phi=np.cos(2 * psi[i] + 2 * phi),
            sin2Psi2Phi=np.sin(2 * psi[i] + 2 * phi),
        )