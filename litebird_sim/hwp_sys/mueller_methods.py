# -*- encoding: utf-8 -*-
import numpy as np
from numba import njit, prange
from common import (
    compute_signal_for_one_sample,
    compute_TQUsolver_for_one_sample,
    integrate_inband_TQUsolver_for_one_sample,
    integrate_inband_signal_for_one_sample,
)

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""
- - - - - - METHODS FOR MUELLER TOD COMPUTATIONS - - - - - 
""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""


################################################################
#################### SINGLE FREQUENCY ##########################
################################################################


@njit(parallel=False)
def compute_signal_for_one_detector(
    tod_det, pixel_ind, m0f, m2f, m4f, theta, psi, maps, cos2Xi2Phi, sin2Xi2Phi, phi
):
    """
    Single-frequency case: compute the signal for a single detector,
    looping over (time) samples.
    """

    for i in prange(len(tod_det)):
        FourRhoPsiPhi = 4 * (theta[i] - phi)
        TwoRhoPsiPhi = 2 * (theta[i] - phi)
        tod_det[i] += compute_signal_for_one_sample(
            T=maps[0, pixel_ind[i]],
            Q=maps[1, pixel_ind[i]],
            U=maps[2, pixel_ind[i]],
            mII=m0f[0, 0]
            + m2f[0, 0] * np.cos(TwoRhoPsiPhi - 2.32)
            + m4f[0, 0] * np.cos(FourRhoPsiPhi - 0.84),
            mQI=m0f[1, 0]
            + m2f[1, 0] * np.cos(TwoRhoPsiPhi + 2.86)
            + m4f[1, 0] * np.cos(FourRhoPsiPhi + 0.14),
            mUI=m0f[2, 0]
            + m2f[2, 0] * np.cos(TwoRhoPsiPhi + 1.29)
            + m4f[2, 0] * np.cos(FourRhoPsiPhi - 1.43),
            mIQ=m0f[0, 1]
            + m2f[0, 1] * np.cos(TwoRhoPsiPhi - 0.49)
            + m4f[0, 1] * np.cos(FourRhoPsiPhi - 0.04),
            mIU=m0f[0, 2]
            + m2f[0, 2] * np.cos(TwoRhoPsiPhi - 2.06)
            + m4f[0, 2] * np.cos(FourRhoPsiPhi - 1.61),
            mQQ=m0f[1, 1]
            + m2f[1, 1] * np.cos(TwoRhoPsiPhi - 0.25)
            + m4f[1, 1] * np.cos(FourRhoPsiPhi - 0.00061),
            mUU=m0f[2, 2]
            + m2f[2, 2] * np.cos(TwoRhoPsiPhi + 2.54)
            + m4f[2, 2] * np.cos(FourRhoPsiPhi + np.pi - 0.00065),
            mUQ=m0f[2, 1]
            + m2f[2, 1] * np.cos(TwoRhoPsiPhi - 2.01)
            + m4f[2, 1] * np.cos(FourRhoPsiPhi - 0.00070 - np.pi / 2),
            mQU=m0f[1, 2]
            + m2f[1, 2] * np.cos(TwoRhoPsiPhi - 2.00)
            + m4f[1, 2] * np.cos(FourRhoPsiPhi - 0.00056 - np.pi / 2),
            psi=psi[i],
            phi=phi,
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )


@njit(parallel=False)
def compute_ata_atd_for_one_detector(
    ata,
    atd,
    tod,
    m0f_solver,
    m2f_solver,
    m4f_solver,
    pixel_ind,
    theta,
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Single-frequency case: compute :math:`A^T A` and :math:`A^T d`
    for a single detector, looping over (time) samples.
    """

    for i in prange(len(tod)):
        FourRhoPsiPhi = 4 * (theta[i] - phi)
        TwoRhoPsiPhi = 2 * (theta[i] - phi)
        Tterm, Qterm, Uterm = compute_TQUsolver_for_one_sample(
            mIIs=m0f_solver[0, 0]
            + m2f_solver[0, 0] * np.cos(TwoRhoPsiPhi - 2.32)
            + m4f_solver[0, 0] * np.cos(FourRhoPsiPhi - 0.84),
            mQIs=m0f_solver[1, 0]
            + m2f_solver[1, 0] * np.cos(TwoRhoPsiPhi + 2.86)
            + m4f_solver[1, 0] * np.cos(FourRhoPsiPhi + 0.14),
            mUIs=m0f_solver[2, 0]
            + m2f_solver[2, 0] * np.cos(TwoRhoPsiPhi + 1.29)
            + m4f_solver[2, 0] * np.cos(FourRhoPsiPhi - 1.43),
            mIQs=m0f_solver[0, 1]
            + m2f_solver[0, 1] * np.cos(TwoRhoPsiPhi - 0.49)
            + m4f_solver[0, 1] * np.cos(FourRhoPsiPhi - 0.04),
            mIUs=m0f_solver[0, 2]
            + m2f_solver[0, 2] * np.cos(TwoRhoPsiPhi - 2.06)
            + m4f_solver[0, 2] * np.cos(FourRhoPsiPhi - 1.61),
            mQQs=m0f_solver[1, 1]
            + m2f_solver[1, 1] * np.cos(TwoRhoPsiPhi - 0.25)
            + m4f_solver[1, 1] * np.cos(FourRhoPsiPhi - 0.00061),
            mUUs=m0f_solver[2, 2]
            + m2f_solver[2, 2] * np.cos(TwoRhoPsiPhi + 2.54)
            + m4f_solver[2, 2] * np.cos(FourRhoPsiPhi + np.pi - 0.00065),
            mUQs=m0f_solver[2, 1]
            + m2f_solver[2, 1] * np.cos(TwoRhoPsiPhi - 2.01)
            + m4f_solver[2, 1] * np.cos(FourRhoPsiPhi - 0.00070 - np.pi / 2),
            mQUs=m0f_solver[1, 2]
            + m2f_solver[1, 2] * np.cos(TwoRhoPsiPhi - 2.00)
            + m4f_solver[1, 2] * np.cos(FourRhoPsiPhi - 0.00056 - np.pi / 2),
            psi=psi[i],
            phi=phi,
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )

        atd[pixel_ind[i], 0] += tod[i] * Tterm
        atd[pixel_ind[i], 1] += tod[i] * Qterm
        atd[pixel_ind[i], 2] += tod[i] * Uterm

        ata[pixel_ind[i], 0, 0] += Tterm * Tterm
        ata[pixel_ind[i], 1, 0] += Tterm * Qterm
        ata[pixel_ind[i], 2, 0] += Tterm * Uterm
        ata[pixel_ind[i], 1, 1] += Qterm * Qterm
        ata[pixel_ind[i], 2, 1] += Qterm * Uterm
        ata[pixel_ind[i], 2, 2] += Uterm * Uterm


################################################################
#################### BAND INTEGRATION ##########################
################################################################


@njit(parallel=False)
def integrate_inband_signal_for_one_detector(
    tod_det,
    freqs,
    band,
    m0f,
    m2f,
    m4f,
    pixel_ind,
    theta,
    psi,
    maps,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    """
    Multi-frequency case: band integration of the signal for a single detector,
    looping over (time) samples.
    """
    for i in range(len(tod_det)):
        FourRhoPsiPhi = 4 * (theta[i] - phi)
        TwoRhoPsiPhi = 2 * (theta[i] - phi)

        tod_det[i] += integrate_inband_signal_for_one_sample(
            T=maps[:, 0, pixel_ind[i]],
            Q=maps[:, 1, pixel_ind[i]],
            U=maps[:, 2, pixel_ind[i]],
            freqs=freqs,
            band=band,
            mII=m0f[:, 0, 0]
            + m2f[:, 0, 0] * np.cos(TwoRhoPsiPhi - 2.32)
            + m4f[:, 0, 0] * np.cos(FourRhoPsiPhi - 0.84),
            mQI=m0f[:, 1, 0]
            + m2f[:, 1, 0] * np.cos(TwoRhoPsiPhi + 2.86)
            + m4f[:, 1, 0] * np.cos(FourRhoPsiPhi + 0.14),
            mUI=m0f[:, 2, 0]
            + m2f[:, 2, 0] * np.cos(TwoRhoPsiPhi + 1.29)
            + m4f[:, 2, 0] * np.cos(FourRhoPsiPhi - 1.43),
            mIQ=m0f[:, 0, 1]
            + m2f[:, 0, 1] * np.cos(TwoRhoPsiPhi - 0.49)
            + m4f[:, 0, 1] * np.cos(FourRhoPsiPhi - 0.04),
            mIU=m0f[:, 0, 2]
            + m2f[:, 0, 2] * np.cos(TwoRhoPsiPhi - 2.06)
            + m4f[:, 0, 2] * np.cos(FourRhoPsiPhi - 1.61),
            mQQ=m0f[:, 1, 1]
            + m2f[:, 1, 1] * np.cos(TwoRhoPsiPhi - 0.25)
            + m4f[:, 1, 1] * np.cos(FourRhoPsiPhi - 0.00061),
            mUU=m0f[:, 2, 2]
            + m2f[:, 2, 2] * np.cos(TwoRhoPsiPhi + 2.54)
            + m4f[:, 2, 2] * np.cos(FourRhoPsiPhi + np.pi - 0.00065),
            mUQ=m0f[:, 2, 1]
            + m2f[:, 2, 1] * np.cos(TwoRhoPsiPhi - 2.01)
            + m4f[:, 2, 1] * np.cos(FourRhoPsiPhi - 0.00070 - np.pi / 2),
            mQU=m0f[:, 1, 2]
            + m2f[:, 1, 2] * np.cos(TwoRhoPsiPhi - 2.00)
            + m4f[:, 1, 2] * np.cos(FourRhoPsiPhi - 0.00056 - np.pi / 2),
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )


@njit(parallel=False)
def integrate_inband_atd_ata_for_one_detector(
    atd,
    ata,
    tod,
    freqs,
    band,
    m0f_solver,
    m2f_solver,
    m4f_solver,
    pixel_ind,
    theta,
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Multi-frequency case: band integration of :math:`A^T A` and :math:`A^T d`
    for a single detector, looping over (time) samples.
    """
    for i in range(len(tod)):
        FourRhoPsiPhi = 4 * (theta[i] - phi)
        TwoRhoPsiPhi = 2 * (theta[i] - phi)
        Tterm, Qterm, Uterm = integrate_inband_TQUsolver_for_one_sample(
            freqs=freqs,
            band=band,
            mIIs=m0f_solver[:, 0, 0]
            + m2f_solver[:, 0, 0] * np.cos(TwoRhoPsiPhi - 2.32)
            + m4f_solver[:, 0, 0] * np.cos(FourRhoPsiPhi - 0.84),
            mQIs=m0f_solver[:, 1, 0]
            + m2f_solver[:, 1, 0] * np.cos(TwoRhoPsiPhi + 2.86)
            + m4f_solver[:, 1, 0] * np.cos(FourRhoPsiPhi + 0.14),
            mUIs=m0f_solver[:, 2, 0]
            + m2f_solver[:, 2, 0] * np.cos(TwoRhoPsiPhi + 1.29)
            + m4f_solver[:, 2, 0] * np.cos(FourRhoPsiPhi - 1.43),
            mIQs=m0f_solver[:, 0, 1]
            + m2f_solver[:, 0, 1] * np.cos(TwoRhoPsiPhi - 0.49)
            + m4f_solver[:, 0, 1] * np.cos(FourRhoPsiPhi - 0.04),
            mIUs=m0f_solver[:, 0, 2]
            + m2f_solver[:, 0, 2] * np.cos(TwoRhoPsiPhi - 2.06)
            + m4f_solver[:, 0, 2] * np.cos(FourRhoPsiPhi - 1.61),
            mQQs=m0f_solver[:, 1, 1]
            + m2f_solver[:, 1, 1] * np.cos(TwoRhoPsiPhi - 0.25)
            + m4f_solver[:, 1, 1] * np.cos(FourRhoPsiPhi - 0.00061),
            mUUs=m0f_solver[:, 2, 2]
            + m2f_solver[:, 2, 2] * np.cos(TwoRhoPsiPhi + 2.54)
            + m4f_solver[:, 2, 2] * np.cos(FourRhoPsiPhi + np.pi - 0.00065),
            mUQs=m0f_solver[:, 2, 1]
            + m2f_solver[:, 2, 1] * np.cos(TwoRhoPsiPhi - 2.01)
            + m4f_solver[:, 2, 1] * np.cos(FourRhoPsiPhi - 0.00070 - np.pi / 2),
            mQUs=m0f_solver[:, 1, 2]
            + m2f_solver[:, 1, 2] * np.cos(TwoRhoPsiPhi - 2.00)
            + m4f_solver[:, 1, 2] * np.cos(FourRhoPsiPhi - 0.00056 - np.pi / 2),
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )
        atd[pixel_ind[i], 0] += tod[i] * Tterm
        atd[pixel_ind[i], 1] += tod[i] * Qterm
        atd[pixel_ind[i], 2] += tod[i] * Uterm

        ata[pixel_ind[i], 0, 0] += Tterm * Tterm
        ata[pixel_ind[i], 1, 0] += Tterm * Qterm
        ata[pixel_ind[i], 2, 0] += Tterm * Uterm
        ata[pixel_ind[i], 1, 1] += Qterm * Qterm
        ata[pixel_ind[i], 2, 1] += Qterm * Uterm
        ata[pixel_ind[i], 2, 2] += Uterm * Uterm
