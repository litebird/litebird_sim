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


################################################################
#################### SINGLE FREQUENCY ##########################
################################################################


@njit(parallel=True)
def compute_signal_for_one_detector(
    tod_det,
    j0f,
    j2f,
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
):
    """
    Single-frequency case: compute the signal for a single detector,
    looping over (time) samples.
    """
    for i in prange(len(tod_det)):
        alpha = rho[i] - phi
        deltas = np.zeros((2, 2))
        for x in range(2):
            for y in range(2):
                deltas[x, y] = np.abs(j0f[x, y]) + np.abs(j2f[x, y]) * np.cos(
                    2 * alpha + 2 * np.angle(j2f[x, y])
                )

        mII_hwp = deltas[0, 0] - deltas[1, 1] + 1
        mQI_hwp = deltas[0, 0] + deltas[0, 1]
        mUI_hwp = deltas[1, 0] - deltas[0, 1]
        mIQ_hwp = deltas[0, 0] + deltas[0, 1]
        mIU_hwp = deltas[0, 1] - deltas[1, 0]
        mQQ_hwp = deltas[0, 1] - deltas[1, 1] + 1
        mUU_hwp = deltas[1, 1] + deltas[0, 0] - 1
        mUQ_hwp = deltas[0, 1] + deltas[1, 0]
        mQU_hwp = deltas[0, 1] + deltas[1, 0]

        c2 = np.cos(2 * alpha)
        s2 = np.sin(2 * alpha)
        tod_det[i] += compute_signal_for_one_sample(
            T=mapT[i],
            Q=mapQ[i],
            U=mapU[i],
            mII=mII_hwp,
            mIQ=mIQ_hwp * c2 - mIU_hwp * s2,
            mIU=mIQ_hwp * s2 - mIU_hwp * c2,
            mQI=mQI_hwp * c2 - mUI_hwp * s2,
            mQQ=c2 * (mQQ_hwp * c2 - mUQ_hwp * s2) - s2 * (mQU_hwp * c2 - mUU_hwp * s2),
            mQU=s2 * (mQQ_hwp * c2 - mUQ_hwp * s2) + c2 * (mQU_hwp * c2 - mUU_hwp * s2),
            mUI=mQI_hwp * s2 + mUI_hwp * c2,
            mUQ=c2 * (mQQ_hwp * s2 + mUQ_hwp * c2) - s2 * (mQU_hwp * s2 + mUU_hwp * c2),
            mUU=s2 * (mQQ_hwp * s2 + mUQ_hwp * c2) + c2 * (mQU_hwp * s2 + mUU_hwp * c2),
            psi=psi[i],
            phi=phi,
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )

        if add_2f_hwpss:
            tod_det[i] += compute_2f_for_one_sample(rho[i], amplitude_2f_k)
        if apply_non_linearity:
            tod_det[i] += g_one_over_k * tod_det[i] ** 2


@njit
def compute_ata_atd_for_one_detector(
    ata,
    atd,
    tod,
    j0f_solver,
    j2f_solver,
    pixel_ind,
    rho,
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
        alpha = rho[i] - phi
        deltas = np.zeros((2, 2))
        for x in range(2):
            for y in range(2):
                deltas[x, y] = np.abs(j0f_solver[x, y]) + np.abs(
                    j2f_solver[x, y]
                ) * np.cos(2 * alpha + 2 * np.angle(j2f_solver[x, y]))

        mIIs_hwp = deltas[0, 0] - deltas[1, 1] + 1
        mQIs_hwp = deltas[0, 0] + deltas[0, 1]
        mUIs_hwp = deltas[1, 0] - deltas[0, 1]
        mIQs_hwp = deltas[0, 0] + deltas[0, 1]
        mIUs_hwp = deltas[0, 1] - deltas[1, 0]
        mQQs_hwp = deltas[0, 1] - deltas[1, 1] + 1
        mUUs_hwp = deltas[1, 1] + deltas[0, 0] - 1
        mUQs_hwp = deltas[0, 1] + deltas[1, 0]
        mQUs_hwp = deltas[0, 1] + deltas[1, 0]

        Tterm, Qterm, Uterm = compute_TQUsolver_for_one_sample(
            mIIs=mIIs_hwp,
            mIQs=mIQs_hwp * np.cos(2 * alpha) + mIUs_hwp * np.sin(2 * alpha),
            mIUs=mIUs_hwp * np.cos(2 * alpha) - mIQs_hwp * np.sin(2 * alpha),
            mQIs=mQIs_hwp * np.cos(2 * alpha) + mUIs_hwp * np.sin(2 * alpha),
            mQQs=np.cos(2 * alpha)
            * (mQQs_hwp * np.cos(2 * alpha) + mUQs_hwp * np.sin(2 * alpha))
            + np.sin(2 * alpha)
            * (mQUs_hwp * np.cos(2 * alpha) + mUUs_hwp * np.sin(2 * alpha)),
            mQUs=-(
                np.cos(2 * alpha)
                * (mQUs_hwp * np.cos(2 * alpha) + mUUs_hwp * np.sin(2 * alpha))
                - np.sin(2 * alpha)
                * (mQQs_hwp * np.cos(2 * alpha) + mUQs_hwp * np.sin(2 * alpha))
            ),
            mUIs=mUIs_hwp * np.cos(2 * alpha) - mQIs_hwp * np.sin(2 * alpha),
            mUQs=-(
                np.cos(2 * alpha)
                * (mUQs_hwp * np.cos(2 * alpha) - mQQs_hwp * np.sin(2 * alpha))
                + np.sin(2 * alpha)
                * (mUUs_hwp * np.cos(2 * alpha) - mQUs_hwp * np.sin(2 * alpha))
            ),
            mUUs=np.cos(2 * alpha)
            * (mUUs_hwp * np.cos(2 * alpha) - mQUs_hwp * np.sin(2 * alpha))
            - np.sin(2 * alpha)
            * (mUQs_hwp * np.cos(2 * alpha) - mQQs_hwp * np.sin(2 * alpha)),
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


@njit
def integrate_inband_signal_for_one_detector(
    tod_det,
    freqs,
    band,
    j0f,
    j2f,
    pixel_ind,
    rho,
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
        alpha = rho[i] - phi
        deltas = np.zeros((2, 2), dtype=np.float64)
        for x in range(2):
            for y in range(2):
                deltas[x, y] = j0f[:][x][y][0] + j2f[:][x][y][0] * np.cos(
                    2 * alpha + 2 * j2f[:][x][y][1]
                )

        delta_11, delta_12, delta_21, delta_22 = (
            deltas[:, 0, 0],
            deltas[:, 0, 1],
            deltas[:, 1, 0],
            deltas[:, 1, 1],
        )

        tod_det[i] += integrate_inband_signal_for_one_sample(
            T=maps[:, 0, pixel_ind[i]],
            Q=maps[:, 1, pixel_ind[i]],
            U=maps[:, 2, pixel_ind[i]],
            freqs=freqs,
            band=band,
            mII=delta_11 - delta_22 + 1,
            mQI=delta_11 + delta_12,
            mUI=delta_21 - delta_12,
            mIQ=delta_11 + delta_12,
            mIU=delta_12 - delta_21,
            mQQ=delta_12 - delta_22 + 1,
            mUU=delta_22 + delta_11 - 1,
            mUQ=delta_12 + delta_21,
            mQU=delta_12 + delta_21,
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )


@njit
def integrate_inband_atd_ata_for_one_detector(
    atd,
    ata,
    tod,
    freqs,
    band,
    j0f_solver,
    j2f_solver,
    pixel_ind,
    rho,
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
        alpha = rho[i] - phi
        deltas = np.zeros((2, 2), dtype=np.float64)
        for x in range(2):
            for y in range(2):
                deltas[x, y] = j0f_solver[:][x][y][0] + j2f_solver[:][x][y][0] * np.cos(
                    2 * alpha + 2 * j2f_solver[:][x][y][1]
                )

        delta_11, delta_12, delta_21, delta_22 = (
            deltas[:, 0, 0],
            deltas[:, 0, 1],
            deltas[:, 1, 0],
            deltas[:, 1, 1],
        )

        Tterm, Qterm, Uterm = integrate_inband_TQUsolver_for_one_sample(
            freqs=freqs,
            band=band,
            mIIs=delta_11 - delta_22 + 1,
            mQIs=delta_11 + delta_12,
            mUIs=delta_21 - delta_12,
            mIQs=delta_11 + delta_12,
            mIUs=delta_12 - delta_21,
            mQQs=delta_12 - delta_22 + 1,
            mUUs=delta_22 + delta_11 - 1,
            mUQs=delta_12 + delta_21,
            mQUs=delta_12 + delta_21,
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
