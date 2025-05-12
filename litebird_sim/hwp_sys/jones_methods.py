# -*- encoding: utf-8 -*-
import numpy as np
from numba import njit, prange
from common import (
    compute_signal_for_one_sample,
    compute_TQUsolver_for_one_sample,
    integrate_inband_TQUsolver_for_one_sample,
    integrate_inband_signal_for_one_sample,
)


################################################################
#################### SINGLE FREQUENCY ##########################
################################################################


@njit(parallel=False)
def compute_signal_for_one_detector(
    tod_det, pixel_ind, j0f, j2f, j4f, theta, psi, maps, cos2Xi2Phi, sin2Xi2Phi, phi
):
    """
    Single-frequency case: compute the signal for a single detector,
    looping over (time) samples.
    """
    for i in prange(len(tod_det)):
        alpha = theta[i] - psi[i] - phi
        deltas = np.zeros((2, 2), dtype=np.float64)
        for x in range(2):
            for y in range(2):
                deltas[x, y] = (
                    j0f[x][y][0]
                    + j2f[x][y][0] * np.cos(2 * alpha + 2 * j2f[x][y][1])
                    + j4f[x][y][0] * np.cos(4 * alpha + 4 * j4f[x][y][1])
                )

        delta_11, delta_12, delta_21, delta_22 = (
            deltas[0, 0],
            deltas[0, 1],
            deltas[1, 0],
            deltas[1, 1],
        )

        tod_det[i] += compute_signal_for_one_sample(
            T=maps[0, pixel_ind[i]],
            Q=maps[1, pixel_ind[i]],
            U=maps[2, pixel_ind[i]],
            mII=delta_11 - delta_22 + 1,
            mQI=delta_11 + delta_12,
            mUI=delta_21 - delta_12,
            mIQ=delta_11 + delta_12,
            mIU=delta_12 - delta_21,
            mQQ=delta_12 - delta_22 + 1,
            mUU=delta_22 + delta_11 - 1,
            mUQ=delta_12 + delta_21,
            mQU=delta_12 + delta_21,
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
    j0f_solver,
    j2f_solver,
    j4f_solver,
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
        alpha = theta[i] - psi[i] - phi
        deltas = np.zeros((2, 2), dtype=np.float64)
        for x in range(2):
            for y in range(2):
                deltas[x, y] = (
                    j0f_solver[x][y][0]
                    + j2f_solver[x][y][0] * np.cos(2 * alpha + 2 * j2f_solver[x][y][1])
                    + j4f_solver[x][y][0] * np.cos(4 * alpha + 4 * j4f_solver[x][y][1])
                )

        delta_11, delta_12, delta_21, delta_22 = (
            deltas[0, 0],
            deltas[0, 1],
            deltas[1, 0],
            deltas[1, 1],
        )

        Tterm, Qterm, Uterm = compute_TQUsolver_for_one_sample(
            mIIs=delta_11 - delta_22 + 1,
            mQIs=delta_11 + delta_12,
            mUIs=delta_21 - delta_12,
            mIQs=delta_11 + delta_12,
            mIUs=delta_12 - delta_21,
            mQQs=delta_12 - delta_22 + 1,
            mUUs=delta_22 + delta_11 - 1,
            mUQs=delta_12 + delta_21,
            mQUs=delta_12 + delta_21,
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
    j0f,
    j2f,
    j4f,
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
        alpha = theta[i] - psi[i] - phi
        deltas = np.zeros((2, 2), dtype=np.float64)
        for x in range(2):
            for y in range(2):
                deltas[x, y] = (
                    j0f[:][x][y][0]
                    + j2f[:][x][y][0] * np.cos(2 * alpha + 2 * j2f[:][x][y][1])
                    + j4f[:][x][y][0] * np.cos(4 * alpha + 4 * j4f[:][x][y][1])
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


@njit(parallel=False)
def integrate_inband_atd_ata_for_one_detector(
    atd,
    ata,
    tod,
    freqs,
    band,
    j0f_solver,
    j2f_solver,
    j4f_solver,
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
        alpha = theta[i] - psi[i] - phi
        deltas = np.zeros((2, 2), dtype=np.float64)
        for x in range(2):
            for y in range(2):
                deltas[x, y] = (
                    j0f_solver[:][x][y][0]
                    + j2f_solver[:][x][y][0]
                    * np.cos(2 * alpha + 2 * j2f_solver[:][x][y][1])
                    + j4f_solver[:][x][y][0]
                    * np.cos(4 * alpha + 4 * j4f_solver[:][x][y][1])
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
