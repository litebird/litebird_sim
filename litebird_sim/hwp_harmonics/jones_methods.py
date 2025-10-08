# -*- encoding: utf-8 -*-
import numpy as np
from numba import njit, prange
from .common import (
    compute_signal_for_one_sample,
    integrate_inband_signal_for_one_sample,
)
from ..hwp_diff_emiss import compute_2f_for_one_sample


@njit(inline="always")
def JonesToMueller(jones):
    j_11 = jones[0, 0]
    j_12 = jones[0, 1]
    j_21 = jones[1, 0]
    j_22 = jones[1, 1]

    mII = np.real(
        0.5
        * np.trace(
            np.array(
                [
                    [
                        j_11 * np.conjugate(j_11) + j_12 * np.conjugate(j_12),
                        j_11 * np.conjugate(j_21) + j_12 * np.conjugate(j_22),
                    ],
                    [
                        j_21 * np.conjugate(j_11) + j_22 * np.conjugate(j_12),
                        j_21 * np.conjugate(j_21) + j_22 * np.conjugate(j_22),
                    ],
                ]
            )
        )
    )
    mIQ = np.real(
        0.5
        * np.trace(
            np.array(
                [
                    [
                        j_11 * np.conjugate(j_11) - j_12 * np.conjugate(j_12),
                        j_11 * np.conjugate(j_21) - j_12 * np.conjugate(j_22),
                    ],
                    [
                        j_21 * np.conjugate(j_11) - j_22 * np.conjugate(j_12),
                        j_21 * np.conjugate(j_21) - j_22 * np.conjugate(j_22),
                    ],
                ]
            )
        )
    )
    mIU = np.real(
        0.5
        * np.trace(
            np.array(
                [
                    [
                        j_11 * np.conjugate(j_12) + j_12 * np.conjugate(j_11),
                        j_11 * np.conjugate(j_22) + j_12 * np.conjugate(j_21),
                    ],
                    [
                        j_21 * np.conjugate(j_12) + j_22 * np.conjugate(j_11),
                        j_21 * np.conjugate(j_22) + j_22 * np.conjugate(j_21),
                    ],
                ]
            )
        )
    )

    mQI = np.real(
        0.5
        * np.trace(
            np.array(
                [
                    [
                        j_11 * np.conjugate(j_11) + j_12 * np.conjugate(j_12),
                        j_11 * np.conjugate(j_21) + j_12 * np.conjugate(j_22),
                    ],
                    [
                        -j_21 * np.conjugate(j_11) - j_22 * np.conjugate(j_12),
                        -j_21 * np.conjugate(j_21) - j_22 * np.conjugate(j_22),
                    ],
                ]
            )
        )
    )
    mQQ = np.real(
        0.5
        * np.trace(
            np.array(
                [
                    [
                        j_11 * np.conjugate(j_11) - j_12 * np.conjugate(j_12),
                        j_11 * np.conjugate(j_21) - j_12 * np.conjugate(j_22),
                    ],
                    [
                        -j_21 * np.conjugate(j_11) + j_22 * np.conjugate(j_12),
                        -j_21 * np.conjugate(j_21) + j_22 * np.conjugate(j_22),
                    ],
                ]
            )
        )
    )
    mQU = np.real(
        0.5
        * np.trace(
            np.array(
                [
                    [
                        j_11 * np.conjugate(j_12) + j_12 * np.conjugate(j_11),
                        j_11 * np.conjugate(j_22) + j_12 * np.conjugate(j_21),
                    ],
                    [
                        -j_21 * np.conjugate(j_12) - j_22 * np.conjugate(j_11),
                        -j_21 * np.conjugate(j_22) - j_22 * np.conjugate(j_21),
                    ],
                ]
            )
        )
    )

    mUI = np.real(
        0.5
        * np.trace(
            np.array(
                [
                    [
                        j_21 * np.conjugate(j_11) + j_22 * np.conjugate(j_12),
                        j_21 * np.conjugate(j_21) + j_22 * np.conjugate(j_22),
                    ],
                    [
                        j_11 * np.conjugate(j_11) + j_12 * np.conjugate(j_12),
                        j_11 * np.conjugate(j_21) + j_12 * np.conjugate(j_22),
                    ],
                ]
            )
        )
    )
    mUQ = np.real(
        0.5
        * np.trace(
            np.array(
                [
                    [
                        j_21 * np.conjugate(j_11) - j_22 * np.conjugate(j_12),
                        j_21 * np.conjugate(j_21) - j_22 * np.conjugate(j_22),
                    ],
                    [
                        j_11 * np.conjugate(j_11) - j_12 * np.conjugate(j_12),
                        j_11 * np.conjugate(j_21) - j_12 * np.conjugate(j_22),
                    ],
                ]
            )
        )
    )
    mUU = np.real(
        0.5
        * np.trace(
            np.array(
                [
                    [
                        j_21 * np.conjugate(j_12) + j_22 * np.conjugate(j_11),
                        j_21 * np.conjugate(j_22) + j_22 * np.conjugate(j_21),
                    ],
                    [
                        j_11 * np.conjugate(j_12) + j_12 * np.conjugate(j_11),
                        j_11 * np.conjugate(j_22) + j_12 * np.conjugate(j_21),
                    ],
                ]
            )
        )
    )

    return np.array(([mII, mIQ, mIU], [mQI, mQQ, mQU], [mUI, mUQ, mUU]))


@njit(inline="always")
def hwp_to_fp_frame(alpha, mueller_hwp):
    c2 = np.cos(2 * alpha)
    s2 = np.sin(2 * alpha)

    mII = mueller_hwp[0, 0]
    mIQ = mueller_hwp[0, 1] * c2 - mueller_hwp[0, 2] * s2
    mIU = mueller_hwp[0, 1] * s2 + mueller_hwp[0, 2] * c2
    mQI = mueller_hwp[1, 0] * c2 - mueller_hwp[2, 0] * s2
    mQQ = (mueller_hwp[1, 1] * c2 - mueller_hwp[2, 1] * s2) * c2 - (
        mueller_hwp[1, 2] * c2 - mueller_hwp[2, 2] * s2
    ) * s2
    mQU = (mueller_hwp[1, 1] * c2 - mueller_hwp[2, 1] * s2) * s2 + (
        mueller_hwp[1, 2] * c2 - mueller_hwp[2, 2] * s2
    ) * c2
    mUI = mueller_hwp[1, 0] * s2 + mueller_hwp[2, 0] * c2
    mUQ = (mueller_hwp[1, 1] * s2 + mueller_hwp[2, 1] * c2) * c2 - (
        mueller_hwp[1, 2] * s2 + mueller_hwp[2, 2] * c2
    ) * s2
    mUU = (mueller_hwp[1, 1] * s2 + mueller_hwp[2, 1] * c2) * s2 + (
        mueller_hwp[1, 2] * s2 + mueller_hwp[2, 2] * c2
    ) * c2

    return np.array([mII, mIQ, mIU, mQI, mQQ, mQU, mUI, mUQ, mUU], dtype=np.float64)


################################################################
#################### SINGLE FREQUENCY ##########################
################################################################


@njit
def compute_signal_for_one_detector(
    tod_det,
    deltas_j0f,
    deltas_j2f,
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
        for x in prange(2):
            for y in prange(2):
                deltas[x, y] = np.abs(deltas_j0f[x, y]) + np.abs(
                    deltas_j2f[x, y]
                ) * np.cos(2 * alpha + 2 * np.angle(deltas_j2f[x, y]))

        jones = np.array(
            [[1 - deltas[0, 0], deltas[0, 1]], [deltas[1, 0], -1 + deltas[1, 1]]],
            dtype=np.complex64,
        )

        mueller_hwp = JonesToMueller(jones)

        mII, mIQ, mIU, mQI, mQQ, mQU, mUI, mUQ, mUU = hwp_to_fp_frame(
            alpha, mueller_hwp
        )

        tod_det[i] += compute_signal_for_one_sample(
            T=mapT[i],
            Q=mapQ[i],
            U=mapU[i],
            mII=mII,
            mIQ=mIQ,
            mIU=mIU,
            mQI=mQI,
            mQQ=mQQ,
            mQU=mQU,
            mUI=mUI,
            mUQ=mUQ,
            mUU=mUU,
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


@njit(parallel=True)
def integrate_inband_signal_for_one_detector(
    tod_det,
    freqs,
    band,
    deltas_j0f,
    deltas_j2f,
    mapT,
    mapQ,
    mapU,
    rho,
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
    apply_non_linearity,
    g_one_over_k,
    add_2f_hwpss,
    amplitude_2f_k,
):
    """
    Multi-frequency case: band integration of the signal for a single detector,
    looping over (time) samples.
    """
    n_freqs = len(freqs)

    # Allocate buffers outside the loop

    for i in prange(len(tod_det)):
        alpha = rho[i] - phi

        deltas = np.empty((n_freqs, 2, 2), dtype=np.float64)
        jones_nu = np.empty((2, 2), dtype=np.complex64)

        mII = np.empty(n_freqs, dtype=np.float64)
        mIQ = np.empty(n_freqs, dtype=np.float64)
        mIU = np.empty(n_freqs, dtype=np.float64)
        mQI = np.empty(n_freqs, dtype=np.float64)
        mQQ = np.empty(n_freqs, dtype=np.float64)
        mQU = np.empty(n_freqs, dtype=np.float64)
        mUI = np.empty(n_freqs, dtype=np.float64)
        mUQ = np.empty(n_freqs, dtype=np.float64)
        mUU = np.empty(n_freqs, dtype=np.float64)
        # Compute deltas without allocating
        for x in range(2):
            for y in range(2):
                for nu in range(n_freqs):
                    delta_j0 = np.abs(deltas_j0f[nu, x, y])
                    delta_j2 = np.abs(deltas_j2f[nu, x, y])
                    angle_j2 = np.angle(deltas_j2f[nu, x, y])
                    deltas[nu, x, y] = delta_j0 + delta_j2 * np.cos(
                        2 * alpha + 2 * angle_j2
                    )

        for nu in range(n_freqs):
            jones_nu[0, 0] = 1 - deltas[nu, 0, 0]
            jones_nu[0, 1] = deltas[nu, 0, 1]
            jones_nu[1, 0] = deltas[nu, 1, 0]
            jones_nu[1, 1] = -1 + deltas[nu, 1, 1]

            mueller_hwp_nu = JonesToMueller(jones_nu)
            (
                mII[nu],
                mIQ[nu],
                mIU[nu],
                mQI[nu],
                mQQ[nu],
                mQU[nu],
                mUI[nu],
                mUQ[nu],
                mUU[nu],
            ) = hwp_to_fp_frame(alpha, mueller_hwp_nu)

        tod_det[i] += integrate_inband_signal_for_one_sample(
            T=mapT[i],
            Q=mapQ[i],
            U=mapU[i],
            freqs=freqs,
            band=band,
            mII=mII,
            mQI=mQI,
            mUI=mUI,
            mIQ=mIQ,
            mIU=mIU,
            mQQ=mQQ,
            mUU=mUU,
            mUQ=mUQ,
            mQU=mQU,
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
            cos2Psi2Phi=np.cos(2 * psi[i] + 2 * phi),
            sin2Psi2Phi=np.sin(2 * psi[i] + 2 * phi),
        )

    if add_2f_hwpss:
        tod_det[i] += compute_2f_for_one_sample(rho[i], amplitude_2f_k)
    if apply_non_linearity:
        tod_det[i] += g_one_over_k * tod_det[i] ** 2

    """
    for i in prange(len(tod_det)):
        alpha = rho[i] - phi
        deltas = np.zeros((len(freqs),2, 2), dtype=np.float64)
        for x in prange(2):
            for y in prange(2):
                deltas[:, x, y] = np.abs(deltas_j0f[:, x, y]) + np.abs(deltas_j2f[:, x, y]) * np.cos(
                    2 * alpha + 2 * np.angle(deltas_j2f[:, x, y])
                )
        
        #jones = np.zeros((len(freqs),2,2))
        #mII=mIQ=mIU=mQI=mQQ=mQU=mUI=mUQ=mUU= np.zeros((len(freqs)))
        #for nu in range(len(freqs)):
        #    jones[nu] = np.array(
        #        [[1 - deltas[nu, 0, 0], deltas[nu, 0, 1]], [deltas[nu, 1, 0], -1 + deltas[nu, 1, 1]]],
        #        dtype=np.complex64,
        #    )
        #
        #
        #
        #for nu in range(len(freqs)):
        #    mueller_hwp_nu = JonesToMueller(jones[nu])
        #    mII[nu], mIQ[nu], mIU[nu], mQI[nu], mQQ[nu], mQU[nu], mUI[nu], mUQ[nu], mUU[nu] = hwp_to_fp_frame(
        #        alpha, mueller_hwp_nu
        #    )
        

        mII=mIQ=mIU=mQI=mQQ=mQU=mUI=mUQ=mUU= np.zeros((len(freqs)))
        for nu in prange(len(freqs)):
            jones_nu = np.array(
                [[1 - deltas[nu, 0, 0], deltas[nu, 0, 1]], [deltas[nu, 1, 0], -1 + deltas[nu, 1, 1]]],
                dtype=np.complex64,
            )
            
            #start=time.time()
            mueller_hwp_nu = JonesToMueller(jones_nu)
            #print("jonestomueller",time.time()-start)

            #start=time.time()
            mII[nu], mIQ[nu], mIU[nu], mQI[nu], mQQ[nu], mQU[nu], mUI[nu], mUQ[nu], mUU[nu] = hwp_to_fp_frame(
                alpha, mueller_hwp_nu
            )
            #print("hwptofp",time.time()-start)



        tod_det[i] += integrate_inband_signal_for_one_sample(
            T=mapT[i],
            Q=mapQ[i],
            U=mapU[i],
            freqs=freqs,
            band=band,
            mII=mII,
            mQI=mQI,
            mUI=mUI,
            mIQ=mIQ,
            mIU=mIU,
            mQQ=mQQ,
            mUU=mUU,
            mUQ=mUQ,
            mQU=mQU,
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
            cos2Psi2Phi=np.cos(2 * psi[i] + 2 * phi),
            sin2Psi2Phi=np.sin(2 * psi[i] + 2 * phi),
        )

        if add_2f_hwpss:
            tod_det[i] += compute_2f_for_one_sample(rho[i], amplitude_2f_k)
        if apply_non_linearity:
            tod_det[i] += g_one_over_k * tod_det[i] ** 2

    """
