# -*- encoding: utf-8 -*-
import numpy as np
from numba import njit, prange

from ..hwp_diff_emiss import compute_2f_for_one_sample
from ..utilities import cos_sin
from .common import compute_signal_for_one_sample


@njit(inline="always")
def JonesToMueller(j_11, j_12, j_21, j_22):
    """Mueller matrix elements (mII, mIQ, mIU, mQI, mQQ, mQU, mUI, mUQ, mUU)
    of a Jones matrix, as 0.5·Re tr(σ_a J σ_b J†)"""
    c = np.conjugate

    mII = 0.5 * np.real(
        j_11 * c(j_11) + j_12 * c(j_12) + j_21 * c(j_21) + j_22 * c(j_22)
    )
    mIQ = 0.5 * np.real(
        j_11 * c(j_11) - j_12 * c(j_12) + j_21 * c(j_21) - j_22 * c(j_22)
    )
    mIU = 0.5 * np.real(
        j_11 * c(j_12) + j_12 * c(j_11) + j_21 * c(j_22) + j_22 * c(j_21)
    )

    mQI = 0.5 * np.real(
        j_11 * c(j_11) + j_12 * c(j_12) - j_21 * c(j_21) - j_22 * c(j_22)
    )
    mQQ = 0.5 * np.real(
        j_11 * c(j_11) - j_12 * c(j_12) - j_21 * c(j_21) + j_22 * c(j_22)
    )
    mQU = 0.5 * np.real(
        j_11 * c(j_12) + j_12 * c(j_11) - j_21 * c(j_22) - j_22 * c(j_21)
    )

    mUI = 0.5 * np.real(
        j_21 * c(j_11) + j_22 * c(j_12) + j_11 * c(j_21) + j_12 * c(j_22)
    )
    mUQ = 0.5 * np.real(
        j_21 * c(j_11) - j_22 * c(j_12) + j_11 * c(j_21) - j_12 * c(j_22)
    )
    mUU = 0.5 * np.real(
        j_21 * c(j_12) + j_22 * c(j_11) + j_11 * c(j_22) + j_12 * c(j_21)
    )

    return mII, mIQ, mIU, mQI, mQQ, mQU, mUI, mUQ, mUU


@njit(inline="always")
def hwp_to_fp_frame(c2, s2, mII, mIQ, mIU, mQI, mQQ, mQU, mUI, mUQ, mUU):
    """Rotate the HWP Mueller matrix by α, with c2 = cos 2α and s2 = sin 2α"""
    fIQ = mIQ * c2 - mIU * s2
    fIU = mIQ * s2 + mIU * c2
    fQI = mQI * c2 - mUI * s2
    fQQ = (mQQ * c2 - mUQ * s2) * c2 - (mQU * c2 - mUU * s2) * s2
    fQU = (mQQ * c2 - mUQ * s2) * s2 + (mQU * c2 - mUU * s2) * c2
    fUI = mQI * s2 + mUI * c2
    fUQ = (mQQ * s2 + mUQ * c2) * c2 - (mQU * s2 + mUU * c2) * s2
    fUU = (mQQ * s2 + mUQ * c2) * s2 + (mQU * s2 + mUU * c2) * c2

    return mII, fIQ, fIU, fQI, fQQ, fQU, fUI, fUQ, fUU


@njit(inline="always")
def hwp_mueller_in_fp_frame(c2, s2, deltas_j0f, deltas_j2f):
    """Mueller matrix of the HWP in the focal-plane frame, for an HWP angle α
    (c2 = cos 2α, s2 = sin 2α); the 2f terms are |δ|·exp(i(2α + arg δ)) = δ·exp(2iα)"""
    e2a = c2 + 1j * s2
    mII, mIQ, mIU, mQI, mQQ, mQU, mUI, mUQ, mUU = JonesToMueller(
        1 + deltas_j0f[0, 0] + deltas_j2f[0, 0] * e2a,
        deltas_j0f[0, 1] + deltas_j2f[0, 1] * e2a,
        deltas_j0f[1, 0] + deltas_j2f[1, 0] * e2a,
        -1 + deltas_j0f[1, 1] + deltas_j2f[1, 1] * e2a,
    )
    return hwp_to_fp_frame(c2, s2, mII, mIQ, mIU, mQI, mQQ, mQU, mUI, mUQ, mUU)


################################################################
#################### SINGLE FREQUENCY ##########################
################################################################


@njit(parallel=True)
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
    xi,
    apply_non_linearity,
    g_one_over_k,
    add_2f_hwpss,
    amplitude_2f_k,
):
    """
    Single-frequency case: compute the signal for a single detector,
    looping over (time) samples.
    """
    for i in prange(len(tod_det)):  # type: ignore[not-iterable]
        c2, s2 = cos_sin(2 * (rho[i] - phi))
        mII, mIQ, mIU, mQI, mQQ, mQU, mUI, mUQ, mUU = hwp_mueller_in_fp_frame(
            c2, s2, deltas_j0f, deltas_j2f
        )
        cos2Psi2Phi, sin2Psi2Phi = cos_sin(2 * psi[i] + 2 * phi)

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
            cos2Psi2Phi=cos2Psi2Phi,
            sin2Psi2Phi=sin2Psi2Phi,
        )

        if add_2f_hwpss:
            tod_det[i] += compute_2f_for_one_sample(rho[i], xi, amplitude_2f_k)
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
    xi,
    cos2Xi2Phi,
    sin2Xi2Phi,
    apply_non_linearity,
    g_one_over_k,
    add_2f_hwpss,
    amplitude_2f_k,
):
    r"""
    Multi-frequency case: band integration of the signal for a single detector,
    looping over (time) samples, with the trapezoidal rule
    :math:`\sum (f(i) + f(i+1)) \cdot (\nu_(i+1) - \nu_i)/2`.
    """
    n_freqs = len(freqs)
    for i in prange(len(tod_det)):  # type: ignore[not-iterable]
        c2, s2 = cos_sin(2 * (rho[i] - phi))
        cos2Psi2Phi, sin2Psi2Phi = cos_sin(2 * psi[i] + 2 * phi)

        # Trapezoidal rule accumulated over ν, without per-sample arrays
        tod = 0.0
        prev = 0.0
        for nu in range(n_freqs):
            mII, mIQ, mIU, mQI, mQQ, mQU, mUI, mUQ, mUU = hwp_mueller_in_fp_frame(
                c2, s2, deltas_j0f[nu], deltas_j2f[nu]
            )
            signal = compute_signal_for_one_sample(
                T=mapT[nu, i],
                Q=mapQ[nu, i],
                U=mapU[nu, i],
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
                cos2Psi2Phi=cos2Psi2Phi,
                sin2Psi2Phi=sin2Psi2Phi,
            )
            if n_freqs == 1:
                tod = signal
            cur = band[nu] * signal
            if nu > 0:
                tod += (prev + cur) * ((freqs[nu] - freqs[nu - 1]) * 1e9 / 2)
            prev = cur

        tod_det[i] += tod

        if add_2f_hwpss:
            tod_det[i] += compute_2f_for_one_sample(rho[i], xi, amplitude_2f_k)
        if apply_non_linearity:
            tod_det[i] += g_one_over_k * tod_det[i] ** 2
