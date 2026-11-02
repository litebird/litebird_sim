# -*- encoding: utf-8 -*-
import numpy as np
from numba import njit, prange

from ..hwp_diff_emiss import compute_2f_for_one_sample
from ..utilities import cos_sin
from .common import (
    compute_signal_for_one_sample,
)

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""
- - - - - - METHODS FOR MUELLER TOD COMPUTATIONS - - - - -
""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""


################################################################
#################### SINGLE FREQUENCY ##########################
################################################################


@njit(inline="always")
def _mueller_element(coeffs, row, col, c2, s2, c4, s4):
    return (
        coeffs[0, row, col]
        + coeffs[1, row, col] * c2
        + coeffs[2, row, col] * s2
        + coeffs[3, row, col] * c4
        + coeffs[4, row, col] * s4
    )


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
    xi,
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

    # m2f·cos(2α + p) = m2f·cos p·cos 2α - m2f·sin p·sin 2α (same for 4f), so
    # that only cos/sin of 2α are computed per sample
    coeffs = np.stack(
        (
            m0f,
            m2f * np.cos(phases_2f),
            -m2f * np.sin(phases_2f),
            m4f * np.cos(phases_4f),
            -m4f * np.sin(phases_4f),
        )
    )

    for i in prange(len(tod_det)):  # type: ignore[not-iterable]
        c2, s2 = cos_sin(2 * (rho[i] - phi))
        c4 = c2 * c2 - s2 * s2
        s4 = 2 * c2 * s2
        cos2Psi2Phi, sin2Psi2Phi = cos_sin(2 * psi[i] + 2 * phi)

        tod_det[i] += compute_signal_for_one_sample(
            T=mapT[i],
            Q=mapQ[i],
            U=mapU[i],
            mII=_mueller_element(coeffs, 0, 0, c2, s2, c4, s4),
            mQI=_mueller_element(coeffs, 1, 0, c2, s2, c4, s4),
            mUI=_mueller_element(coeffs, 2, 0, c2, s2, c4, s4),
            mIQ=_mueller_element(coeffs, 0, 1, c2, s2, c4, s4),
            mIU=_mueller_element(coeffs, 0, 2, c2, s2, c4, s4),
            mQQ=_mueller_element(coeffs, 1, 1, c2, s2, c4, s4),
            mUU=_mueller_element(coeffs, 2, 2, c2, s2, c4, s4),
            mUQ=_mueller_element(coeffs, 2, 1, c2, s2, c4, s4),
            mQU=_mueller_element(coeffs, 1, 2, c2, s2, c4, s4),
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
            cos2Psi2Phi=cos2Psi2Phi,
            sin2Psi2Phi=sin2Psi2Phi,
        )

        if add_2f_hwpss:
            tod_det[i] += compute_2f_for_one_sample(rho[i], xi, amplitude_2f_k)
        if apply_non_linearity:
            tod_det[i] += g_one_over_k * tod_det[i] ** 2
