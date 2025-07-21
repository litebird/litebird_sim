# -*- encoding: utf-8 -*-
from numba import njit


@njit(parallel=False)
def compute_Tterm_for_one_sample(mII, mQI, mUI, cos2Xi2Phi, sin2Xi2Phi):
    Tterm = mII + mQI * cos2Xi2Phi + mUI * sin2Xi2Phi

    return Tterm


@njit(parallel=False)
def compute_Qterm_for_one_sample(
    mIQ, mQQ, mUU, mIU, mUQ, mQU, cos2Xi2Phi, sin2Xi2Phi, cos2Psi2Phi, sin2Psi2Phi
):
    Qterm = cos2Psi2Phi * (mIQ + mQQ * cos2Xi2Phi + mUQ * sin2Xi2Phi) - sin2Psi2Phi * (
        mIU + mQU * cos2Xi2Phi + mUU * sin2Xi2Phi
    )

    return Qterm


@njit(parallel=False)
def compute_Uterm_for_one_sample(
    mIU, mQU, mUQ, mIQ, mQQ, mUU, cos2Xi2Phi, sin2Xi2Phi, cos2Psi2Phi, sin2Psi2Phi
):
    Uterm = sin2Psi2Phi * (mIQ + mQQ * cos2Xi2Phi + mUQ * sin2Xi2Phi) + cos2Psi2Phi * (
        mIU + mQU * cos2Xi2Phi + mUU * sin2Xi2Phi
    )

    return Uterm


def compute_signal_for_one_sample(
    T,
    Q,
    U,
    mII,
    mQI,
    mUI,
    mIQ,
    mIU,
    mQQ,
    mUU,
    mUQ,
    mQU,
    cos2Xi2Phi,
    sin2Xi2Phi,
    cos2Psi2Phi,
    sin2Psi2Phi,
):
    # Bolometric equation, tod filling for a single (time) sample
    d = T * compute_Tterm_for_one_sample(mII, mQI, mUI, cos2Xi2Phi, sin2Xi2Phi)

    d += Q * compute_Qterm_for_one_sample(
        mIQ, mQQ, mUU, mIU, mUQ, mQU, cos2Xi2Phi, sin2Xi2Phi, cos2Psi2Phi, sin2Psi2Phi
    )

    d += U * compute_Uterm_for_one_sample(
        mIU, mQU, mUQ, mIQ, mQQ, mUU, cos2Xi2Phi, sin2Xi2Phi, cos2Psi2Phi, sin2Psi2Phi
    )

    return d


@njit(parallel=False)
def compute_TQUsolver_for_one_sample(
    mIIs,
    mQIs,
    mUIs,
    mIQs,
    mIUs,
    mQQs,
    mUUs,
    mUQs,
    mQUs,
    cos2Xi2Phi,
    sin2Xi2Phi,
    cos2Psi2Phi,
    sin2Psi2Phi,
):
    r"""
    Single-frequency case: computes :math:`A^T A` and :math:`A^T d`
    for a single detector, for one (time) sample.
    """

    Tterm = compute_Tterm_for_one_sample(mIIs, mQIs, mUIs, cos2Xi2Phi, sin2Xi2Phi)

    Qterm = compute_Qterm_for_one_sample(
        mIQs,
        mQQs,
        mUUs,
        mIUs,
        mUQs,
        mQUs,
        cos2Xi2Phi,
        sin2Xi2Phi,
        cos2Psi2Phi,
        sin2Psi2Phi,
    )

    Uterm = compute_Uterm_for_one_sample(
        mIUs,
        mQUs,
        mUQs,
        mIQs,
        mQQs,
        mUUs,
        cos2Xi2Phi,
        sin2Xi2Phi,
        cos2Psi2Phi,
        sin2Psi2Phi,
    )

    return Tterm, Qterm, Uterm


@njit(parallel=False)
def integrate_inband_signal_for_one_sample(
    T,
    Q,
    U,
    freqs,
    band,
    mII,
    mQI,
    mUI,
    mIQ,
    mIU,
    mQQ,
    mUU,
    mUQ,
    mQU,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Multi-frequency case: band integration with trapezoidal rule,
    :math:`\sum (f(i) + f(i+1)) \cdot (\nu_(i+1) - \nu_i)/2`
    for a single (time) sample.
    """
    tod = 0
    for i in range(len(band) - 1):
        dnu = freqs[i + 1] - freqs[i]
        tod += (
            (
                band[i]
                * compute_signal_for_one_sample(
                    T=T[i],
                    Q=Q[i],
                    U=U[i],
                    mII=mII[i],
                    mQI=mQI[i],
                    mUI=mUI[i],
                    mIQ=mIQ[i],
                    mIU=mIU[i],
                    mQQ=mQQ[i],
                    mUU=mUU[i],
                    mUQ=mUQ[i],
                    mQU=mQU[i],
                    cos2Xi2Phi=cos2Xi2Phi,
                    sin2Xi2Phi=sin2Xi2Phi,
                )
                + band[i + 1]
                * compute_signal_for_one_sample(
                    T=T[i + 1],
                    Q=Q[i + 1],
                    U=U[i + 1],
                    mII=mII[i + 1],
                    mQI=mQI[i + 1],
                    mUI=mUI[i + 1],
                    mIQ=mIQ[i + 1],
                    mIU=mIU[i + 1],
                    mQQ=mQQ[i + 1],
                    mUU=mUU[i + 1],
                    mUQ=mUQ[i + 1],
                    mQU=mQU[i + 1],
                    cos2Xi2Phi=cos2Xi2Phi,
                    sin2Xi2Phi=sin2Xi2Phi,
                )
            )
            * dnu
            / 2
        )

    return tod


@njit(parallel=False)
def integrate_inband_TQUsolver_for_one_sample(
    freqs,
    band,
    mIIs,
    mQIs,
    mUIs,
    mIQs,
    mIUs,
    mQQs,
    mUUs,
    mUQs,
    mQUs,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Multi-frequency case: band integration with trapezoidal rule,
    :math:`\sum (f(i) + f(i+1)) \cdot (\nu_(i+1) - \nu_i)/2`
    for a single (time) sample.
    """
    intTterm = 0
    intQterm = 0
    intUterm = 0
    for i in range(len(band) - 1):
        dnu = freqs[i + 1] - freqs[i]

        Tterm, Qterm, Uterm = compute_TQUsolver_for_one_sample(
            mIIs=mIIs[i],
            mQIs=mQIs[i],
            mUIs=mUIs[i],
            mIQs=mIQs[i],
            mIUs=mIUs[i],
            mQQs=mQQs[i],
            mUUs=mUUs[i],
            mUQs=mUQs[i],
            mQUs=mQUs[i],
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )

        Ttermp1, Qtermp1, Utermp1 = compute_TQUsolver_for_one_sample(
            mIIs=mIIs[i + 1],
            mQIs=mQIs[i + 1],
            mUIs=mUIs[i + 1],
            mIQs=mIQs[i + 1],
            mIUs=mIUs[i + 1],
            mQQs=mQQs[i + 1],
            mUUs=mUUs[i + 1],
            mUQs=mUQs[i + 1],
            mQUs=mQUs[i + 1],
            cos2Xi2Phi=cos2Xi2Phi,
            sin2Xi2Phi=sin2Xi2Phi,
        )

        intTterm += (band[i] * Tterm + band[i + 1] * Ttermp1) * dnu / 2.0
        intQterm += (band[i] * Qterm + band[i + 1] * Qtermp1) * dnu / 2.0
        intUterm += (band[i] * Uterm + band[i + 1] * Utermp1) * dnu / 2.0

    return intTterm, intQterm, intUterm
