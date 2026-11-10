# -*- encoding: utf-8 -*-
from numba import njit


@njit
def compute_Tterm_for_one_sample(mII, mQI, mUI, cos2Xi2Phi, sin2Xi2Phi):
    Tterm = mII + mQI * cos2Xi2Phi + mUI * sin2Xi2Phi

    return Tterm


@njit
def compute_Qterm_for_one_sample(
    mIQ, mQQ, mUU, mIU, mUQ, mQU, cos2Xi2Phi, sin2Xi2Phi, cos2Psi2Phi, sin2Psi2Phi
):
    Qterm = cos2Psi2Phi * (mIQ + mQQ * cos2Xi2Phi + mUQ * sin2Xi2Phi) - sin2Psi2Phi * (
        mIU + mQU * cos2Xi2Phi + mUU * sin2Xi2Phi
    )

    return Qterm


@njit
def compute_Uterm_for_one_sample(
    mIU, mQU, mUQ, mIQ, mQQ, mUU, cos2Xi2Phi, sin2Xi2Phi, cos2Psi2Phi, sin2Psi2Phi
):
    Uterm = sin2Psi2Phi * (mIQ + mQQ * cos2Xi2Phi + mUQ * sin2Xi2Phi) + cos2Psi2Phi * (
        mIU + mQU * cos2Xi2Phi + mUU * sin2Xi2Phi
    )

    return Uterm


@njit
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
