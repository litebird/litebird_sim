# -*- encoding: utf-8 -*-
import litebird_sim as lbs
from numba import njit
import numpy as np
import healpy as hp
import logging
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from litebird_sim import mpi
from typing import Union, List
from ..mbs.mbs import MbsParameters
from ..detectors import FreqChannelInfo
from ..observations import Observation
from .bandpass_template_module import bandpass_profile
from ..coordinates import rotate_coordinates_e2g

COND_THRESHOLD = 1e10


def _dBodTrj(nu):
    return 2 * const.k_B.value * nu * nu * 1e18 / const.c.value / const.c.value


def _dBodTth(nu):
    x = const.h.value * nu * 1e9 / const.k_B.value / cosmo.Tcmb0.value
    ex = np.exp(x)
    exm1 = ex - 1.0e0
    return (
        2
        * const.h.value
        * nu
        * nu
        * nu
        * 1e27
        / const.c.value
        / const.c.value
        / exm1
        / exm1
        * ex
        * x
        / cosmo.Tcmb0.value
    )


@njit
def compute_polang_from_detquat(quat):
    if quat[2] == 0:
        polang = 0
    else:
        polang = 2 * np.arctan2(
            np.sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2), quat[3]
        )
        if quat[2] < 0:
            polang = -polang

    return polang


def JonesToMueller(jones):
    """
    Converts a Jones matrix to a Mueller matrix.
    The Jones matrix is assumed to be a 2x2 complex matrix (np.array).
    The Mueller matrix is a 4x4 real matrix.
    Credits to Yuki Sakurai for the function.
    """

    # Pauli matrix basis
    pauli_basis = np.array(
        object=[
            [[1, 0], [0, 1]],  # Identity matrix
            [[1, 0], [0, -1]],  # Pauli matrix z
            [[0, 1], [1, 0]],  # Pauli matrix x
            [[0, -1j], [1j, 0]],  # Pauli matrix y
        ],
        dtype=complex,
    )

    # Mueller matrix is M_ij = 1/2 * Tr(pauli[i] * J * pauli[j] * J^dagger)
    mueller = np.zeros((4, 4), dtype=float)

    for i in range(4):
        for j in range(4):
            Mij = 0.5 * np.trace(
                np.dot(
                    pauli_basis[i],
                    np.dot(jones, np.dot(pauli_basis[j], np.conjugate(jones).T)),
                )
            )
            # Sanity check to ensure the formula operates correctly and
            # does not yield any imaginary components.
            # Mueller-matrix elements should be real.
            if np.imag(Mij) > 1e-9:
                logging.warning("Discarding the unnecessary imaginary part!")

            mueller[i, j] += np.real(Mij)
    return mueller


def get_mueller_from_jones(h1, h2, z1, z2, beta):
    r"""
    Converts the (frequency-dependent) input Jones matrix to a Mueller matrix.
    Returns Mueller matrix (3x3xNfreq), V-mode related terms are discarded,
    given the assumption of vanishing circular polarization.

    Inputs: :math:`h_1`, :math:`h_2`, :math:`\zeta_1`, :math:`\zeta_2`,
    :math:`\beta` (i.e. systematics of the HWP, not the full Jones matrix)
    Returns: :math:`M^{II}`, :math:`M^{QI}`, :math:`M^{UI}`, :math:`M^{IQ}`,
    :math:`M^{IU}`, :math:`M^{QQ}`, :math:`M^{UU}`, :math:`M^{UQ}`, :math:`M^{QU}`
    (single/multi-frequency Mueller matrix terms)
    """

    # Convert inputs to numpy arrays
    h1, h2, z1, z2, beta = np.atleast_1d(h1, h2, z1, z2, beta)

    # Check if input arrays are of the same length
    # map returns an iterator that contains the length of each parameter
    # The set() function creates a set from the iterator obtained in the previous step
    # and remove eventual duplicates: if the length of the set is greater than 1,
    # it means that the input arrays have different lengths.
    if len(set(map(len, (h1, h2, z1, z2, beta)))) > 1:
        raise ValueError("Input arrays must have the same length.")

    if len(h1) == 1:
        # Single frequency case
        Mueller = np.zeros((4, 4))
        jones_1d = np.array(
            [[1 + h1[0], z1[0]], [z2[0], -(1 + h2[0]) * np.exp(1j * beta[0])]]
        )
        Mueller[:, :] += JonesToMueller(jones_1d)
        mII, mQI, mUI, mIQ, mIU, mQQ, mUU, mUQ, mQU = (
            Mueller[0, 0],
            Mueller[1, 0],
            Mueller[2, 0],
            Mueller[0, 1],
            Mueller[0, 2],
            Mueller[1, 1],
            Mueller[2, 2],
            Mueller[2, 1],
            Mueller[1, 2],
        )
    else:
        # Frequency-dependent case
        Mueller = np.zeros((4, 4, len(h1)))
        for i in range(len(h1)):
            jones_1d = np.array(
                [[1 + h1[i], z1[i]], [z2[i], -(1 + h2[i]) * np.exp(1j * beta[i])]]
            )
            Mueller[:, :, i] += JonesToMueller(jones_1d)
        mII, mQI, mUI, mIQ, mIU, mQQ, mUU, mUQ, mQU = (
            Mueller[0, 0, :],
            Mueller[1, 0, :],
            Mueller[2, 0, :],
            Mueller[0, 1, :],
            Mueller[0, 2, :],
            Mueller[1, 1, :],
            Mueller[2, 2, :],
            Mueller[2, 1, :],
            Mueller[1, 2, :],
        )

    return mII, mQI, mUI, mIQ, mIU, mQQ, mUU, mUQ, mQU


@njit
def compute_Tterm_for_one_sample(mII, mQI, mUI, c2ThXi, s2ThXi):
    Tterm = 0.5 * (mII + mQI * c2ThXi - mUI * s2ThXi)
    return Tterm


@njit
def compute_Qterm_for_one_sample(
    mIQ, mQQ, mUU, mIU, mUQ, mQU, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
):
    Qterm = 0.25 * (
        2 * mIQ * c2ThPs
        + (mQQ + mUU) * c2PsXi
        + (mQQ - mUU) * c4Th
        - 2 * mIU * s2ThPs
        + (mUQ - mQU) * s2PsXi
        - (mQU + mUQ) * s4Th
    )
    return Qterm


@njit
def compute_Uterm_for_one_sample(
    mIU, mQU, mUQ, mIQ, mQQ, mUU, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
):
    Uterm = 0.25 * (
        2 * mIU * c2ThPs
        + (mQU - mUQ) * c2PsXi
        + (mQU + mUQ) * c4Th
        + 2 * mIQ * s2ThPs
        + (mQQ + mUU) * s2PsXi
        + (mQQ - mUU) * s4Th
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
    c2ThPs,
    s2ThPs,
    c2PsXi,
    s2PsXi,
    c2ThXi,
    s2ThXi,
    c4Th,
    s4Th,
):
    """Bolometric equation, tod filling for a single (time) sample"""
    d = T * compute_Tterm_for_one_sample(mII, mQI, mUI, c2ThXi, s2ThXi)

    d += Q * compute_Qterm_for_one_sample(
        mIQ, mQQ, mUU, mIU, mUQ, mQU, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
    )

    d += U * compute_Uterm_for_one_sample(
        mIU, mQU, mUQ, mIQ, mQQ, mUU, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
    )

    return d


@njit
def compute_signal_for_one_detector(
    tod_det,
    mII,
    mQI,
    mUI,
    mIQ,
    mIU,
    mQQ,
    mUU,
    mUQ,
    mQU,
    pixel_ind,
    theta,
    psi,
    xi,
    maps,
):
    """
    Single-frequency case: compute the signal for a single detector,
    looping over (time) samples.
    """
    for i in range(len(tod_det)):
        tod_det[i] += compute_signal_for_one_sample(
            T=maps[0, pixel_ind[i]],
            Q=maps[1, pixel_ind[i]],
            U=maps[2, pixel_ind[i]],
            mII=mII,
            mQI=mQI,
            mUI=mUI,
            mIQ=mIQ,
            mIU=mIU,
            mQQ=mQQ,
            mUU=mUU,
            mUQ=mUQ,
            mQU=mQU,
            c2ThPs=np.cos(2 * theta[i] + 2 * psi[i]),
            s2ThPs=np.sin(2 * theta[i] + 2 * psi[i]),
            c2PsXi=np.cos(2 * psi[i] + 2 * xi),
            s2PsXi=np.sin(2 * psi[i] + 2 * xi),
            c2ThXi=np.cos(2 * theta[i] - 2 * xi),
            s2ThXi=np.sin(2 * theta[i] - 2 * xi),
            c4Th=np.cos(4 * theta[i] + 2 * psi[i] - 2 * xi),
            s4Th=np.sin(4 * theta[i] + 2 * psi[i] - 2 * xi),
        )


@njit
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
    c2ThPs,
    s2ThPs,
    c2PsXi,
    s2PsXi,
    c2ThXi,
    s2ThXi,
    c4Th,
    s4Th,
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
                    c2ThPs=c2ThPs,
                    s2ThPs=s2ThPs,
                    c2PsXi=c2PsXi,
                    s2PsXi=s2PsXi,
                    c2ThXi=c2ThXi,
                    s2ThXi=s2ThXi,
                    c4Th=c4Th,
                    s4Th=s4Th,
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
                    c2ThPs=c2ThPs,
                    s2ThPs=s2ThPs,
                    c2PsXi=c2PsXi,
                    s2PsXi=s2PsXi,
                    c2ThXi=c2ThXi,
                    s2ThXi=s2ThXi,
                    c4Th=c4Th,
                    s4Th=s4Th,
                )
            )
            * dnu
            / 2
        )

    return tod


@njit
def integrate_inband_signal_for_one_detector(
    tod_det,
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
    pixel_ind,
    theta,
    psi,
    xi,
    maps,
):
    """
    Multi-frequency case: band integration of the signal for a single detector,
    looping over (time) samples.
    """
    for i in range(len(tod_det)):
        tod_det[i] += integrate_inband_signal_for_one_sample(
            T=maps[:, 0, pixel_ind[i]],
            Q=maps[:, 1, pixel_ind[i]],
            U=maps[:, 2, pixel_ind[i]],
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
            c2ThPs=np.cos(2 * theta[i] + 2 * psi[i]),
            s2ThPs=np.sin(2 * theta[i] + 2 * psi[i]),
            c2PsXi=np.cos(2 * psi[i] + 2 * xi),
            s2PsXi=np.sin(2 * psi[i] + 2 * xi),
            c2ThXi=np.cos(2 * theta[i] - 2 * xi),
            s2ThXi=np.sin(2 * theta[i] - 2 * xi),
            c4Th=np.cos(4 * theta[i] + 2 * psi[i] - 2 * xi),
            s4Th=np.sin(4 * theta[i] + 2 * psi[i] - 2 * xi),
        )


@njit
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
    c2ThPs,
    s2ThPs,
    c2PsXi,
    s2PsXi,
    c2ThXi,
    s2ThXi,
    c4Th,
    s4Th,
):
    r"""
    Single-frequency case: computes :math:`A^T A` and :math:`A^T d`
    for a single detector, for one (time) sample.
    """
    Tterm = compute_Tterm_for_one_sample(mIIs, mQIs, mUIs, c2ThXi, s2ThXi)
    Qterm = compute_Qterm_for_one_sample(
        mIQs, mQQs, mUUs, mIUs, mUQs, mQUs, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
    )
    Uterm = compute_Uterm_for_one_sample(
        mIUs, mQUs, mUQs, mIQs, mQQs, mUUs, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
    )
    return Tterm, Qterm, Uterm


@njit
def compute_atd_ata_for_one_detector(
    atd,
    ata,
    tod,
    mIIs,
    mQIs,
    mUIs,
    mIQs,
    mIUs,
    mQQs,
    mUUs,
    mUQs,
    mQUs,
    pixel_ind,
    theta,
    psi,
    xi,
):
    r"""
    Single-frequency case: compute :math:`A^T A` and :math:`A^T d`
    for a single detector, looping over (time) samples.
    """
    for i in range(len(tod)):
        Tterm, Qterm, Uterm = compute_TQUsolver_for_one_sample(
            mIIs=mIIs,
            mQIs=mQIs,
            mUIs=mUIs,
            mIQs=mIQs,
            mIUs=mIUs,
            mQQs=mQQs,
            mUUs=mUUs,
            mUQs=mUQs,
            mQUs=mQUs,
            c2ThPs=np.cos(2 * theta[i] + 2 * psi[i]),
            s2ThPs=np.sin(2 * theta[i] + 2 * psi[i]),
            c2PsXi=np.cos(2 * psi[i] + 2 * xi),
            s2PsXi=np.sin(2 * psi[i] + 2 * xi),
            c2ThXi=np.cos(2 * theta[i] - 2 * xi),
            s2ThXi=np.sin(2 * theta[i] - 2 * xi),
            c4Th=np.cos(4 * theta[i] + 2 * psi[i] - 2 * xi),
            s4Th=np.sin(4 * theta[i] + 2 * psi[i] - 2 * xi),
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


@njit
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
    c2ThPs,
    s2ThPs,
    c2PsXi,
    s2PsXi,
    c2ThXi,
    s2ThXi,
    c4Th,
    s4Th,
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
            c2ThPs=c2ThPs,
            s2ThPs=s2ThPs,
            c2PsXi=c2PsXi,
            s2PsXi=s2PsXi,
            c2ThXi=c2ThXi,
            s2ThXi=s2ThXi,
            c4Th=c4Th,
            s4Th=s4Th,
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
            c2ThPs=c2ThPs,
            s2ThPs=s2ThPs,
            c2PsXi=c2PsXi,
            s2PsXi=s2PsXi,
            c2ThXi=c2ThXi,
            s2ThXi=s2ThXi,
            c4Th=c4Th,
            s4Th=s4Th,
        )

        intTterm += (band[i] * Tterm + band[i + 1] * Ttermp1) * dnu / 2.0
        intQterm += (band[i] * Qterm + band[i + 1] * Qtermp1) * dnu / 2.0
        intUterm += (band[i] * Uterm + band[i + 1] * Utermp1) * dnu / 2.0

    return intTterm, intQterm, intUterm


@njit
def integrate_inband_atd_ata_for_one_detector(
    atd,
    ata,
    tod,
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
    pixel_ind,
    theta,
    psi,
    xi,
):
    r"""
    Multi-frequency case: band integration of :math:`A^T A` and :math:`A^T d`
    for a single detector, looping over (time) samples.
    """
    for i in range(len(tod)):
        Tterm, Qterm, Uterm = integrate_inband_TQUsolver_for_one_sample(
            freqs=freqs,
            band=band,
            mIIs=mIIs,
            mQIs=mQIs,
            mUIs=mUIs,
            mIQs=mIQs,
            mIUs=mIUs,
            mQQs=mQQs,
            mUUs=mUUs,
            mUQs=mUQs,
            mQUs=mQUs,
            c2ThPs=np.cos(2 * theta[i] + 2 * psi[i]),
            s2ThPs=np.sin(2 * theta[i] + 2 * psi[i]),
            c2PsXi=np.cos(2 * psi[i] + 2 * xi),
            s2PsXi=np.sin(2 * psi[i] + 2 * xi),
            c2ThXi=np.cos(2 * theta[i] - 2 * xi),
            s2ThXi=np.sin(2 * theta[i] - 2 * xi),
            c4Th=np.cos(4 * theta[i] + 2 * psi[i] - 2 * xi),
            s4Th=np.sin(4 * theta[i] + 2 * psi[i] - 2 * xi),
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


class HwpSys:
    """A container object for handling tod filling in presence of hwp non-idealities
    following the approach of Giardiello et al. 2021
    https://arxiv.org/abs/2106.08031
    Args:
         simulation (:class:`.Simulation`): an instance of the class \
         :class:`.Simulation`
    """

    def __init__(self, simulation):
        self.sim = simulation

    def set_parameters(
        self,
        nside: Union[int, None] = None,
        mueller_or_jones: str = "mueller" or "jones",
        Mbsparams: Union[MbsParameters, None] = None,
        integrate_in_band: Union[bool, None] = None,
        built_map_on_the_fly: Union[bool, None] = None,
        correct_in_solver: Union[bool, None] = None,
        integrate_in_band_solver: Union[bool, None] = None,
        Channel: Union[FreqChannelInfo, None] = None,
        maps: Union[np.ndarray, None] = None,
        parallel: Union[bool, None] = None,
    ):
        r"""It sets the input paramters reading a dictionary `sim.parameters`
            with key "hwp_sys" and the following input arguments

            Args:
              nside (integer): nside used in the analysis
              mueller_or_jones (str): "mueller" or "jones" (case insensitive)
                  it is the kind of HWP matrix to be injected as a starting point
                  if 'jones' is chosen, the parameters :math:`h_1`, :math:`h_2`,
                  :math:`\beta`, :math:`\zeta_1`, :math:`\zeta_2`
                  are used to build the Jones matrix
                  :math:`\begin{pmatrix} 1 + h_1 & \zeta_1 \\
                  \zeta_2 & - (1 + h_2) e^{i \beta} \\ \end{pmatrix}`
                  and then converted to Mueller.
                  :math:`\zeta_1`, :math:`\zeta_2` are assumed to be complex
                  :math:`h_1`, :math:`h_2`, :math:`\beta` are assumed to be real
                  :math:`\beta` is assumed to be in degrees (later converted to radians.
                  To reproduce the ideal HWP case, set all Jones parameters to 0.
                  If Mueller parameters are used, set :math:`M^{II/QQ} = 1`,
                  :math:`M^{UU} = -1` and all the others to 0.
              Mbsparams (:class:`.Mbs`): an instance of the :class:`.Mbs` class
                  Input maps needs to be in galactic (mbs default)
              integrate_in_band (bool): performs the band integration for tod generation
              built_map_on_the_fly (bool): fills :math:`A^T A` and :math:`A^T d`
              correct_in_solver (bool): if the map is computed on the fly,
                                        fills :math:`A^T A` using map-making (solver)
                                        HWP parameters
              integrate_in_band_solver (bool): performs the band integration for the
                                               map-making solver
              Channel (:class:`.FreqChannelInfo`): an instance of the
                                                    :class:`.FreqChannelInfo` class
              maps (float): input maps (3, npix) coherent with nside provided,
                  Input maps needs to be in galactic (mbs default)
                  if `maps` is not None, `Mbsparams` is ignored
                  (i.e. input maps are not generated)
              parallel (bool): uses parallelization if set to True
        """
        # for parallelization
        if parallel:
            comm = lbs.MPI_COMM_WORLD
            rank = comm.Get_rank()
        else:
            comm = None
            rank = 0

        # set defaults for band integration
        hwp_sys_Mbs_make_cmb = True
        hwp_sys_Mbs_make_fg = True
        hwp_sys_Mbs_fg_models = ["pysm_synch_0", "pysm_freefree_1", "pysm_dust_0"]
        hwp_sys_Mbs_gaussian_smooth = True

        # This part sets from parameter file
        if (self.sim.parameters is not None) and (
            "hwp_sys" in self.sim.parameters.keys()
        ):
            paramdict = self.sim.parameters["hwp_sys"]

            self.nside = paramdict.get("nside", False)

            self.integrate_in_band = paramdict.get("integrate_in_band", False)
            self.built_map_on_the_fly = paramdict.get("built_map_on_the_fly", False)
            self.correct_in_solver = paramdict.get("correct_in_solver", False)
            self.integrate_in_band_solver = paramdict.get(
                "integrate_in_band_solver", False
            )

            self.bandpass = paramdict.get("bandpass", False)
            self.bandpass_solver = paramdict.get("bandpass_solver", False)
            self.include_beam_throughput = paramdict.get(
                "include_beam_throughput", False
            )

            mueller_or_jones = mueller_or_jones.lower()
            if mueller_or_jones == "jones":
                self.h1 = paramdict.get("h1", False)
                self.h2 = paramdict.get("h2", False)
                self.beta = paramdict.get("beta", False)
                if paramdict.get("z1", False):
                    self.z1 = complex(paramdict.get("z1"))
                if paramdict.get("z2", False):
                    self.z2 = complex(paramdict.get("z2"))

                self.h1s = paramdict.get("h1s", False)
                self.h2s = paramdict.get("h2s", False)
                self.betas = paramdict.get("betas", False)
                if paramdict.get("z1s", False):
                    self.z1s = complex(paramdict.get("z1s"))
                if paramdict.get("z2s", False):
                    self.z2s = complex(paramdict.get("z2s"))

            elif mueller_or_jones == "mueller":
                self.mII = paramdict.get("mII", False)
                self.mQI = paramdict.get("mQI", False)
                self.mUI = paramdict.get("mUI", False)
                self.mIQ = paramdict.get("mIQ", False)
                self.mIU = paramdict.get("mIU", False)
                self.mQQ = paramdict.get("mQQ", False)
                self.mUU = paramdict.get("mUU", False)
                self.mUQ = paramdict.get("mUQ", False)
                self.mQU = paramdict.get("mQU", False)

                self.mIIs = paramdict.get("mIIs", False)
                self.mQIs = paramdict.get("mQIs", False)
                self.mUIs = paramdict.get("mUIs", False)
                self.mIQs = paramdict.get("mIQs", False)
                self.mIUs = paramdict.get("mIUs", False)
                self.mQQs = paramdict.get("mQQs", False)
                self.mUUs = paramdict.get("mUUs", False)
                self.mUQs = paramdict.get("mUQs", False)
                self.mQUs = paramdict.get("mQUs", False)

            else:
                raise ValueError("mueller_or_jones not specified")

            self.band_filename = paramdict.get("band_filename", False)
            self.band_filename_solver = paramdict.get("band_filename_solver", False)

            # here we set the values for Mbs used in the code if present
            # in paramdict, otherwise defaults
            hwp_sys_Mbs_make_cmb = paramdict.get("hwp_sys_Mbs_make_cmb", True)
            hwp_sys_Mbs_make_fg = paramdict.get("hwp_sys_Mbs_make_fg", True)
            hwp_sys_Mbs_fg_models = paramdict.get(
                "hwp_sys_Mbs_fg_models",
                ["pysm_synch_1", "pysm_freefree_1", "pysm_dust_1", "pysm_ame_1"],
            )
            hwp_sys_Mbs_gaussian_smooth = paramdict.get(
                "hwp_sys_Mbs_gaussian_smooth", True
            )
        # This part sets from input_parameters()
        # if not self.nside:
        if nside is None:
            self.nside = 512
        else:
            self.nside = nside

        if (self.sim.parameters is not None) and (
            "hwp_sys" in self.sim.parameters.keys()
        ):
            if "general" in self.sim.parameters.keys():
                if "nside" in self.sim.parameters["general"].keys():
                    if self.sim.parameters["general"]["nside"] != self.nside:
                        print(
                            "Warning!! nside from general "
                            "(=%i) and hwp_sys (=%i) do not match. Using hwp_sys"
                            % (
                                self.sim.parameters["general"]["nside"],
                                self.nside,
                            )
                        )

        if not self.integrate_in_band:
            if integrate_in_band is not None:
                self.integrate_in_band = integrate_in_band

        if not self.built_map_on_the_fly:
            if built_map_on_the_fly is not None:
                self.built_map_on_the_fly = built_map_on_the_fly

        if not self.correct_in_solver:
            if correct_in_solver is not None:
                self.correct_in_solver = correct_in_solver

        if not self.integrate_in_band_solver:
            if integrate_in_band_solver is not None:
                self.integrate_in_band_solver = integrate_in_band_solver

        if Mbsparams is None and np.any(maps) is None:
            Mbsparams = lbs.MbsParameters(
                make_cmb=hwp_sys_Mbs_make_cmb,
                make_fg=hwp_sys_Mbs_make_fg,
                fg_models=hwp_sys_Mbs_fg_models,
                gaussian_smooth=hwp_sys_Mbs_gaussian_smooth,
                bandpass_int=False,
                maps_in_ecliptic=False,
                nside=self.nside,
            )

        if np.any(maps) is None:
            Mbsparams.nside = self.nside

        self.npix = hp.nside2npix(self.nside)

        if Channel is None:
            Channel = lbs.FreqChannelInfo(bandcenter_ghz=100)

        if self.integrate_in_band:
            if mueller_or_jones == "jones":
                try:
                    (
                        self.freqs,
                        self.h1,
                        self.h2,
                        self.beta,
                        self.z1,
                        self.z2,
                    ) = np.loadtxt(
                        self.band_filename, dtype=complex, unpack=True, comments="#"
                    )

                    self.freqs = np.array(self.freqs, dtype=float)
                    self.h1 = np.array(self.h1, dtype=float)
                    self.h2 = np.array(self.h2, dtype=float)
                    self.beta = np.array(self.beta, dtype=float)
                    self.beta = np.deg2rad(self.beta)

                except Exception:
                    print(
                        "missing band_filename in the parameter file"
                        + " or wrong number of columns for jones matrix"
                    )
            else:  # mueller_or_jones == "mueller"
                try:
                    (
                        self.freqs,
                        self.mII,
                        self.mQI,
                        self.mUI,
                        self.mIQ,
                        self.mIU,
                        self.mQQ,
                        self.mUU,
                        self.mUQ,
                        self.mQU,
                    ) = np.loadtxt(self.band_filename, unpack=True, comments="#")
                except Exception:
                    print(
                        "missing band_filename in the parameter file"
                        + " or wrong number of columns for mueller matrix"
                    )
            self.nfreqs = len(self.freqs)

            if not self.bandpass:
                self.cmb2bb = _dBodTth(self.freqs)

            elif self.bandpass:
                self.freqs, self.bandpass_profile = bandpass_profile(
                    self.freqs, self.bandpass, self.include_beam_throughput
                )

                self.cmb2bb = _dBodTth(self.freqs) * self.bandpass_profile

            # Normalize the band
            self.cmb2bb /= np.trapz(self.cmb2bb, self.freqs)

            if np.any(maps) is None:
                if rank == 0:
                    myinstr = {}
                    for ifreq in range(self.nfreqs):
                        myinstr["ch" + str(ifreq)] = {
                            "bandcenter_ghz": self.freqs[ifreq],
                            "bandwidth_ghz": 0,
                            "fwhm_arcmin": Channel.fwhm_arcmin,
                            "p_sens_ukarcmin": 0.0,
                            "band": None,
                        }

                    mbs = lbs.Mbs(
                        simulation=self.sim, parameters=Mbsparams, instrument=myinstr
                    )

                    maps = mbs.run_all()[0]
                    self.maps = np.empty((self.nfreqs, 3, self.npix))
                    for ifreq in range(self.nfreqs):
                        self.maps[ifreq] = maps["ch" + str(ifreq)]
                else:
                    self.maps = None
                if parallel:
                    self.maps = comm.bcast(self.maps, root=0)
            else:
                self.maps = maps
            del maps

        else:
            if mueller_or_jones == "jones":
                default_attrs = {
                    "h1": 0.0,
                    "h2": 0.0,
                    "beta": 0.0,
                    "z1": 0.0,
                    "z2": 0.0,
                }

                for attr, default_value in default_attrs.items():
                    if not hasattr(self, attr):
                        setattr(self, attr, default_value)
                self.beta = np.deg2rad(self.beta)
            else:  # mueller_or_jones == "mueller":
                default_attrs = {
                    "mII": 0.0,
                    "mQI": 0.0,
                    "mUI": 0.0,
                    "mIQ": 0.0,
                    "mIU": 0.0,
                    "mQQ": 0.0,
                    "mUU": 0.0,
                    "mUQ": 0.0,
                    "mQU": 0.0,
                }

                for attr, default_value in default_attrs.items():
                    if not hasattr(self, attr):
                        setattr(self, attr, default_value)

            if np.any(maps) is None:
                mbs = lbs.Mbs(
                    simulation=self.sim, parameters=Mbsparams, channel_list=Channel
                )
                self.maps = mbs.run_all()[0][Channel.channel]
            else:
                self.maps = maps
                del maps

        if self.correct_in_solver:
            if self.integrate_in_band_solver:
                if mueller_or_jones == "jones":
                    try:
                        (
                            self.freqs_solver,
                            self.h1s,
                            self.h2s,
                            self.betas,
                            self.z1s,
                            self.z2s,
                        ) = np.loadtxt(
                            self.band_filename_solver,
                            dtype=complex,
                            unpack=True,
                            comments="#",
                        )

                        self.freqs_solver = np.array(self.freqs_solver, dtype=float)
                        self.h1s = np.array(self.h1s, dtype=float)
                        self.h2s = np.array(self.h2s, dtype=float)
                        self.betas = np.array(self.betas, dtype=float)
                        self.betas = np.deg2rad(self.betas)

                    except Exception:
                        print(
                            "you have not provided a band_filename_solver"
                            + "in the parameter file, or wrong number of columns!"
                        )

                else:  # mueller_or_jones == "mueller":
                    try:
                        (
                            self.freqs_solver,
                            self.mIIs,
                            self.mQIs,
                            self.mUIs,
                            self.mIQs,
                            self.mIUs,
                            self.mQQs,
                            self.mUUs,
                            self.mUQs,
                            self.mQUs,
                        ) = np.loadtxt(
                            self.band_filename_solver,
                            unpack=True,
                            comments="#",
                        )
                    except Exception:
                        print(
                            "you have not provided a band_filename_solver"
                            + "in the parameter file, or wrong number of columns!"
                        )

                if not self.bandpass_solver:
                    self.cmb2bb_solver = _dBodTth(self.freqs_solver)

                elif self.bandpass_solver:
                    self.freqs_solver, self.bandpass_profile_solver = bandpass_profile(
                        self.freqs_solver,
                        self.bandpass_solver,
                        self.include_beam_throughput,
                    )
                    self.cmb2bb_solver = (
                        _dBodTth(self.freqs_solver) * self.bandpass_profile_solver
                    )

                self.cmb2bb_solver /= np.trapz(self.cmb2bb_solver, self.freqs_solver)

            else:
                if mueller_or_jones == "jones":
                    default_attrs = {
                        "h1s": 0.0,
                        "h2s": 0.0,
                        "betas": 0.0,
                        "z1s": 0.0,
                        "z2s": 0.0,
                    }

                for attr, default_value in default_attrs.items():
                    if not hasattr(self, attr):
                        setattr(self, attr, default_value)
                    self.betas = np.deg2rad(self.betas)
                else:  # mueller_or_jones == "mueller":
                    default_attrs = {
                        "mIIs": 0.0,
                        "mQIs": 0.0,
                        "mUIs": 0.0,
                        "mIQs": 0.0,
                        "mIUs": 0.0,
                        "mQQs": 0.0,
                        "mUUs": 0.0,
                        "mUQs": 0.0,
                        "mQUs": 0.0,
                    }

                for attr, default_value in default_attrs.items():
                    if not hasattr(self, attr):
                        setattr(self, attr, default_value)

        # conversion from Jones to Mueller
        if mueller_or_jones == "jones":
            # Mueller terms of fixed HWP (single/multi freq), to fill tod
            (
                self.mII,
                self.mQI,
                self.mUI,
                self.mIQ,
                self.mIU,
                self.mQQ,
                self.mUU,
                self.mUQ,
                self.mQU,
            ) = get_mueller_from_jones(
                h1=self.h1, h2=self.h2, z1=self.z1, z2=self.z2, beta=self.beta
            )

            # Mueller terms of fixed HWP (single/multi freq), to fill A^TA and A^Td
            if self.correct_in_solver:
                (
                    self.mIIs,
                    self.mQIs,
                    self.mUIs,
                    self.mIQs,
                    self.mIUs,
                    self.mQQs,
                    self.mUUs,
                    self.mUQs,
                    self.mQUs,
                ) = get_mueller_from_jones(
                    h1=self.h1s, h2=self.h2s, z1=self.z1s, z2=self.z2s, beta=self.betas
                )

    def fill_tod(
        self,
        observations: Union[Observation, List[Observation]],
        pointings: Union[np.ndarray, List[np.ndarray], None] = None,
        hwp_angle: Union[np.ndarray, List[np.ndarray], None] = None,
        input_map_in_galactic: bool = True,
        save_tod: bool = False,
        dtype_pointings=np.float32,
    ):
        r"""It fills tod and/or :math:`A^T A` and :math:`A^T d` for the
                "on the fly" map production

                Args:

                observations (:class:`Observation`): container for tod. If the tod is
                         not required, you can avoid allocating ``observations.tod``
                         i.e. in ``lbs.Observation`` use ``allocate_tod=False``.

                pointings (optional): if not passed, it is either computed on the fly
                        (generated by :func:`lbs.get_pointings` per detector),
                        or read from ``observations.pointing_matrix`` (if present).
        <<<<<<< HEAD
                        if ``observations`` is not a list, ``pointings`` must be a np.array
                        of dimensions (N_det, N_samples, 3)
                        if ``observations`` is a list, ``pointings`` must be a list of same length
                        When generating pointing information, set the variable
                        ``hwp`` to None since the hwp rotation angle is added to
                        the orientation angle within the ``fill_tod`` function.
        =======
                        If ``observations`` is not a list, ``pointings`` must be a np.array
                            of dimensions (N_det, N_samples, 3).
                        If ``observations`` is a list, ``pointings`` must be a list of same length.
        >>>>>>> 87b0eaa34357e6dc6870500f85a2d3a894bce45e

                hwp_angle (optional): `2ωt`, hwp rotation angles (radians). If ``pointings`` is passed,
                        ``hwp_angle`` must be passed as well, otherwise both must be ``None``.
                        If not passed, it is computed on the fly (generated by :func:`lbs.get_pointings`
                        per detector).
                        If ``observations`` is not a list, ``hwp_angle`` must be a np.array
                            of dimensions (N_samples).
                        If ``observations`` is a list, ``hwp_angle`` must be a list of same length.

                input_map_in_galactic (bool): if True, the input map is in galactic coordinates, pointings
                        are rotated from ecliptic to galactic and output map will also be in galactic.

                save_tod (bool): if True, ``tod`` is saved in ``observations.tod``; if False,
        <<<<<<< HEAD
                        ``tod`` gets deleted

                dtype_pointings: if ``pointings`` is None and is computed within ``fill_tod``, this
                        is the dtype for pointings and tod (default: np.float32)
        =======
                         ``tod`` gets deleted.

                dtype_pointings: if ``pointings`` is None and is computed within ``fill_tod``, this
                                 is the dtype for pointings and tod (default: np.float32).
        >>>>>>> 87b0eaa34357e6dc6870500f85a2d3a894bce45e
        """

        if pointings is None:
            if hwp_angle:
                raise Warning(
                    "You passed hwp_angle, but you did not pass pointings, "
                    + "so hwp_angle will be ignored and re-computed on the fly."
                )
            hwp_angle_list = []
            if isinstance(observations, Observation):
                obs_list = [observations]
                if hasattr(observations, "pointing_matrix"):
                    ptg_list = [observations.pointing_matrix]
                else:
                    ptg_list = []
            else:
                obs_list = observations
                ptg_list = []
                for ob in observations:
                    if hasattr(ob, "pointing_matrix"):
                        ptg_list.append(ob.pointing_matrix)
        else:
            if isinstance(observations, Observation):
                assert isinstance(pointings, np.ndarray), (
                    "For one observation you need to pass a np.array "
                    + "of pointings to fill_tod"
                )
                assert (
                    observations.n_detectors == pointings.shape[0]
                    and observations.n_samples == pointings.shape[1]
                    and pointings.shape[2] == 3
                ), (
                    "You need to pass a pointing np.array with shape"
                    + "(N_det, N_samples, 3) for the observation"
                )
                obs_list = [observations]
                ptg_list = [pointings]
                if hwp_angle:
                    assert isinstance(hwp_angle, np.ndarray), (
                        "For one observation, hwp_angle must be passed "
                        + "as a np.array to fill_tod"
                    )
                    assert observations.n_samples == hwp_angle.shape[0], (
                        "You need to pass a hwp_angle np.array with shape"
                        + "N_samples for the observation"
                    )
                    hwp_angle_list = [hwp_angle]
                else:
                    raise ValueError(
                        "If you pass pointings, you must also pass hwp_angle."
                    )
            else:
                assert isinstance(pointings, list), (
                    "When you pass a list of observations to fill_tod, "
                    + "you must a list of `pointings`"
                )
                assert len(observations) == len(pointings), (
                    f"The list of observations has {len(observations)} elements, but "
                    + f"the list of pointings has {len(pointings)} elements"
                )
                obs_list = observations
                ptg_list = pointings
                if hwp_angle:
                    assert len(observations) == len(hwp_angle), (
                        f"The list of observations has {len(observations)} elements, but "
                        + f"the list of hwp_angle has {len(hwp_angle)} elements"
                    )
                    hwp_angle_list = hwp_angle
                else:
                    raise ValueError(
                        "If you pass pointings, you must also pass hwp_angle."
                    )

        for idx_obs, cur_obs in enumerate(obs_list):
            if self.built_map_on_the_fly:
                self.atd = np.zeros((self.npix, 3))
                self.ata = np.zeros((self.npix, 3, 3))
            else:
                # allocate those for "make_binned_map", later filled
                if not hasattr(cur_obs, "pointing_matrix"):
                    cur_obs.pointing_matrix = np.empty(
                        (cur_obs.n_detectors, cur_obs.n_samples, 3),
                        dtype=dtype_pointings,
                    )

            for idet in range(cur_obs.n_detectors):
                if pointings is None:
                    if (not ptg_list) or (not hwp_angle_list):
                        cur_point, cur_hwp_angle = cur_obs.get_pointings(
                            detector_idx=idet, pointings_dtype=dtype_pointings
                        )
                        cur_point = cur_point.reshape(-1, 3)
                    else:
                        cur_point = ptg_list[idx_obs][idet, :, :]
                        cur_hwp_angle = hwp_angle_list[idx_obs]
                else:
                    cur_point = ptg_list[idx_obs][idet, :, :]
                    cur_hwp_angle = hwp_angle_list[idx_obs]

                # rotating pointing from ecliptic to galactic as the input map
                if input_map_in_galactic:
                    cur_point = rotate_coordinates_e2g(cur_point)

                # all observed pixels over time (for each sample),
                # i.e. len(pix)==len(times)
                pix = hp.ang2pix(self.nside, cur_point[:, 0], cur_point[:, 1])

                # separating polarization angle xi from cur_point[:, 2] = psi + xi
                # xi: polarization angle, i.e. detector dependent
                # psi: instrument angle, i.e. boresight direction from focal plane POV
                xi = compute_polang_from_detquat(cur_obs.quat[idet].quats[0]) % (
                    2 * np.pi
                )
                psi = (cur_point[:, 2] - xi) % (2 * np.pi)

                tod = cur_obs.tod[idet, :]

                if self.integrate_in_band:
                    integrate_inband_signal_for_one_detector(
                        tod_det=tod,
                        freqs=self.freqs,
                        band=self.cmb2bb,
                        mII=self.mII,
                        mQI=self.mQI,
                        mUI=self.mUI,
                        mIQ=self.mIQ,
                        mIU=self.mIU,
                        mQQ=self.mQQ,
                        mUU=self.mUU,
                        mUQ=self.mUQ,
                        mQU=self.mQU,
                        pixel_ind=pix,
                        theta=cur_hwp_angle / 2,  # hwp angle returns 2ωt
                        psi=psi,
                        xi=xi,
                        maps=self.maps,
                    )
                else:
                    compute_signal_for_one_detector(
                        tod_det=tod,
                        mII=self.mII,
                        mQI=self.mQI,
                        mUI=self.mUI,
                        mIQ=self.mIQ,
                        mIU=self.mIU,
                        mQQ=self.mQQ,
                        mUU=self.mUU,
                        mUQ=self.mUQ,
                        mQU=self.mQU,
                        pixel_ind=pix,
                        theta=cur_hwp_angle / 2,  # hwp angle returns 2ωt
                        psi=psi,
                        xi=xi,
                        maps=self.maps,
                    )

                if self.built_map_on_the_fly:
                    if self.correct_in_solver:
                        if self.integrate_in_band_solver:
                            integrate_inband_atd_ata_for_one_detector(
                                atd=self.atd,
                                ata=self.ata,
                                tod=tod,
                                freqs=self.freqs_solver,
                                band=self.cmb2bb_solver,
                                mIIs=self.mIIs,
                                mQIs=self.mQIs,
                                mUIs=self.mUIs,
                                mIQs=self.mIQs,
                                mIUs=self.mIUs,
                                mQQs=self.mQQs,
                                mUUs=self.mUUs,
                                mUQs=self.mUQs,
                                mQUs=self.mQUs,
                                pixel_ind=pix,
                                theta=cur_hwp_angle / 2,  # hwp angle returns 2ωt
                                psi=psi,
                                xi=xi,
                            )
                        else:
                            compute_atd_ata_for_one_detector(
                                atd=self.atd,
                                ata=self.ata,
                                tod=tod,
                                mIIs=self.mIIs,
                                mQIs=self.mQIs,
                                mUIs=self.mUIs,
                                mIQs=self.mIQs,
                                mIUs=self.mIUs,
                                mQQs=self.mQQs,
                                mUUs=self.mUUs,
                                mUQs=self.mUQs,
                                mQUs=self.mQUs,
                                pixel_ind=pix,
                                theta=cur_hwp_angle / 2,  # hwp angle returns 2ωt
                                psi=psi,
                                xi=xi,
                            )

                    else:
                        # in this case factor 4 included here
                        ca = np.cos(
                            2 * cur_point[:, 2] + 4 * cur_hwp_angle / 2
                        )  # hwp angle returns 2ωt
                        sa = np.sin(
                            2 * cur_point[:, 2] + 4 * cur_hwp_angle / 2
                        )  # hwp angle returns 2ωt

                        self.atd[pix, 0] += tod * 0.5
                        self.atd[pix, 1] += tod * ca * 0.5
                        self.atd[pix, 2] += tod * sa * 0.5

                        self.ata[pix, 0, 0] += 0.25
                        self.ata[pix, 1, 0] += 0.25 * ca
                        self.ata[pix, 2, 0] += 0.25 * sa
                        self.ata[pix, 1, 1] += 0.25 * ca * ca
                        self.ata[pix, 2, 1] += 0.25 * ca * sa
                        self.ata[pix, 2, 2] += 0.25 * sa * sa
                        del (ca, sa)

                else:
                    # this fills variables needed by make_binned_map
                    cur_obs.pointing_matrix[idet, :, :] = cur_point

        del (pix, self.maps)
        if not save_tod:
            del tod

        del (
            self.mII,
            self.mQI,
            self.mUI,
            self.mIQ,
            self.mIU,
            self.mQQ,
            self.mUU,
            self.mUQ,
            self.mQU,
        )
        if self.correct_in_solver:
            del (
                self.mIIs,
                self.mQIs,
                self.mUIs,
                self.mIQs,
                self.mIUs,
                self.mQQs,
                self.mUUs,
                self.mUQs,
                self.mQUs,
            )

        return

    def make_map(self, observations):
        """It generates "on the fly" map. This option is only availabe if
        `built_map_on_the_fly` is set to True.

        Args:
             observations list of class:`Observations`: only necessary for the communicator
             pointings (float): pointing for each sample and detector
                 generated by lbs.get_pointings
             hwp_radpsec (float): hwp rotation speed in radiants per second
        Returns:
            map (float): rebinned T,Q,U maps
        """

        assert (
            self.built_map_on_the_fly
        ), "make_map available only with built_map_on_the_fly option activated"

        # from mapping.py
        if all([obs.comm is None for obs in observations]) or not mpi.MPI_ENABLED:
            # Serial call
            pass
        elif all(
            [
                mpi.MPI.Comm.Compare(observations[i].comm, observations[i + 1].comm) < 2
                for i in range(len(observations) - 1)
            ]
        ):
            self.atd = observations[0].comm.allreduce(self.atd, mpi.MPI.SUM)
            self.ata = observations[0].comm.allreduce(self.ata, mpi.MPI.SUM)
        else:
            raise NotImplementedError(
                "All observations must be distributed over the same MPI groups"
            )

        self.ata[:, 0, 1] = self.ata[:, 1, 0]
        self.ata[:, 0, 2] = self.ata[:, 2, 0]
        self.ata[:, 1, 2] = self.ata[:, 2, 1]

        cond = np.linalg.cond(self.ata)
        res = np.full_like(self.atd, hp.UNSEEN)
        mask = cond < COND_THRESHOLD
        res[mask] = np.linalg.solve(self.ata[mask], self.atd[mask])
        return res.T
