# -*- encoding: utf-8 -*-
import litebird_sim as lbs
from numba import njit
import numpy as np
import healpy as hp
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from litebird_sim import mpi
from typing import Union, List
from ..mbs.mbs import MbsParameters
from ..detectors import FreqChannelInfo
from ..observations import Observation
from ..noise import rescale_noise
from ..coordinates import rotate_coordinates_e2g, CoordinateSystem

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
                print("Discarding the unnecessary imaginary part!")

            mueller[i, j] += np.real(Mij)
    return mueller


def get_mueller_from_jones(h1, h2, z1, z2, beta):
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
    """Bolometric equation"""
    d = T * compute_Tterm_for_one_sample(mII, mQI, mUI, c2ThXi, s2ThXi)

    d += Q * compute_Qterm_for_one_sample(
        mIQ, mQQ, mUU, mIU, mUQ, mQU, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
    )

    d += U * compute_Uterm_for_one_sample(
        mIU, mQU, mUQ, mIQ, mQQ, mUU, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
    )

    return d


@njit
def integrate_inband_signal_for_one_sample(
    T,
    Q,
    U,
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
    # perform band integration, assumed delta nu = 1GHz
    tod = 0
    for i in range(len(band)):
        tod += band[i] * compute_signal_for_one_sample(
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

    return tod


@njit
def integrate_inband_signal_for_one_detector(
    tod_det,
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
    for i in range(len(tod_det)):
        tod_det[i] += integrate_inband_signal_for_one_sample(
            T=maps[:, 0, pixel_ind[i]],
            Q=maps[:, 1, pixel_ind[i]],
            U=maps[:, 2, pixel_ind[i]],
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
    # single frequency case
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
    Tterm = compute_Tterm_for_one_sample(mIIs, mQIs, mUIs, c2ThXi, s2ThXi)
    Qterm = compute_Qterm_for_one_sample(
        mIQs, mQQs, mUUs, mIUs, mUQs, mQUs, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
    )
    Uterm = compute_Uterm_for_one_sample(
        mIUs, mQUs, mUQs, mIQs, mQQs, mUUs, c2ThPs, c2PsXi, c4Th, s2ThPs, s2PsXi, s4Th
    )
    return Tterm, Qterm, Uterm


@njit
def integrate_inband_TQUsolver_for_one_sample(
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
    # inband integration
    intTterm = 0
    intQterm = 0
    intUterm = 0
    for i in range(len(band)):
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

        intTterm += band[i] * Tterm
        intQterm += band[i] * Qterm
        intUterm += band[i] * Uterm

    return intTterm, intQterm, intUterm


@njit
def integrate_inband_atd_ata_for_one_detector(
    atd,
    ata,
    tod,
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
    for i in range(len(tod)):
        Tterm, Qterm, Uterm = integrate_inband_TQUsolver_for_one_sample(
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
    # single frequency case
    for i in range(len(tod)):
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
        inband_profile: Union[np.ndarray, None] = None,
        built_map_on_the_fly: Union[bool, None] = None,
        correct_in_solver: Union[bool, None] = None,
        integrate_in_band_solver: Union[bool, None] = None,
        Channel: Union[FreqChannelInfo, None] = None,
        maps: Union[np.ndarray, None] = None,
    ):
        """It sets the input paramters
        Args:
             nside (integer): nside used in the analysis
             mueller_or_jones (str): "mueller" or "jones" (case insensitive)
                    it is the kind of HWP matrix to be injected as a starting point
                    if 'jones' is chosen, the parameters h1, h2, beta, z1, z2
                    are used to build the Jones matrix and then converted to Mueller
             Mbsparams (:class:`.Mbs`): an instance of the :class:`.Mbs` class
             integrate_in_band (bool): performs the band integration for tod generation
             built_map_on_the_fly (bool): fills A^TA and A^Td for integrating
             correct_in_solver (bool): if the map is computed on the fly, A^TA
             integrate_in_band_solver (bool): performs the band integration for the
                                              map-making solver
             Channel (:class:`.FreqChannelInfo`): an instance of the
                                                  :class:`.FreqChannelInfo` class
             maps (float): input maps (3, npix) coherent with nside provided.
        """

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

            if "nside" in paramdict.keys():
                self.nside = paramdict["nside"]
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

            if "integrate_in_band" in paramdict.keys():
                self.integrate_in_band = paramdict["integrate_in_band"]

            if "built_map_on_the_fly" in paramdict.keys():
                self.built_map_on_the_fly = paramdict["built_map_on_the_fly"]

            if "correct_in_solver" in paramdict.keys():
                self.correct_in_solver = paramdict["correct_in_solver"]

            if "integrate_in_band_solver" in paramdict.keys():
                self.integrate_in_band_solver = paramdict["integrate_in_band_solver"]

            mueller_or_jones = mueller_or_jones.lower()
            if mueller_or_jones == "jones":
                if "h1" in paramdict.keys():
                    self.h1 = paramdict["h1"]

                if "h2" in paramdict.keys():
                    self.h2 = paramdict["h2"]

                if "beta" in paramdict.keys():
                    self.beta = paramdict["beta"]

                if "z1" in paramdict.keys():
                    self.z1 = paramdict["z1"]

                if "z2" in paramdict.keys():
                    self.z2 = paramdict["z2"]

                if "h1s" in paramdict.keys():
                    self.h1s = paramdict["h1s"]

                if "h2s" in paramdict.keys():
                    self.h2s = paramdict["h2s"]

                if "betas" in paramdict.keys():
                    self.betas = paramdict["betas"]

                if "z1s" in paramdict.keys():
                    self.z1s = paramdict["z1s"]

                if "z2s" in paramdict.keys():
                    self.z2s = paramdict["z2s"]
            elif mueller_or_jones == "mueller":
                if "mII" in paramdict.keys():
                    self.mII = paramdict["mII"]

                if "mQI" in paramdict.keys():
                    self.mQI = paramdict["mQI"]

                if "mUI" in paramdict.keys():
                    self.mUI = paramdict["mUI"]

                if "mIQ" in paramdict.keys():
                    self.mIQ = paramdict["mIQ"]

                if "mIU" in paramdict.keys():
                    self.mIU = paramdict["mIU"]

                if "mQQ" in paramdict.keys():
                    self.mQQ = paramdict["mQQ"]

                if "mUU" in paramdict.keys():
                    self.mUU = paramdict["mUU"]

                if "mUQ" in paramdict.keys():
                    self.mUQ = paramdict["mUQ"]

                if "mQU" in paramdict.keys():
                    self.mQU = paramdict["mQU"]

                if "mIIs" in paramdict.keys():
                    self.mIIs = paramdict["mIIs"]

                if "mQIs" in paramdict.keys():
                    self.mQIs = paramdict["mQIs"]

                if "mUIs" in paramdict.keys():
                    self.mUIs = paramdict["mUIs"]

                if "mIQs" in paramdict.keys():
                    self.mIQs = paramdict["mIQs"]

                if "mIUs" in paramdict.keys():
                    self.mIUs = paramdict["mIUs"]

                if "mQQs" in paramdict.keys():
                    self.mQQs = paramdict["mQQs"]

                if "mUUs" in paramdict.keys():
                    self.mUUs = paramdict["mUUs"]

                if "mUQs" in paramdict.keys():
                    self.mUQs = paramdict["mUQs"]

                if "mQUs" in paramdict.keys():
                    self.mQUs = paramdict["mQUs"]
            else:
                raise ValueError("mueller_or_jones not specified")

            if "band_filename" in paramdict.keys():
                self.band_filename = paramdict["band_filename"]

            if "band_filename_solver" in paramdict.keys():
                self.band_filename_solver = paramdict["band_filename_solver"]

            # here we set the values for Mbs used in the code
            if "hwp_sys_Mbs_make_cmb" in paramdict.keys():
                hwp_sys_Mbs_make_cmb = paramdict["hwp_sys_Mbs_make_cmb"]

            if "hwp_sys_Mbs_make_fg" in paramdict.keys():
                hwp_sys_Mbs_make_fg = paramdict["hwp_sys_Mbs_make_fg"]

            if "hwp_sys_Mbs_fg_models" in paramdict.keys():
                hwp_sys_Mbs_fg_models = paramdict["hwp_sys_Mbs_fg_models"]

            if "hwp_sys_Mbs_gaussian_smooth" in paramdict.keys():
                hwp_sys_Mbs_gaussian_smooth = paramdict["hwp_sys_Mbs_gaussian_smooth"]

        # This part sets from input_parameters()
        try:
            self.nside
        except Exception:
            if nside is None:
                self.nside = 512
            else:
                self.nside = nside

        try:
            self.integrate_in_band
        except Exception:
            if integrate_in_band is None:
                self.integrate_in_band = False
            else:
                self.integrate_in_band = integrate_in_band

        try:
            self.built_map_on_the_fly
        except Exception:
            if built_map_on_the_fly is None:
                self.built_map_on_the_fly = False
            else:
                self.built_map_on_the_fly = built_map_on_the_fly

        try:
            self.correct_in_solver
        except Exception:
            if correct_in_solver is None:
                self.correct_in_solver = False
            else:
                self.correct_in_solver = correct_in_solver

        try:
            self.integrate_in_band_solver
        except Exception:
            if integrate_in_band_solver is None:
                self.integrate_in_band_solver = False
            else:
                self.integrate_in_band_solver = integrate_in_band_solver

        if Mbsparams is None:
            Mbsparams = lbs.MbsParameters(
                make_cmb=hwp_sys_Mbs_make_cmb,
                make_fg=hwp_sys_Mbs_make_fg,
                fg_models=hwp_sys_Mbs_fg_models,
                gaussian_smooth=hwp_sys_Mbs_gaussian_smooth,
                bandpass_int=False,
                maps_in_ecliptic=True,
            )

        Mbsparams.nside = self.nside

        self.npix = hp.nside2npix(self.nside)

        if Channel is None:
            Channel = lbs.FreqChannelInfo(bandcenter_ghz=100)

        if self.integrate_in_band:
            if mueller_or_jones == "jones":
                self.freqs, self.h1, self.h2, self.beta, self.z1, self.z2 = np.loadtxt(
                    self.band_filename, unpack=True, skiprows=1
                )
            else:  # mueller_or_jones == "mueller"
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
                ) = np.loadtxt(self.band_filename, unpack=True, skiprows=1)

            self.nfreqs = len(self.freqs)

            if inband_profile is not None:
                self.cmb2bb = _dBodTth(self.freqs) * inband_profile
            else:
                self.cmb2bb = _dBodTth(self.freqs)
            # Normalize the bandpass
            self.cmb2bb /= np.trapz(self.cmb2bb, self.freqs)

            myinstr = {}
            for ifreq in range(self.nfreqs):
                myinstr["ch" + str(ifreq)] = {
                    "bandcenter_ghz": self.freqs[ifreq],
                    "bandwidth_ghz": 0,
                    "fwhm_arcmin": Channel.fwhm_arcmin,
                    "p_sens_ukarcmin": 0.0,
                }

            mbs = lbs.Mbs(simulation=self.sim, parameters=Mbsparams, instrument=myinstr)

            if np.any(maps) is None:
                maps = mbs.run_all()[0]
                self.maps = np.empty((self.nfreqs, 3, self.npix))
                for ifreq in range(self.nfreqs):
                    self.maps[ifreq] = maps["ch" + str(ifreq)]
            else:
                assert (
                    hp.npix2nside(len(maps[0, 0, :])) == self.nside
                ), "wrong nside in the input map!"
                assert (
                    len(maps[:, 0, 0]) == self.nfreqs
                ), "wrong number of frequencies: expected a different number of maps!"
                self.maps = maps
            del maps

        else:
            if mueller_or_jones == "jones":
                if not hasattr(self, "h1"):
                    self.h1 = 0.0
                if not hasattr(self, "h2"):
                    self.h2 = 0.0
                if not hasattr(self, "beta"):
                    self.beta = 0.0
                if not hasattr(self, "z1"):
                    self.z1 = 0.0
                if not hasattr(self, "z2"):
                    self.z2 = 0.0
            else:  # mueller_or_jones == "mueller":
                if not hasattr(self, "mII"):
                    self.mII = 0.0
                if not hasattr(self, "mQI"):
                    self.mQI = 0.0
                if not hasattr(self, "mUI"):
                    self.mUI = 0.0
                if not hasattr(self, "mIQ"):
                    self.mIQ = 0.0
                if not hasattr(self, "mIU"):
                    self.mIU = 0.0
                if not hasattr(self, "mQQ"):
                    self.mQQ = 0.0
                if not hasattr(self, "mUU"):
                    self.mUU = 0.0
                if not hasattr(self, "mUQ"):
                    self.mUQ = 0.0
                if not hasattr(self, "mQU"):
                    self.mQU = 0.0

            if np.any(maps) is None:
                mbs = lbs.Mbs(
                    simulation=self.sim, parameters=Mbsparams, channel_list=Channel
                )
                self.maps = mbs.run_all()[0][Channel.channel]
            else:
                assert (
                    hp.npix2nside(len(maps[0, :])) == self.nside
                ), "wrong nside in the input map!"
                self.maps = maps
                del maps

        if self.correct_in_solver:
            if self.integrate_in_band_solver:
                if mueller_or_jones == "jones":
                    self.h1s, self.h2s, self.betas, self.z1s, self.z2s = np.loadtxt(
                        self.band_filename_solver,
                        usecols=(1, 2, 3, 4, 5),
                        unpack=True,
                        skiprows=1,
                    )
                else:  # mueller_or_jones == "mueller":
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
                    ) = np.loadtxt(
                        self.band_filename_solver,
                        usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                        unpack=True,
                        skiprows=1,
                    )

            else:
                if mueller_or_jones == "jones":
                    if not hasattr(self, "h1s"):
                        self.h1s = 0.0
                    if not hasattr(self, "h2s"):
                        self.h2s = 0.0
                    if not hasattr(self, "betas"):
                        self.betas = 0.0
                    if not hasattr(self, "z1s"):
                        self.z1s = 0.0
                    if not hasattr(self, "z2s"):
                        self.z2s = 0.0
                else:  # mueller_or_jones == "mueller":
                    if not hasattr(self, "mIIs"):
                        self.mIIs = 0.0
                    if not hasattr(self, "mQIs"):
                        self.mQIs = 0.0
                    if not hasattr(self, "mUIs"):
                        self.mUIs = 0.0
                    if not hasattr(self, "mIQs"):
                        self.mIQs = 0.0
                    if not hasattr(self, "mIUs"):
                        self.mIUs = 0.0
                    if not hasattr(self, "mQQs"):
                        self.mQQs = 0.0
                    if not hasattr(self, "mUUs"):
                        self.mUUs = 0.0
                    if not hasattr(self, "mUQs"):
                        self.mUQs = 0.0
                    if not hasattr(self, "mQUs"):
                        self.mQUs = 0.0

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
            del (self.h1, self.h2, self.z1, self.z2, self.beta)
            # Mueller terms of fixed HWP (single/multi freq), to fill A^TA and A^Td
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
            del (self.h1s, self.h2s, self.z1s, self.z2s, self.betas)

    def fill_tod(
        self,
        obs: Observation,
        pointings: np.ndarray,
        hwp_radpsec: float,
    ):
        """It fills tod and/or A^TA and A^Td for the "on the fly" map production
        Args:
            - obs class:`Observations`: container for tod.
                 If the tod is not required, obs.tod can be not allocated
                 i.e. in lbs.Observation allocate_tod=False.
            - pointings (float): pointing for each sample and detector
                 generated by func:lbs.get_pointings
            - hwp_radpsec (float): hwp rotation speed in radiants per second
        """

        times = obs.get_times()

        if self.built_map_on_the_fly:
            self.atd = np.zeros((self.npix, 3))
            self.ata = np.zeros((self.npix, 3, 3))
        else:
            # allocate those for "make_binned_map"
            # later filled
            obs.psi = np.empty_like(obs.tod)
            obs.pixind = np.empty_like(obs.tod, dtype=np.int)

        for idet in range(obs.n_detectors):
            cur_ptg, cur_psi = rotate_coordinates_e2g(
                pointings[idet, :, 0:2], pointings[idet, :, 2]
            )
            # all observed pixels over time (for each sample), i.e. len(pix)==len(times)
            pix = hp.ang2pix(self.nside, cur_ptg[:, 0], cur_ptg[:, 1])
            # theta = hwp_radpsec * times hwp: rotation angle
            # xi: polarization angle, i.e. detector dependent
            # psi: instrument angle, i.e. boresight angle
            xi = compute_polang_from_detquat(obs.quat[idet])
            psi = cur_psi - xi
            del (cur_ptg, cur_psi)
            tod = np.zeros(len(times))

            if self.integrate_in_band:
                integrate_inband_signal_for_one_detector(
                    tod_det=tod,
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
                    theta=times * hwp_radpsec,
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
                    theta=times * hwp_radpsec,
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
                            band=self.cmb2bb,
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
                            theta=times * hwp_radpsec,
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
                            theta=times * hwp_radpsec,
                            psi=psi,
                            xi=xi,
                        )
                else:
                    # re-use ca and sa, factor 4 included here
                    ca = np.cos(2 * pointings[idet, :, 2] + 4 * times * hwp_radpsec)
                    sa = np.sin(2 * pointings[idet, :, 2] + 4 * times * hwp_radpsec)

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
                obs.psi[idet, :] = pointings[idet, :, 2] + 2 * times * hwp_radpsec
                obs.pixind[idet, :] = pix

        del (tod, pix, xi, psi, times, self.maps)
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

    def make_map(self, obss):
        """It generates "on the fly" map. This option is only availabe if
        `built_map_on_the_fly` is set to True.

        Args:
             obss list of class:`Observations`: only necessary for the communicator
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
        if all([obs.comm is None for obs in obss]) or not mpi.MPI_ENABLED:
            # Serial call
            pass
        elif all(
            [
                mpi.MPI.Comm.Compare(obss[i].comm, obss[i + 1].comm) < 2
                for i in range(len(obss) - 1)
            ]
        ):
            self.atd = obss[0].comm.allreduce(self.atd, mpi.MPI.SUM)
            self.ata = obss[0].comm.allreduce(self.ata, mpi.MPI.SUM)
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
