# -*- encoding: utf-8 -*-
import logging
from typing import Union, List

import healpy as hp
import numpy as np
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from numba import njit, prange

import litebird_sim as lbs
from litebird_sim import mpi
from ..coordinates import rotate_coordinates_e2g
from ..detectors import FreqChannelInfo
from ..hwp_diff_emiss import compute_2f_for_one_sample
from ..mbs.mbs import MbsParameters
from ..observations import Observation

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
def compute_orientation_from_detquat(quat):
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
def mueller_interpolation(Theta, harmonic, i, j):
    mueller0deg = {
        "0f": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        "2f": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64),
        "4f": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float64),
    }

    mueller10deg = {
        "0f": np.array(
            [
                [0.961, 8.83 * 1e-5, -7.87 * 1e-6],
                [9.60 * 1e-5, 1.88 * 1e-4, 4.87 * 1e-4],
                [4.39 * 1e-6, -4.63 * 1e-4, 7.48 * 1e-4],
            ],
            dtype=np.float64,
        ),
        "2f": np.array(
            [
                [4.89 * 1e-6, 5.15 * 1e-4, 5.16 * 1e-4],
                [5.43 * 1e-4, 3.10 * 1e-3, 3.28 * 1e-3],
                [5.42 * 1e-4, 2.96 * 1e-3, 3.24 * 1e-3],
            ],
            dtype=np.float64,
        ),
        "4f": np.array(
            [
                [1.09 * 1e-7, 9.26 * 1e-5, 9.25 * 1e-5],
                [8.86 * 1e-5, 0.959, 0.959],
                [8.86 * 1e-5, 0.959, 0.959],
            ],
            dtype=np.float64,
        ),
    }

    f_factor = (
        np.sin(np.deg2rad(0.078 * Theta)) ** 2 / np.sin(np.deg2rad(0.078 * 10)) ** 2
    )

    return (
        mueller0deg[harmonic][i, j]
        + (mueller10deg[harmonic][i, j] - mueller0deg[harmonic][i, j]) * f_factor
    )


@njit
def compute_Tterm_for_one_sample_for_tod(mII, mQI, mUI, cos2Xi2Phi, sin2Xi2Phi):
    Tterm = mII + mQI * cos2Xi2Phi + mUI * sin2Xi2Phi

    return Tterm


@njit
def compute_Qterm_for_one_sample_for_tod(
    mIQ, mQQ, mUU, mIU, mUQ, mQU, psi, phi, cos2Xi2Phi, sin2Xi2Phi
):
    Qterm = np.cos(2 * psi + 2 * phi) * (
        mIQ + mQQ * cos2Xi2Phi + mUQ * sin2Xi2Phi
    ) - np.sin(2 * psi + 2 * phi) * (mIU + mQU * cos2Xi2Phi + mUU * sin2Xi2Phi)

    return Qterm


@njit
def compute_Uterm_for_one_sample_for_tod(
    mIU, mQU, mUQ, mIQ, mQQ, mUU, psi, phi, cos2Xi2Phi, sin2Xi2Phi
):
    Uterm = np.sin(2 * psi + 2 * phi) * (
        mIQ + mQQ * cos2Xi2Phi + mUQ * sin2Xi2Phi
    ) + np.cos(2 * psi + 2 * phi) * (mIU + mQU * cos2Xi2Phi + mUU * sin2Xi2Phi)

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
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    """Bolometric equation, tod filling for a single (time) sample"""
    d = T * compute_Tterm_for_one_sample_for_tod(mII, mQI, mUI, cos2Xi2Phi, sin2Xi2Phi)

    d += Q * compute_Qterm_for_one_sample_for_tod(
        mIQ, mQQ, mUU, mIU, mUQ, mQU, psi, phi, cos2Xi2Phi, sin2Xi2Phi
    )

    d += U * compute_Uterm_for_one_sample_for_tod(
        mIU, mQU, mUQ, mIQ, mQQ, mUU, psi, phi, cos2Xi2Phi, sin2Xi2Phi
    )

    return d


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

    for i in prange(len(tod_det)):
        Four_rho_phi = 4 * (rho[i] - phi)
        Two_rho_phi = 2 * (rho[i] - phi)
        tod_det[i] += compute_signal_for_one_sample(
            T=mapT[i],
            Q=mapQ[i],
            U=mapU[i],
            mII=m0f[0, 0]
            + m2f[0, 0] * np.cos(Two_rho_phi + phases_2f[0, 0])
            + m4f[0, 0] * np.cos(Four_rho_phi + phases_4f[0, 0]),
            mQI=m0f[1, 0]
            + m2f[1, 0] * np.cos(Two_rho_phi + phases_2f[1, 0])
            + m4f[1, 0] * np.cos(Four_rho_phi + phases_4f[1, 0]),
            mUI=m0f[2, 0]
            + m2f[2, 0] * np.cos(Two_rho_phi + phases_2f[2, 0])
            + m4f[2, 0] * np.cos(Four_rho_phi + phases_4f[2, 0]),
            mIQ=m0f[0, 1]
            + m2f[0, 1] * np.cos(Two_rho_phi + phases_2f[0, 1])
            + m4f[0, 1] * np.cos(Four_rho_phi + phases_4f[0, 1]),
            mIU=m0f[0, 2]
            + m2f[0, 2] * np.cos(Two_rho_phi + phases_2f[0, 2])
            + m4f[0, 2] * np.cos(Four_rho_phi + phases_4f[0, 2]),
            mQQ=m0f[1, 1]
            + m2f[1, 1] * np.cos(Two_rho_phi + phases_2f[1, 1])
            + m4f[1, 1] * np.cos(Four_rho_phi + phases_4f[1, 1]),
            mUU=m0f[2, 2]
            + m2f[2, 2] * np.cos(Two_rho_phi + phases_2f[2, 2])
            + m4f[2, 2] * np.cos(Four_rho_phi + phases_4f[2, 2]),
            mUQ=m0f[2, 1]
            + m2f[2, 1] * np.cos(Two_rho_phi + phases_2f[2, 1])
            + m4f[2, 1] * np.cos(Four_rho_phi + phases_4f[2, 1]),
            mQU=m0f[1, 2]
            + m2f[1, 2] * np.cos(Two_rho_phi + phases_2f[1, 2])
            + m4f[1, 2] * np.cos(Four_rho_phi + phases_4f[1, 2]),
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
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
):
    r"""
    Single-frequency case: computes :math:`A^T A` and :math:`A^T d`
    for a single detector, for one (time) sample.
    """
    Tterm = compute_Tterm_for_one_sample_for_tod(
        mIIs, mQIs, mUIs, cos2Xi2Phi, sin2Xi2Phi
    )

    Qterm = compute_Qterm_for_one_sample_for_tod(
        mIQs, mQQs, mUUs, mIUs, mUQs, mQUs, psi, phi, cos2Xi2Phi, sin2Xi2Phi
    )
    Uterm = compute_Uterm_for_one_sample_for_tod(
        mIUs, mQUs, mUQs, mIQs, mQQs, mUUs, psi, phi, cos2Xi2Phi, sin2Xi2Phi
    )

    return Tterm, Qterm, Uterm


@njit
def compute_ata_atd_for_one_detector(
    ata,
    atd,
    tod,
    m0f_solver,
    m2f_solver,
    m4f_solver,
    pixel_ind,
    rho,
    psi,
    phi,
    cos2Xi2Phi,
    sin2Xi2Phi,
    phases_2f,
    phases_4f,
):
    r"""
    Single-frequency case: compute :math:`A^T A` and :math:`A^T d`
    for a single detector, looping over (time) samples.
    """

    for i in prange(len(tod)):
        Four_rho_phi = 4 * (rho[i] - phi)
        Two_rho_phi = 2 * (rho[i] - phi)
        Tterm, Qterm, Uterm = compute_TQUsolver_for_one_sample(
            mIIs=m0f_solver[0, 0]
            + m2f_solver[0, 0] * np.cos(Two_rho_phi + phases_2f[0, 0])
            + m4f_solver[0, 0] * np.cos(Four_rho_phi + phases_4f[0, 0]),
            mQIs=m0f_solver[1, 0]
            + m2f_solver[1, 0] * np.cos(Two_rho_phi + phases_2f[1, 0])
            + m4f_solver[1, 0] * np.cos(Four_rho_phi + phases_4f[1, 0]),
            mUIs=m0f_solver[2, 0]
            + m2f_solver[2, 0] * np.cos(Two_rho_phi + phases_2f[2, 0])
            + m4f_solver[2, 0] * np.cos(Four_rho_phi + phases_4f[2, 0]),
            mIQs=m0f_solver[0, 1]
            + m2f_solver[0, 1] * np.cos(Two_rho_phi + phases_2f[0, 1])
            + m4f_solver[0, 1] * np.cos(Four_rho_phi + phases_4f[0, 1]),
            mIUs=m0f_solver[0, 2]
            + m2f_solver[0, 2] * np.cos(Two_rho_phi + phases_2f[0, 2])
            + m4f_solver[0, 2] * np.cos(Four_rho_phi + phases_4f[0, 2]),
            mQQs=m0f_solver[1, 1]
            + m2f_solver[1, 1] * np.cos(Two_rho_phi + phases_2f[1, 1])
            + m4f_solver[1, 1] * np.cos(Four_rho_phi + phases_4f[1, 1]),
            mUUs=m0f_solver[2, 2]
            + m2f_solver[2, 2] * np.cos(Two_rho_phi + phases_2f[2, 2])
            + m4f_solver[2, 2] * np.cos(Four_rho_phi + phases_4f[2, 2]),
            mUQs=m0f_solver[2, 1]
            + m2f_solver[2, 1] * np.cos(Two_rho_phi + phases_2f[2, 1])
            + m4f_solver[2, 1] * np.cos(Four_rho_phi + phases_4f[2, 1]),
            mQUs=m0f_solver[1, 2]
            + m2f_solver[1, 2] * np.cos(Two_rho_phi + phases_2f[1, 2])
            + m4f_solver[1, 2] * np.cos(Four_rho_phi + phases_4f[1, 2]),
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
        nside_out: Union[int, None] = None,
        mbs_params: Union[MbsParameters, None] = None,
        build_map_on_the_fly: Union[bool, None] = False,
        apply_non_linearity: Union[bool, None] = False,
        add_2f_hwpss: Union[bool, None] = False,
        interpolation: Union[str, None] = "",
        channel: Union[FreqChannelInfo, None] = None,
        maps: Union[np.ndarray, None] = None,
        comm: Union[bool, None] = None,
        mueller_phases: Union[dict, None] = None,
    ):
        r"""It sets the input paramters reading a dictionary `sim.parameters`
        with key "hwp_sys" and the following input arguments

        Args:
          nside (integer): nside used in the analysis
          nside_out (integer): nside for the output maps. If not provided, same as nside
          mbs_params (:class:`.Mbs`): an instance of the :class:`.Mbs` class
          build_map_on_the_fly (bool): fills :math:`A^T A` and :math:`A^T d`
          apply_non_linearity (bool): applies the coupling of the non-linearity
              systematics with hwp_sys
          add_2f_hwpss (bool): adds the 2f hwpss signal to the TOD
          interpolation (str): if it is ``""`` (the default), pixels in the map
              won’t be interpolated. If it is ``linear``, a linear interpolation
              will be used
          channel (:class:`.FreqChannelInfo`): an instance of the
                                                :class:`.FreqChannelInfo` class
          maps (float): input maps (3, npix) coherent with nside provided,
              Input maps needs to be in galactic (mbs default)
              if `maps` is not None, `mbs_params` is ignored
              (i.e. input maps are not generated)
          comm (SerialMpiCommunicator): MPI communicator
        """

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
            self.nside_out = paramdict.get("nside_out", False)

            assert self.nside_out <= self.nside, (
                f"Error, {self.nside_out=} cannot be larger than {self.nside=}"
            )

            self.build_map_on_the_fly = paramdict.get("build_map_on_the_fly", False)

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
        if nside is None:
            self.nside = 512
        else:
            self.nside = nside

        if nside_out is None:
            self.nside_out = self.nside
        else:
            self.nside_out = nside_out

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

        if not hasattr(self, "build_map_on_the_fly"):
            if build_map_on_the_fly is not None:
                self.build_map_on_the_fly = build_map_on_the_fly

        if not hasattr(self, "apply_non_linearity"):
            if apply_non_linearity is not None:
                self.apply_non_linearity = apply_non_linearity

        if not hasattr(self, "add_2f_hwpss"):
            if add_2f_hwpss is not None:
                self.add_2f_hwpss = add_2f_hwpss

        if not hasattr(self, "comm"):
            if comm is not None:
                self.comm = comm

        if mbs_params is None and np.any(maps) is None:
            mbs_params = lbs.MbsParameters(
                make_cmb=hwp_sys_Mbs_make_cmb,
                make_fg=hwp_sys_Mbs_make_fg,
                fg_models=hwp_sys_Mbs_fg_models,
                gaussian_smooth=hwp_sys_Mbs_gaussian_smooth,
                bandpass_int=False,
                maps_in_ecliptic=False,
                nside=self.nside,
            )

        if np.any(maps) is None:
            mbs_params.nside = self.nside

        self.npix = hp.nside2npix(self.nside)
        self.npix_out = hp.nside2npix(self.nside_out)

        self.interpolation = interpolation

        if channel is None:
            channel = lbs.FreqChannelInfo(bandcenter_ghz=140)

        if np.any(maps) is None:
            mbs = lbs.Mbs(
                simulation=self.sim, parameters=mbs_params, channel_list=channel
            )
            self.maps = mbs.run_all()[0][
                f"{channel.channel.split()[0]}_{channel.channel.split()[1]}"
            ]
        else:
            self.maps = maps

            del maps

        if self.build_map_on_the_fly:
            self.atd = np.zeros((self.npix_out, 3), dtype=np.float64)
            self.ata = np.zeros((self.npix_out, 3, 3), dtype=np.float64)

        if mueller_phases is not None:
            self.mueller_phases = mueller_phases
        else:
            # (temporary solution) using phases from Patanchon et al 2021 as the default.
            self.mueller_phases = {
                "2f": np.array(
                    [[-2.32, -0.49, -2.06], [2.86, -0.25, -2.00], [1.29, -2.01, 2.54]],
                    dtype=np.float64,
                ),
                "4f": np.array(
                    [
                        [-0.84, -0.04, -1.61],
                        [0.14, -0.00061, -0.00056 - np.pi / 2],
                        [-1.43, -0.00070 - np.pi / 2, np.pi - 0.00065],
                    ],
                    dtype=np.float64,
                ),
            }

    def fill_tod(
        self,
        observations: Union[Observation, List[Observation]] = None,
        pointings: Union[np.ndarray, List[np.ndarray], None] = None,
        hwp_angle: Union[np.ndarray, List[np.ndarray], None] = None,
        input_map_in_galactic: bool = True,
        save_tod: bool = False,
        dtype_pointings=np.float64,
    ):
        r"""Fill a TOD and/or :math:`A^T A` and :math:`A^T d` for the
        "on-the-fly" map production

        Args:
            observations (:class:`Observation`): container for the
                TOD. If the TOD is not required, you can avoid
                allocating ``observations.tod`` by setting
                ``allocate_tod=False`` in :class:`.Observation`.

            pointings (optional): if not present, it is either computed
                on the fly (generated by :func:`lbs.get_pointings` per
                detector), or read from
                ``observations.pointing_matrix`` (if present).

                If ``observations`` is not a list, ``pointings`` must
                be a np.array of shape ``(N_det, N_samples, 3)``. If
                ``observations`` is a list, ``pointings`` must be a
                list of same length.

            hwp_angle (optional): `2ωt`, hwp rotation angles
                (radians). If ``pointings`` is passed, ``hwp_angle``
                must be passed as well, otherwise both must be
                ``None``. If not passed, it is computed on the fly
                (generated by :func:`lbs.get_pointings` per detector).
                If ``observations`` is not a list, ``hwp_angle`` must
                be a np.array of dimensions (N_samples).

                If ``observations`` is a list, ``hwp_angle`` must be a
                list of same length.

            input_map_in_galactic (bool): if True, the input map is in
                galactic coordinates, pointings are rotated from
                ecliptic to galactic and output map will also be in
                galactic.

            save_tod (bool): if True, ``obs.tod`` is saved in
                ``observations.tod`` and locally as a .npy file;
                if False, ``obs.tod`` gets deleted.

            dtype_pointings: if ``pointings`` is None and is computed
                within ``fill_tod``, this is the dtype for
                pointings and tod (default: np.float32).

        """

        assert observations is not None, (
            "You need to pass at least one observation to fill_tod."
        )

        if pointings is None:
            if hwp_angle:
                raise Warning(
                    "You passed hwp_angle, but you did not pass pointings, "
                    + "so hwp_angle will be ignored and re-computed on the fly."
                )

            if isinstance(observations, Observation):
                obs_list = [observations]
                if hasattr(observations, "pointing_matrix"):
                    ptg_list = [observations.pointing_matrix]
                else:
                    ptg_list = []
                if hasattr(observations, "hwp_angle"):
                    hwp_angle_list = [observations.hwp_angle]
                else:
                    hwp_angle_list = []

            else:
                obs_list = observations
                ptg_list = []
                hwp_angle_list = []
                for ob in observations:
                    if hasattr(ob, "pointing_matrix"):
                        ptg_list.append(ob.pointing_matrix)
                    if hasattr(ob, "hwp_angle"):
                        hwp_angle_list.append(ob.hwp_angle)

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
            if not self.build_map_on_the_fly:
                # allocate those for "make_binned_map", later filled
                if not hasattr(cur_obs, "pointing_matrix"):
                    cur_obs.pointing_matrix = np.empty(
                        (cur_obs.n_detectors, cur_obs.n_samples, 3),
                        dtype=dtype_pointings,
                    )

            for idet in range(cur_obs.n_detectors):
                # if no mueller_hwp has been set, it will be the ideal one from hwp.py,
                # we must turn it into the rotation harmonics one (for the ideal case)
                if (cur_obs.mueller_hwp[idet] == np.diag([1.0, 1.0, -1.0, -1.0])).all():
                    cur_obs.mueller_hwp[idet] = {
                        "0f": np.array(
                            [[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64
                        ),
                        "2f": np.array(
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64
                        ),
                        "4f": np.array(
                            [[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float64
                        ),
                    }

                if cur_obs.mueller_hwp_solver[idet] is None:
                    cur_obs.mueller_hwp_solver[idet] = {
                        "0f": np.array(
                            [[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64
                        ),
                        "2f": np.array(
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64
                        ),
                        "4f": np.array(
                            [[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float64
                        ),
                    }

                tod = cur_obs.tod[idet, :]

                if pointings is None and ((not ptg_list) or (not hwp_angle_list)):
                    cur_point, cur_hwp_angle = cur_obs.get_pointings(
                        detector_idx=idet, pointings_dtype=dtype_pointings
                    )
                    cur_point = cur_point.reshape(-1, 3)
                else:
                    cur_point = ptg_list[idx_obs][idet, :, :]
                    cur_hwp_angle = hwp_angle_list[idx_obs]

                # rotating pointing from ecliptic to galactic as the input map
                if input_map_in_galactic:
                    cur_point = rotate_coordinates_e2g(cur_point)

                # all observed pixels over time (for each sample),
                # i.e. len(pix)==len(times)
                if self.interpolation in ["", None]:
                    pix = hp.ang2pix(self.nside, cur_point[:, 0], cur_point[:, 1])

                if self.build_map_on_the_fly:
                    pix_out = hp.ang2pix(
                        self.nside_out, cur_point[:, 0], cur_point[:, 1]
                    )

                if self.interpolation in ["", None]:
                    input_T = self.maps[0, pix]
                    input_Q = self.maps[1, pix]
                    input_U = self.maps[2, pix]

                elif self.interpolation == "linear":
                    input_T = hp.get_interp_val(
                        self.maps[0, :],
                        cur_point[:, 0],
                        cur_point[:, 1],
                    )
                    input_Q = hp.get_interp_val(
                        self.maps[1, :],
                        cur_point[:, 0],
                        cur_point[:, 1],
                    )
                    input_U = hp.get_interp_val(
                        self.maps[2, :],
                        cur_point[:, 0],
                        cur_point[:, 1],
                    )
                else:
                    raise ValueError(
                        "Wrong value for interpolation. It should be one of the following:\n"
                        + '- "" for no interpolation\n'
                        + '- "linear" for linear interpolation\n'
                    )

                # separating polarization angle xi from cur_point[:, 2] = psi + xi
                # xi: polarization angle, i.e. detector dependent
                # psi: instrument angle, i.e. boresight direction from focal plane POV
                xi = cur_obs.pol_angle_rad[idet]
                psi = cur_point[:, 2]

                phi = np.deg2rad(cur_obs.pointing_theta_phi_psi_deg[idet][1])

                cos2Xi2Phi = np.cos(2 * xi - 2 * phi)
                sin2Xi2Phi = np.sin(2 * xi - 2 * phi)

                compute_signal_for_one_detector(
                    tod_det=tod,
                    m0f=cur_obs.mueller_hwp[idet]["0f"],
                    m2f=cur_obs.mueller_hwp[idet]["2f"],
                    m4f=cur_obs.mueller_hwp[idet]["4f"],
                    rho=np.array(cur_hwp_angle, dtype=np.float64),
                    psi=np.array(psi, dtype=np.float64),
                    mapT=input_T,
                    mapQ=input_Q,
                    mapU=input_U,
                    cos2Xi2Phi=cos2Xi2Phi,
                    sin2Xi2Phi=sin2Xi2Phi,
                    phi=phi,
                    apply_non_linearity=self.apply_non_linearity,
                    g_one_over_k=cur_obs.g_one_over_k[idet],
                    add_2f_hwpss=self.add_2f_hwpss,
                    amplitude_2f_k=cur_obs.amplitude_2f_k[idet],
                    phases_2f=self.mueller_phases["2f"],
                    phases_4f=self.mueller_phases["4f"],
                )

                if self.build_map_on_the_fly:
                    compute_ata_atd_for_one_detector(
                        ata=self.ata,
                        atd=self.atd,
                        tod=tod,
                        m0f_solver=cur_obs.mueller_hwp_solver[idet]["0f"],
                        m2f_solver=cur_obs.mueller_hwp_solver[idet]["2f"],
                        m4f_solver=cur_obs.mueller_hwp_solver[idet]["4f"],
                        pixel_ind=pix_out,
                        rho=np.array(cur_hwp_angle, dtype=np.float64),
                        psi=np.array(psi, dtype=np.float64),
                        phi=phi,
                        cos2Xi2Phi=cos2Xi2Phi,
                        sin2Xi2Phi=sin2Xi2Phi,
                        phases_2f=self.mueller_phases["2f"],
                        phases_4f=self.mueller_phases["4f"],
                    )

                cur_obs.tod[idet] = tod

        if self.interpolation in ["", None]:
            del pix
        del input_T, input_Q, input_U
        if not save_tod:
            del cur_obs.tod

    def make_map(self, observations):
        """It generates "on the fly" map. This option is only availabe if
        `build_map_on_the_fly` is set to True.

        Args:
             observations list of class:`Observations`: only necessary for the communicator
             pointings (float): pointing for each sample and detector
                 generated by lbs.get_pointings
             hwp_radpsec (float): hwp rotation speed in radiants per second
        Returns:
            map (float): rebinned T,Q,U maps
        """

        assert self.build_map_on_the_fly, (
            "make_map available only with build_map_on_the_fly option activated"
        )
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
            self.comm.Allreduce(mpi.MPI.IN_PLACE, self.atd, mpi.MPI.SUM)
            self.comm.Allreduce(mpi.MPI.IN_PLACE, self.ata, mpi.MPI.SUM)
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
