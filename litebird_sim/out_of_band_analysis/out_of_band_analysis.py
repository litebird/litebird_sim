# -*- encoding: utf-8 -*-
from numba import njit
import numpy as np

import healpy as hp
from astropy import constants as const
from astropy.cosmology import Planck18_arXiv_v2 as cosmo

from typing import Union, List
from pathlib import Path

import litebird_sim as lbs
from litebird_sim import mpi
from ..mbs.mbs import MbsParameters
from ..detectors import FreqChannelInfo
from ..observations import Observation
from .bandpass_template_module import bandpass_profile
from ..coordinates import rotate_coordinates_e2g, CoordinateSystem

COND_THRESHOLD = 1e10


@njit
def compute_Tterm_for_one_sample(h1, h2, cb, z1, z2, co, si):
    Tterm = 0.25 * (
        2
        + h1 * (2 + h1)
        + h2 * (2 + h2)
        + z1 * z1
        + z2 * z2
        + 4 * (-((1 + h1) * z2) + (1 + h2) * z1 * cb) * co * si
        + (h1 * (2 + h1) - h2 * (2 + h2) + (z1 - z2) * (z1 + z2)) * (co**2 - si**2)
    )
    return Tterm


@njit
def compute_Qterm_for_one_sample(h1, h2, cb, z1, z2, co, si):
    Qterm = 0.125 * (
        2
        + h1 * (2 + h1)
        + h2 * (2 + h2)
        - (z1 - z2) * (z1 - z2)
        + 2
        * (
            h1 * (2 + h1)
            - h2 * (2 + h2)
            - z1 * z1
            + z2 * z2
            - 4 * (z1 + z2) * (1 + h1 + cb + h2 * cb) * co * si
        )
        * (co**2 - si**2)
        + (2 + h1 * (2 + h1) + h2 * (2 + h2) - (z1 + z2) * (z1 + z2))
        * (co**4 - 6 * co**2 * si**2 + si**4)
        + 8
        * co
        * si
        * (-((1 + h1) * z1) - (1 + h2) * cb * (-z2 + 2 * (1 + h1) * co * si))
    )
    return Qterm


@njit
def compute_Uterm_for_one_sample(h1, h2, cb, z1, z2, co, si):
    Uterm = (
        (1 + h1) * z1 * co**4
        + ((1 + h1 - z1) * (1 + h1 + z1) - z1 * z2 + (1 + h1) * (1 + h2) * cb)
        * co**3
        * si
        - (
            (1 + h1) * z1
            + 2 * (1 + h1) * z2
            + 2 * (1 + h2) * z1 * cb
            + (1 + h2) * z2 * cb
        )
        * co**2
        * si**2
        - (-z1 * z2 + (1 + h2 - z2) * (1 + h2 + z2) + (1 + h1) * (1 + h2) * cb)
        * co
        * si**3
        + (1 + h2) * z2 * cb * si**4
    )
    return Uterm


@njit
def compute_signal_for_one_sample(T, Q, U, h1, h2, cb, z1, z2, co, si):
    """Bolometric equation"""
    d = T * compute_Tterm_for_one_sample(h1, h2, cb, z1, z2, co, si)
    d += Q * compute_Qterm_for_one_sample(h1, h2, cb, z1, z2, co, si)
    d += U * compute_Uterm_for_one_sample(h1, h2, cb, z1, z2, co, si)
    return d


@njit
def integrate_in_band_signal_for_one_sample(T, Q, U, band, h1, h2, cb, z1, z2, co, si):
    tod = 0
    for i in range(len(band)):
        tod += band[i] * compute_signal_for_one_sample(
            T[i], Q[i], U[i], h1[i], h2[i], cb[i], z1[i], z2[i], co, si
        )

    return tod


@njit
def compute_signal_for_one_detector(
    tod_det, h1, h2, cb, z1, z2, pixel_ind, polangle, maps
):

    for i in range(len(tod_det)):
        tod_det[i] += compute_signal_for_one_sample(
            T=maps[0, pixel_ind[i]],
            Q=maps[1, pixel_ind[i]],
            U=maps[2, pixel_ind[i]],
            h1=h1,
            h2=h2,
            cb=cb,
            z1=z1,
            z2=z2,
            co=np.cos(polangle[i]),
            si=np.sin(polangle[i]),
        )


@njit
def integrate_in_band_signal_for_one_detector(
    tod_det, band, h1, h2, cb, z1, z2, pixel_ind, polangle, maps
):

    for i in range(len(tod_det)):
        tod_det[i] += integrate_in_band_signal_for_one_sample(
            T=maps[:, 0, pixel_ind[i]],
            Q=maps[:, 1, pixel_ind[i]],
            U=maps[:, 2, pixel_ind[i]],
            band=band,
            h1=h1,
            h2=h2,
            cb=cb,
            z1=z1,
            z2=z2,
            co=np.cos(polangle[i]),
            si=np.sin(polangle[i]),
        )


@njit
def compute_mueller_for_one_sample(h1, h2, cb, z1, z2, co, si):
    Tterm = compute_Tterm_for_one_sample(h1, h2, cb, z1, z2, co, si)
    Qterm = compute_Qterm_for_one_sample(h1, h2, cb, z1, z2, co, si)
    Uterm = compute_Uterm_for_one_sample(h1, h2, cb, z1, z2, co, si)
    return Tterm, Qterm, Uterm


@njit
def integrate_in_band_mueller_for_one_sample(band, h1, h2, cb, z1, z2, co, si):
    intTterm = 0
    intQterm = 0
    intUterm = 0
    for i in range(len(band)):
        Tterm, Qterm, Uterm = compute_mueller_for_one_sample(
            h1[i], h2[i], cb[i], z1[i], z2[i], co, si
        )
        intTterm += band[i] * Tterm
        intQterm += band[i] * Qterm
        intUterm += band[i] * Uterm

    return intTterm, intQterm, intUterm


@njit
def compute_atd_ata_for_one_detector(
    atd,
    ata,
    tod,
    h1,
    h2,
    cb,
    z1,
    z2,
    pixel_ind,
    polangle,
):

    for i in range(len(tod)):
        Tterm, Qterm, Uterm = compute_mueller_for_one_sample(
            h1=h1,
            h2=h2,
            cb=cb,
            z1=z1,
            z2=z2,
            co=np.cos(polangle[i]),
            si=np.sin(polangle[i]),
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
def integrate_in_band_atd_ata_for_one_detector(
    atd,
    ata,
    tod,
    band,
    h1,
    h2,
    cb,
    z1,
    z2,
    pixel_ind,
    polangle,
):

    for i in range(len(tod)):
        Tterm, Qterm, Uterm = integrate_in_band_mueller_for_one_sample(
            band=band,
            h1=h1,
            h2=h2,
            cb=cb,
            z1=z1,
            z2=z2,
            co=np.cos(polangle[i]),
            si=np.sin(polangle[i]),
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


def _dBodTrj(nu):
    """Radiance to Rayleigh-Jeans conversion factor
    nu: frequency in Ghz
    """
    return 2 * const.k_B.value * nu * nu * 1e18 / const.c.value / const.c.value


def _dBodTth(nu):
    """Radiance to CMB-units conversion factor (dB/dT)
    nu: frwquency in GHz
    """

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


class HwpSysAndBandpass:
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
        Mbsparams: Union[MbsParameters, None] = None,
        integrate_in_band: Union[bool, None] = None,
        built_map_on_the_fly: Union[bool, None] = None,
        correct_in_solver: Union[bool, None] = None,
        integrate_in_band_solver: Union[bool, None] = None,
        Channel: Union[FreqChannelInfo, None] = None,
        maps: Union[np.ndarray, None] = None,
    ):
        """It sets the input paramters

        Args:
             nside (integer): nside used in the analysis
             Mbsparams (:class:`.Mbs`): an instance of the :class:`.Mbs` class.
             Input maps needs to be in galactic (mbs default)
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
        hwp_sys_Mbs_fg_models = [
            "pysm_synch_1",
            "pysm_freefree_1",
            "pysm_dust_1",
            "pysm_ame_1",
        ]
        hwp_sys_Mbs_gaussian_smooth = True

        # This part sets from parameter file
        if (self.sim.parameter_file is not None) and (
            "hwp_sys" in self.sim.parameter_file.keys()
        ):
            paramdict = self.sim.parameter_file["hwp_sys"]

            self.nside = paramdict.get("nside", False)
            if "general" in self.sim.parameter_file.keys():
                if "nside" in self.sim.parameter_file["general"].keys():
                    if self.sim.parameter_file["general"]["nside"] != self.nside:
                        print(
                            "Warning!! nside from general "
                            "(=%i) and hwp_sys (=%i) do not match. Using hwp_sys"
                            % (
                                self.sim.parameter_file["general"]["nside"],
                                self.nside,
                            )
                        )

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

            self.h1 = paramdict.get("h1", False)
            self.h2 = paramdict.get("h2", False)
            self.beta = paramdict.get("beta", False)
            self.z1 = paramdict.get("z1", False)
            self.z2 = paramdict.get("z2", False)

            self.h1s = paramdict.get("h1s", False)
            self.h2s = paramdict.get("h2s", False)
            self.betas = paramdict.get("betas", False)
            self.z1s = paramdict.get("z1s", False)
            self.z2s = paramdict.get("z2s", False)

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
        if not self.nside:
            if nside is None:
                self.nside = 512
            else:
                self.nside = nside

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

        if Channel is None:
            Channel = lbs.FreqChannelInfo(bandcenter_ghz=100)

        if Mbsparams is None:
            Mbsparams = lbs.MbsParameters(
                make_cmb=hwp_sys_Mbs_make_cmb,
                make_fg=hwp_sys_Mbs_make_fg,
                fg_models=hwp_sys_Mbs_fg_models,
                gaussian_smooth=hwp_sys_Mbs_gaussian_smooth,
                bandpass_int=False,
                maps_in_ecliptic=False,
                save=False,
                output_string=Channel.channel,
            )

        Mbsparams.nside = self.nside

        self.npix = hp.nside2npix(self.nside)

        if self.integrate_in_band:
            try:
                self.freqs, self.h1, self.h2, self.beta, self.z1, self.z2 = np.loadtxt(
                    self.band_filename, unpack=True, skiprows=1
                )
            except Exception:
                print("you have not provided a band_filename in the parameter file!")

            self.nfreqs = len(self.freqs)

            if not self.bandpass:

                self.cmb2bb = _dBodTth(self.freqs)

            elif self.bandpass:

                self.freqs, self.bandpass_profile = bandpass_profile(
                    self.freqs, self.bandpass, self.include_beam_throughput
                )

                self.cmb2bb = _dBodTth(self.freqs) * self.bandpass_profile

            # Normalize the band
            self.cmb2bb /= self.cmb2bb.sum()

            myinstr = {}
            for ifreq in range(self.nfreqs):
                myinstr["ch" + str(ifreq)] = {
                    "bandcenter_ghz": self.freqs[ifreq],
                    "bandwidth_ghz": 0,
                    "fwhm_arcmin": Channel.fwhm_arcmin,
                    "p_sens_ukarcmin": 0.0,
                }

            mbs = lbs.Mbs(simulation=self.sim, parameters=Mbsparams, instrument=myinstr)

            maps = mbs.run_all()[0]
            self.maps = np.empty((self.nfreqs, 3, self.npix))
            for ifreq in range(self.nfreqs):
                self.maps[ifreq] = maps["ch" + str(ifreq)]
            del maps

        else:

            if not self.h1:
                self.h1 = 0.0
            if not self.h2:
                self.h2 = 0.0
            if not self.beta:
                self.beta = 0.0
            if not self.z1:
                self.z1 = 0.0
            if not self.z2:
                self.z2 = 0.0

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

        if self.correct_in_solver:
            if self.integrate_in_band_solver:
                try:
                    (
                        self.freqs,
                        self.h1s,
                        self.h2s,
                        self.betas,
                        self.z1s,
                        self.z2s,
                    ) = np.loadtxt(self.band_filename_solver, unpack=True, skiprows=1)
                except Exception:
                    print(
                        "you have not provided a band_filename_solver"
                        + "in the parameter file!"
                    )

            else:
                if not self.h1s:
                    self.h1s = 0.0
                if not self.h2s:
                    self.h2s = 0.0
                if not self.betas:
                    self.betas = 0.0
                if not self.z1s:
                    self.z1s = 0.0
                if not self.z2s:
                    self.z2s = 0.0

            if not self.bandpass_solver:

                self.cmb2bb_solver = _dBodTth(self.freqs)

            elif self.bandpass_solver:

                self.freqs_solver, self.bandpass_profile_solver = bandpass_profile(
                    self.freqs, self.bandpass_solver, self.include_beam_throughput
                )
                self.cmb2bb_solver = (
                    _dBodTth(self.freqs_solver) * self.bandpass_profile_solver
                )

            self.cmb2bb_solver /= self.cmb2bb_solver.sum()

        self.cbeta = np.cos(np.deg2rad(self.beta))
        self.cbetas = np.cos(np.deg2rad(self.betas))

    def fill_tod(
        self,
        obs: Union[Observation, List[Observation]],
        hwp_radpsec: float,
        pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    ):
        """It fills tod and/or A^TA and A^Td for the "on the fly" map production

        Args:
             obs class:`Observations`: container for tod.
                 If the tod is not required, obs.tod can be not allocated
                 i.e. in lbs.Observation allocate_tod=False.
             pointings (float): pointing for each sample and detector
                 generated by func:lbs.get_pointings_for_observations. Don't add HWP
                 rotation angle to the polarization angle, it will be added here.
             hwp_radpsec (float): hwp rotation speed in radiants per second
        """
        if pointings is None:
            if isinstance(obs, Observation):
                obs_list = [obs]
                ptg_list = [obs.pointings]
                psi_list = [obs.psi]
            else:
                obs_list = obs
                ptg_list = [ob.pointings for ob in obs]
                psi_list = [ob.psi for ob in obs]
        else:
            if isinstance(obs, Observation):
                assert isinstance(pointings, np.ndarray), (
                    "You must pass a list of observations *and* a list "
                    + "of pointing matrices to scan_map_in_observations"
                )
                obs_list = [obs]
                ptg_list = [pointings[:, :, 0:2]]
                psi_list = [pointings[:, :, 2]]
            else:
                assert isinstance(pointings, list), (
                    "When you pass a list of observations to scan_map_in_observations, "
                    + "you must do the same for `pointings`"
                )
                assert len(obs) == len(pointings), (
                    f"The list of observations has {len(obs)} elements, but "
                    + f"the list of pointings has {len(pointings)} elements"
                )
                obs_list = obs
                ptg_list = [point[:, :, 0:2] for point in pointings]
                psi_list = [point[:, :, 2] for point in pointings]

        for cur_obs, cur_point, cur_Psi in zip(obs_list, ptg_list, psi_list):

            times = cur_obs.get_times()

            if self.built_map_on_the_fly:
                self.atd = np.zeros((self.npix, 3))
                self.ata = np.zeros((self.npix, 3, 3))
            else:
                # allocate those for "make_bin_map"
                # later filled
                cur_obs.psi = np.empty_like(cur_obs.tod)
                cur_obs.pixind = np.empty_like(cur_obs.tod, dtype=np.int)

            for idet in range(cur_obs.n_detectors):
                # rotating pointing from ecliptic to galactic as input map
                cur_ptg, cur_psi = rotate_coordinates_e2g(
                    cur_point[idet, :, :], cur_Psi[idet, :]
                )

                pix = hp.ang2pix(self.nside, cur_ptg[:, 0], cur_ptg[:, 1])

                tod = cur_obs.tod[idet, :]

                if self.integrate_in_band:
                    integrate_in_band_signal_for_one_detector(
                        tod_det=tod,
                        band=self.cmb2bb,
                        h1=self.h1,
                        h2=self.h2,
                        cb=self.cbeta,
                        z1=self.z1,
                        z2=self.z2,
                        pixel_ind=pix,
                        polangle=0.5 * cur_psi + times * hwp_radpsec,
                        maps=self.maps,
                    )
                else:
                    compute_signal_for_one_detector(
                        tod_det=tod,
                        h1=self.h1,
                        h2=self.h2,
                        cb=self.cbeta,
                        z1=self.z1,
                        z2=self.z2,
                        pixel_ind=pix,
                        polangle=0.5 * cur_psi + times * hwp_radpsec,
                        maps=self.maps,
                    )

                if self.built_map_on_the_fly:
                    if self.correct_in_solver:
                        if self.integrate_in_band_solver:
                            integrate_in_band_atd_ata_for_one_detector(
                                atd=self.atd,
                                ata=self.ata,
                                tod=tod,
                                band=self.cmb2bb_solver,
                                h1=self.h1s,
                                h2=self.h2s,
                                cb=self.cbetas,
                                z1=self.z1s,
                                z2=self.z2s,
                                pixel_ind=pix,
                                polangle=0.5 * cur_psi + times * hwp_radpsec,
                            )
                        else:
                            compute_atd_ata_for_one_detector(
                                atd=self.atd,
                                ata=self.ata,
                                tod=tod,
                                h1=self.h1s,
                                h2=self.h2s,
                                cb=self.cbetas,
                                z1=self.z1s,
                                z2=self.z2s,
                                pixel_ind=pix,
                                polangle=0.5 * cur_psi + times * hwp_radpsec,
                            )

                    else:
                        # in this case factor 4 included here
                        ca = np.cos(2 * cur_psi + 4 * times * hwp_radpsec)
                        sa = np.sin(2 * cur_psi + 4 * times * hwp_radpsec)

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

                    # del tod

                else:
                    # this fills variables needed by bin_map
                    cur_obs.psi[idet, :] = cur_psi + 2 * times * hwp_radpsec
                    cur_obs.pixind[idet, :] = pix

            return

    def make_map(self, obss):
        """It generates "on the fly" map. This option is only availabe if `built_map_on_the_fly`
        is set to True.

        Args:
             obss list of class:`Observations`: only necessary for the communicator
             pointings (float): pointing for each sample and detector
                 generated by lbs.get_pointings_for_observations
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
