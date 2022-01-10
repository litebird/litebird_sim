# -*- encoding: utf-8 -*-
import litebird_sim as lbs
import numpy as np
import healpy as hp
from astropy import constants as const
from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from litebird_sim import mpi
from typing import Union, List
from ..mbs.mbs import MbsParameters
from ..detectors import FreqChannelInfo
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
        if (self.sim.parameter_file is not None) and (
            "hwp_sys" in self.sim.parameter_file.keys()
        ):
            paramdict = self.sim.parameter_file["hwp_sys"]

            if "nside" in paramdict.keys():
                self.nside = paramdict["nside"]
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

            if "integrate_in_band" in paramdict.keys():
                self.integrate_in_band = paramdict["integrate_in_band"]

            if "built_map_on_the_fly" in paramdict.keys():
                self.built_map_on_the_fly = paramdict["built_map_on_the_fly"]

            if "correct_in_solver" in paramdict.keys():
                self.correct_in_solver = paramdict["correct_in_solver"]

            if "integrate_in_band_solver" in paramdict.keys():
                self.integrate_in_band_solver = paramdict["integrate_in_band_solver"]

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
            self.freqs, self.h1, self.h2, self.beta, self.z1, self.z2 = np.loadtxt(
                self.band_filename, unpack=True, skiprows=1
            )

            self.nfreqs = len(self.freqs)

            self.cmb2bb = _dBodTth(self.freqs)
            self.norm = self.cmb2bb.sum()

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
                self.h1s, self.h2s, self.betas, self.z1s, self.z2s = np.loadtxt(
                    self.band_filename_solver,
                    usecols=(1, 2, 3, 4, 5),
                    unpack=True,
                    skiprows=1,
                )
            else:
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

    def fill_tod(self, obs: Observation, pointings: np.ndarray, hwp_radpsec: float):

        """It fills tod and/or A^TA and A^Td for the "on the fly" map production

        Args:
             obs class:`Observations`: container for tod.
                 If the tod is not required, obs.tod can be not allocated
                 i.e. in lbs.Observation allocate_tod=False.
             pointings (float): pointing for each sample and detector
                 generated by func:lbs.scanning.get_pointings
             hwp_radpsec (float): hwp rotation speed in radiants per second
        """

        times = obs.get_times()

        if self.built_map_on_the_fly:
            self.atd = np.zeros((self.npix, 3))
            self.ata = np.zeros((self.npix, 3, 3))
        else:
            # allocate those for "make_bin_map"
            # later filled
            obs.psi = np.empty_like(obs.tod)
            obs.pixind = np.empty_like(obs.tod, dtype=np.int)

        for idet in range(obs.n_detectors):
            pix = hp.ang2pix(self.nside, pointings[idet, :, 0], pointings[idet, :, 1])

            # add hwp rotation
            ca = np.cos(0.5 * pointings[idet, :, 2] + times * hwp_radpsec)
            sa = np.sin(0.5 * pointings[idet, :, 2] + times * hwp_radpsec)

            if self.integrate_in_band:
                J11 = (
                    (1 + self.h1[:, np.newaxis]) * ca ** 2
                    - (1 + self.h2[:, np.newaxis])
                    * sa ** 2
                    * np.exp(1j * self.beta[:, np.newaxis])
                    - (self.z1[:, np.newaxis] + self.z2[:, np.newaxis]) * ca * sa
                )
                J12 = (
                    (
                        (1 + self.h1[:, np.newaxis])
                        + (1 + self.h2[:, np.newaxis])
                        * np.exp(1j * self.beta[:, np.newaxis])
                    )
                    * ca
                    * sa
                    + self.z1[:, np.newaxis] * ca ** 2
                    - self.z2[:, np.newaxis] * sa ** 2
                )

                if self.built_map_on_the_fly:
                    tod = (
                        (
                            0.5
                            * (np.abs(J11) ** 2 + np.abs(J12) ** 2)
                            * self.maps[:, 0, pix]
                            + 0.5
                            * (np.abs(J11) ** 2 - np.abs(J12) ** 2)
                            * self.maps[:, 1, pix]
                            + (J11 * J12.conjugate()).real * self.maps[:, 2, pix]
                        )
                        * self.cmb2bb[:, np.newaxis]
                    ).sum(axis=0) / self.norm
                else:
                    obs.tod[idet, :] += (
                        (
                            0.5
                            * (np.abs(J11) ** 2 + np.abs(J12) ** 2)
                            * self.maps[:, 0, pix]
                            + 0.5
                            * (np.abs(J11) ** 2 - np.abs(J12) ** 2)
                            * self.maps[:, 1, pix]
                            + (J11 * J12.conjugate()).real * self.maps[:, 2, pix]
                        )
                        * self.cmb2bb[:, np.newaxis]
                    ).sum(axis=0) / self.norm

            else:
                J11 = (
                    (1 + self.h1) * ca ** 2
                    - (1 + self.h2) * sa ** 2 * np.exp(1j * self.beta)
                    - (self.z1 + self.z2) * ca * sa
                )
                J12 = (
                    ((1 + self.h1) + (1 + self.h2) * np.exp(1j * self.beta)) * ca * sa
                    + self.z1 * ca ** 2
                    - self.z2 * sa ** 2
                )

                if self.built_map_on_the_fly:
                    tod = (
                        0.5 * (np.abs(J11) ** 2 + np.abs(J12) ** 2) * self.maps[0, pix]
                        + 0.5
                        * (np.abs(J11) ** 2 - np.abs(J12) ** 2)
                        * self.maps[1, pix]
                        + (J11 * J12.conjugate()).real * self.maps[2, pix]
                    )
                else:
                    obs.tod[idet, :] += (
                        0.5 * (np.abs(J11) ** 2 + np.abs(J12) ** 2) * self.maps[0, pix]
                        + 0.5
                        * (np.abs(J11) ** 2 - np.abs(J12) ** 2)
                        * self.maps[1, pix]
                        + (J11 * J12.conjugate()).real * self.maps[2, pix]
                    )

            if self.built_map_on_the_fly:

                if self.correct_in_solver:

                    if self.integrate_in_band_solver:
                        J11 = (
                            (1 + self.h1s[:, np.newaxis]) * ca ** 2
                            - (1 + self.h2s[:, np.newaxis])
                            * sa ** 2
                            * np.exp(1j * self.betas[:, np.newaxis])
                            - (self.z1s[:, np.newaxis] + self.z2s[:, np.newaxis])
                            * ca
                            * sa
                        )
                        J12 = (
                            (
                                (1 + self.h1s[:, np.newaxis])
                                + (1 + self.h2s[:, np.newaxis])
                                * np.exp(1j * self.betas[:, np.newaxis])
                            )
                            * ca
                            * sa
                            + self.z1s[:, np.newaxis] * ca ** 2
                            - self.z2s[:, np.newaxis] * sa ** 2
                        )
                    else:
                        J11 = (
                            (1 + self.h1s) * ca ** 2
                            - (1 + self.h2s) * sa ** 2 * np.exp(1j * self.betas)
                            - (self.z1s + self.z2s) * ca * sa
                        )
                        J12 = (
                            ((1 + self.h1s) + (1 + self.h2s) * np.exp(1j * self.betas))
                            * ca
                            * sa
                            + self.z1s * ca ** 2
                            - self.z2s * sa ** 2
                        )

                    del (ca, sa)

                    Tterm = 0.5 * (np.abs(J11) ** 2 + np.abs(J12) ** 2)
                    Qterm = 0.5 * (np.abs(J11) ** 2 - np.abs(J12) ** 2)
                    Uterm = (J11 * J12.conjugate()).real

                    self.atd[pix, 0] += tod * Tterm
                    self.atd[pix, 1] += tod * Qterm
                    self.atd[pix, 2] += tod * Uterm

                    self.ata[pix, 0, 0] += Tterm * Tterm
                    self.ata[pix, 1, 0] += Tterm * Qterm
                    self.ata[pix, 2, 0] += Tterm * Uterm
                    self.ata[pix, 1, 1] += Qterm * Qterm
                    self.ata[pix, 2, 1] += Qterm * Uterm
                    self.ata[pix, 2, 2] += Uterm * Uterm

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
            else:
                obs.psi[idet, :] = pointings[idet, :, 2] + 2 * times * hwp_radpsec
                obs.pixind[idet, :] = pix

        return

    def make_map(self, obss):

        """It generates "on the fly" map. This option is only availabe if `built_map_on_the_fly`
        is set to True.

        Args:
             obss list of class:`Observations`: only necessary for the communicator
             pointings (float): pointing for each sample and detector
                 generated by lbs.scanning.get_pointings
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
