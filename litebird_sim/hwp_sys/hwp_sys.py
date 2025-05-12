# -*- encoding: utf-8 -*-
from typing import Union, List

import healpy as hp
import numpy as np
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from numba import njit
import litebird_sim as lbs
from litebird_sim import mpi
from .bandpass_template_module import bandpass_profile
from ..coordinates import rotate_coordinates_e2g
from ..detectors import FreqChannelInfo
from ..mbs.mbs import MbsParameters
from ..observations import Observation
import mueller_methods
import jones_methods

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


@njit(parallel=False)
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


def jones_complex_to_polar(j0, j1, j2, integrate_in_band):
    if integrate_in_band:
        return tuple(
            [
                [
                    [[(np.abs(val), np.angle(val)) for val in row] for row in matrix]
                    for matrix in arr
                ]
                for arr in [j0, j1, j2]
            ]
        )
    else:
        return tuple(
            [
                [[(np.abs(val), np.angle(val)) for val in row] for row in matrix]
                for matrix in [j0, j1, j2]
            ]
        )


def extract_deltas(j0f, j2f, j4f, integrate_in_band):
    if integrate_in_band:
        for key in [j0f, j2f, j4f]:
            key[:, 0, 0] -= 1
            key[:, 1, 1] += 1
    else:
        for key in [j0f, j2f, j4f]:
            key[0, 0] -= 1
            key[1, 1] += 1

    return j0f, j2f, j4f


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
        mueller_or_jones: Union[str, None] = None,
        integrate_in_band: Union[bool, None] = None,
        build_map_on_the_fly: Union[bool, None] = None,
        integrate_in_band_solver: Union[bool, None] = None,
        Channel: Union[FreqChannelInfo, None] = None,
        maps: Union[np.ndarray, None] = None,
        comm: Union[bool, None] = None,
    ):
        r"""It sets the input paramters reading a dictionary `sim.parameters`
        with key "hwp_sys" and the following input arguments

        Args:
          nside (integer): nside used in the analysis
          Mbsparams (:class:`.Mbs`): an instance of the :class:`.Mbs` class
              Input maps needs to be in galactic (mbs default)
          integrate_in_band (bool): performs the band integration for tod generation
          build_map_on_the_fly (bool): fills :math:`A^T A` and :math:`A^T d`
          integrate_in_band_solver (bool): performs the band integration for the
                                           map-making solver
          Channel (:class:`.FreqChannelInfo`): an instance of the
                                                :class:`.FreqChannelInfo` class
          maps (float): input maps (3, npix) coherent with nside provided,
              Input maps needs to be in galactic (mbs default)
              if `maps` is not None, `Mbsparams` is ignored
              (i.e. input maps are not generated)
          comm (SerialMpiCommunicator): MPI communicator
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

            self.nside = paramdict.get("nside", False)

            self.integrate_in_band = paramdict.get("integrate_in_band", False)
            self.build_map_on_the_fly = paramdict.get("build_map_on_the_fly", False)
            self.integrate_in_band_solver = paramdict.get(
                "integrate_in_band_solver", False
            )

            self.bandpass = paramdict.get("bandpass", False)
            self.bandpass_solver = paramdict.get("bandpass_solver", False)
            self.include_beam_throughput = paramdict.get(
                "include_beam_throughput", False
            )

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

        if not hasattr(self, "integrate_in_band"):
            if integrate_in_band is not None:
                self.integrate_in_band = integrate_in_band

        if not hasattr(self, "build_map_on_the_fly"):
            if build_map_on_the_fly is not None:
                self.build_map_on_the_fly = build_map_on_the_fly

        if not hasattr(self, "integrate_in_band_solver"):
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
            Channel = lbs.FreqChannelInfo(bandcenter_ghz=140)

        if self.integrate_in_band:
            if not self.bandpass:
                self.cmb2bb = _dBodTth(self.freqs)

            elif self.bandpass:
                self.freqs, self.bandpass_profile = bandpass_profile(
                    self.freqs, self.bandpass, self.include_beam_throughput
                )

                self.cmb2bb = _dBodTth(self.freqs) * self.bandpass_profile

            # Normalize the band
            self.cmb2bb /= np.trapz(self.cmb2bb, self.freqs)

            rank = comm.rank

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
                if comm is not None:
                    self.maps = comm.bcast(self.maps, root=0)
            else:
                self.maps = maps
            del maps

        else:
            if np.any(maps) is None:
                mbs = lbs.Mbs(
                    simulation=self.sim, parameters=Mbsparams, channel_list=Channel
                )
                self.maps = mbs.run_all()[0][
                    f"{Channel.channel.split()[0]}_{Channel.channel.split()[1]}"
                ]
            else:
                self.maps = maps

                del maps

        if self.integrate_in_band_solver:
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

        if self.build_map_on_the_fly:
            self.atd = np.zeros((self.npix, 3), dtype=np.float64)
            self.ata = np.zeros((self.npix, 3, 3), dtype=np.float64)

        self.mueller_or_jones = mueller_or_jones

        self.comm = comm

    def fill_tod(
        self,
        observations: Union[Observation, List[Observation]] = None,
        pointings: Union[np.ndarray, List[np.ndarray], None] = None,
        hwp_angle: Union[np.ndarray, List[np.ndarray], None] = None,
        input_map_in_galactic: bool = True,
        save_tod: bool = False,
        dtype_pointings=np.float32,
        apply_non_linearity=False,
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

                If ``observations`` is not a list, ``pointings`` must be a np.array
                    of dimensions (N_det, N_samples, 3).
                If ``observations`` is a list, ``pointings`` must be a list of same length.


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
                 ``tod`` gets deleted.

        dtype_pointings: if ``pointings`` is None and is computed within ``fill_tod``, this
                         is the dtype for pointings and tod (default: np.float64).
        """

        rank = self.comm.rank

        assert (
            observations is not None
        ), "You need to pass at least one observation to fill_tod."

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
            if not self.build_map_on_the_fly:
                # allocate those for "make_binned_map", later filled
                if not hasattr(cur_obs, "pointing_matrix"):
                    cur_obs.pointing_matrix = np.empty(
                        (cur_obs.n_detectors, cur_obs.n_samples, 3),
                        dtype=dtype_pointings,
                    )

            for idet in range(cur_obs.n_detectors):
                cur_det = self.sim.detectors[idet * rank + idet]

                if self.mueller_or_jones == "mueller":
                    if cur_det.mueller_hwp is None:
                        cur_det.mueller_hwp = {
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

                    if cur_det.mueller_hwp_solver is None:
                        cur_det.mueller_hwp_solver = {
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

                elif self.mueller_or_jones == "jones":
                    if cur_det.jones_hwp is None:
                        cur_det.jones_hwp = {
                            "0f": np.array([[1, 0], [0, -1]], dtype=np.float64),
                            "2f": np.array([[1, 0], [0, -1]], dtype=np.float64),
                            "4f": np.array([[1, 0], [0, -1]], dtype=np.float64),
                        }

                    if cur_det.jones_hwp_solver is None:
                        cur_det.jones_hwp_solver = {
                            "0f": np.array([[1, 0], [0, -1]], dtype=np.float64),
                            "2f": np.array([[1, 0], [0, -1]], dtype=np.float64),
                            "4f": np.array([[1, 0], [0, -1]], dtype=np.float64),
                        }

                tod = np.float64(cur_obs.tod[idet, :])

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
                # xi = compute_polang_from_detquat(cur_obs.quat[idet].quats[0]) % (
                #    2 * np.pi
                # )

                xi = cur_det.pol_angle_rad
                psi = cur_point[:, 2]
                phi = np.deg2rad(cur_det.pointing_theta_phi_psi_deg[1])

                cos2Xi2Phi = np.cos(2 * xi - 2 * phi)
                sin2Xi2Phi = np.sin(2 * xi - 2 * phi)

                if self.integrate_in_band:
                    if self.mueller_or_jones == "mueller":
                        assert (
                            len(cur_det.mueller_hwp["0f"]) == 1
                            or len(cur_det.mueller_hwp["2f"]) == 1
                            or len(cur_det.mueller_hwp["4f"]) == 1
                        ), "integrate_in_band is set to true but at least one of the harmonics has only one matrix"

                        mueller_methods.integrate_inband_signal_for_one_detector(
                            tod_det=tod,
                            freqs=self.freqs,
                            band=self.cmb2bb,
                            m0f=cur_det.mueller_hwp["0f"],
                            m2f=cur_det.mueller_hwp["2f"],
                            m4f=cur_det.mueller_hwp["4f"],
                            pixel_ind=pix,
                            theta=np.array(cur_hwp_angle, dtype=np.float64),
                            psi=np.array(psi, dtype=np.float64),
                            maps=np.array(self.maps, dtype=np.float64),
                            phi=phi,
                            cos2Xi2Phi=cos2Xi2Phi,
                            sin2Xi2Phi=sin2Xi2Phi,
                        )

                    elif self.mueller_or_jones == "jones":
                        assert (
                            len(cur_det.jones_hwp["0f"]) == 1
                            or len(cur_det.jones_hwp["2f"]) == 1
                            or len(cur_det.jones_hwp["4f"]) == 1
                        ), "integrate_in_band is set to true but at least one of the harmonics has only one matrix"

                        j0f = cur_det.jones_hwp["0f"]
                        j2f = cur_det.jones_hwp["2f"]
                        j4f = cur_det.jones_hwp["4f"]

                        j0f, j2f, j4f = extract_deltas(
                            j0f, j2f, j4f, self.integrate_in_band
                        )
                        j0f, j2f, j4f = jones_complex_to_polar(
                            j0f, j2f, j4f, self.integrate_in_band
                        )

                        mueller_methods.integrate_inband_signal_for_one_detector(
                            tod_det=tod,
                            freqs=self.freqs_solver,
                            band=self.cmb2bb_solver,
                            j0f=j0f,
                            j2f=j2f,
                            j4f=j4f,
                            pixel_ind=pix,
                            theta=np.array(cur_hwp_angle, dtype=np.float64),
                            psi=np.array(psi, dtype=np.float64),
                            maps=np.array(self.maps, dtype=np.float64),
                            phi=phi,
                            cos2Xi2Phi=cos2Xi2Phi,
                            sin2Xi2Phi=sin2Xi2Phi,
                        )

                else:
                    if self.mueller_or_jones == "mueller":
                        mueller_methods.compute_signal_for_one_detector(
                            tod_det=tod,
                            pixel_ind=pix,
                            m0f=cur_det.mueller_hwp["0f"],
                            m2f=cur_det.mueller_hwp["2f"],
                            m4f=cur_det.mueller_hwp["4f"],
                            theta=np.array(cur_hwp_angle, dtype=np.float64),
                            psi=np.array(psi, dtype=np.float64),
                            maps=np.array(self.maps, dtype=np.float64),
                            cos2Xi2Phi=cos2Xi2Phi,
                            sin2Xi2Phi=sin2Xi2Phi,
                            phi=phi,
                        )

                    elif self.mueller_or_jones == "jones":
                        j0f = cur_det.jones_hwp["0f"]
                        j2f = cur_det.jones_hwp["2f"]
                        j4f = cur_det.jones_hwp["4f"]

                        j0f, j2f, j4f = extract_deltas(
                            j0f, j2f, j4f, self.integrate_in_band
                        )
                        j0f, j2f, j4f = jones_complex_to_polar(
                            j0f, j2f, j4f, self.integrate_in_band
                        )

                        jones_methods.compute_signal_for_one_detector(
                            tod_det=tod,
                            pixel_ind=pix,
                            j0f=j0f,
                            j2f=j2f,
                            j4f=j4f,
                            theta=np.array(cur_hwp_angle, dtype=np.float64),
                            psi=np.array(psi, dtype=np.float64),
                            maps=np.array(self.maps, dtype=np.float64),
                            cos2Xi2Phi=cos2Xi2Phi,
                            sin2Xi2Phi=sin2Xi2Phi,
                            phi=phi,
                        )

                if self.build_map_on_the_fly:
                    if self.integrate_in_band_solver:
                        if self.mueller_or_jones == "mueller":
                            assert (
                                len(cur_det.mueller_hwp_solver["0f"]) == 1
                                or len(cur_det.mueller_hwp_solver["2f"]) == 1
                                or len(cur_det.mueller_hwp_solver["4f"]) == 1
                            ), "integrate_in_band is set to true but at least one of the harmonics has only one solver matrix"
                            mueller_methods.integrate_inband_atd_ata_for_one_detector(
                                ata=self.ata,
                                atd=self.atd,
                                tod=tod,
                                freqs=self.freqs,
                                band=self.cmb2bb,
                                m0f_solver=cur_det.mueller_hwp_solver["0f"],
                                m2f_solver=cur_det.mueller_hwp_solver["2f"],
                                m4f_solver=cur_det.mueller_hwp_solver["4f"],
                                pixel_ind=pix,
                                theta=np.array(cur_hwp_angle, dtype=np.float64),
                                psi=np.array(psi, dtype=np.float64),
                                phi=phi,
                                cos2Xi2Phi=cos2Xi2Phi,
                                sin2Xi2Phi=sin2Xi2Phi,
                            )

                        elif self.mueller_or_jones == "jones":
                            assert (
                                len(cur_det.jones_hwp_solver["0f"]) == 1
                                or len(cur_det.jones_hwp_solver["2f"]) == 1
                                or len(cur_det.jones_hwp_solver["4f"]) == 1
                            ), "integrate_in_band is set to true but at least one of the harmonics has only one solver matrix"

                            j0fs = cur_det.jones_hwp_solver["0f"]
                            j2fs = cur_det.jones_hwp_solver["2f"]
                            j4fs = cur_det.jones_hwp_solver["4f"]

                            j0fs, j2fs, j4fs = extract_deltas(
                                j0fs, j2fs, j4fs, self.integrate_in_band_solver
                            )
                            j0fs, j2fs, j4fs = jones_complex_to_polar(
                                j0fs, j2fs, j4fs, self.integrate_in_band_solver
                            )

                            jones_methods.integrate_inband_atd_ata_for_one_detector(
                                ata=self.ata,
                                atd=self.atd,
                                tod=tod,
                                freqs=self.freqs,
                                band=self.cmb2bb,
                                j0f_solver=j0fs,
                                j2f_solver=j2fs,
                                j4f_solver=j4fs,
                                pixel_ind=pix,
                                theta=np.array(cur_hwp_angle, dtype=np.float64),
                                psi=np.array(psi, dtype=np.float64),
                                phi=phi,
                                cos2Xi2Phi=cos2Xi2Phi,
                                sin2Xi2Phi=sin2Xi2Phi,
                            )

                    else:
                        if self.mueller_or_jones == "mueller":
                            mueller_methods.compute_ata_atd_for_one_detector(
                                ata=self.ata,
                                atd=self.atd,
                                tod=tod,
                                m0f_solver=cur_det.mueller_hwp_solver["0f"],
                                m2f_solver=cur_det.mueller_hwp_solver["2f"],
                                m4f_solver=cur_det.mueller_hwp_solver["4f"],
                                pixel_ind=pix,
                                theta=np.array(cur_hwp_angle, dtype=np.float64),
                                psi=np.array(psi, dtype=np.float64),
                                phi=phi,
                                cos2Xi2Phi=cos2Xi2Phi,
                                sin2Xi2Phi=sin2Xi2Phi,
                            )

                        elif self.mueller_or_jones == "jones":
                            j0fs = cur_det.jones_hwp_solver["0f"]
                            j2fs = cur_det.jones_hwp_solver["2f"]
                            j4fs = cur_det.jones_hwp_solver["4f"]

                            # print(cur_det.jones_hwp)

                            j0fs, j2fs, j4fs = extract_deltas(
                                j0fs, j2fs, j4fs, self.integrate_in_band_solver
                            )
                            j0fs, j2fs, j4fs = jones_complex_to_polar(
                                j0fs, j2fs, j4fs, self.integrate_in_band_solver
                            )

                            jones_methods.compute_ata_atd_for_one_detector(
                                ata=self.ata,
                                atd=self.atd,
                                tod=tod,
                                j0f_solver=j0fs,
                                j2f_solver=j2fs,
                                j4f_solver=j4fs,
                                pixel_ind=pix,
                                theta=np.array(cur_hwp_angle, dtype=np.float64),
                                psi=np.array(psi, dtype=np.float64),
                                phi=phi,
                                cos2Xi2Phi=cos2Xi2Phi,
                                sin2Xi2Phi=sin2Xi2Phi,
                            )

        del (pix, self.maps)
        if not save_tod:
            del tod

        return

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

        assert (
            self.build_map_on_the_fly
        ), "make_map available only with build_map_on_the_fly option activated"
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
            self.atd = self.comm.allreduce(self.atd, mpi.MPI.SUM)
            self.ata = self.comm.allreduce(self.ata, mpi.MPI.SUM)
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
        res[mask] = np.linalg.solve(self.ata, self.atd)
        return res.T
