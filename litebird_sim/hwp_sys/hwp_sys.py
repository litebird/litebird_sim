# -*- encoding: utf-8 -*-
from typing import Union, List

import healpy as hp
import numpy as np
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from numba import njit

import litebird_sim as lbs
from litebird_sim import mpi
from ..coordinates import rotate_coordinates_e2g
from ..detectors import FreqChannelInfo
from ..mbs.mbs import MbsParameters
from ..observations import Observation
from . import mueller_methods
from . import jones_methods

COND_THRESHOLD = 1e10


def _dBodTrj(nu):
    return 2 * const.k_B.value * nu * nu * 1e18 / const.c.value / const.c.value


def _dBodTth(nu):
    x = const.h.value * nu.astype(int) * 1e9 / const.k_B.value / cosmo.Tcmb0.value
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
        integrate_in_band: Union[bool, None] = False,
        integrate_in_band_solver: Union[bool, None] = False,
        apply_non_linearity: Union[bool, None] = False,
        add_2f_hwpss: Union[bool, None] = False,
        interpolation: Union[str, None] = "",
        channel: Union[FreqChannelInfo, None] = None,
        maps: Union[np.ndarray, None] = None,
        comm: Union[bool, None] = None,
        mueller_phases: Union[dict, None] = None,
        mueller_or_jones: Union[str, None] = "mueller",
        band_filenames: Union[list, None] = "",
        bandpass=None,
    ):
        r"""It sets the input paramters reading a dictionary `sim.parameters`
        with key "hwp_sys" and the following input arguments

        Args:
          nside (integer): nside used in the analysis
          nside_out (integer): nside for the output maps. If not provided, same as nside
          mbs_params (:class:`.Mbs`): an instance of the :class:`.Mbs` class
          build_map_on_the_fly (bool): fills :math:`A^T A` and :math:`A^T d`
          integrate_in_band (bool): performs the band integration for tod generation
          integrate_in_band_solver (bool): performs the band integration for the
                                              map-making solver
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
          mueller_phases (dict): Phases for the HWP Mueller matrices
          mueller_or_jones (str): if "mueller", the Mueller formalism is applied to the
                HWP computations. If "jones", the jones formalism is applied. Default is "mueller"

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

        if not hasattr(self, "integrate_in_band"):
            if integrate_in_band is not None:
                self.integrate_in_band = integrate_in_band

        if not hasattr(self, "integrate_in_band_solver"):
            if integrate_in_band_solver is not None:
                self.integrate_in_band_solver = integrate_in_band_solver

        if not hasattr(self, "comm"):
            if comm is not None:
                self.comm = comm

        if not hasattr(self, "bandpass"):
            self.bandpass = bandpass

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

        if self.integrate_in_band:
            self.band_filenames = band_filenames

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

        self.mueller_or_jones = mueller_or_jones

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

    def set_band_params_for_one_detector(self, idet):
        if self.mueller_or_jones == "jones":
            variables = [
                "freqs",
                "h1_0f",
                "h2_0f",
                "beta_0f",
                "z1_0f",
                "z2_0f",
                "h1_2f",
                "h2_2f",
                "beta_2f",
                "z1_2f",
                "z2_2f",
                "h1_0f_slv",
                "h2_0f_slv",
                "beta_0f_slv",
                "z1_0f_slv",
                "z2_0f_slv",
                "h1_2f_slv",
                "h2_2f_slv",
                "beta_2f_slv",
                "z1_2f_slv",
                "z2_2f_slv",
            ]

            loaded_data = np.loadtxt(
                self.band_filenames[idet],
                delimiter=" ",
                dtype=object,
                unpack=True,
                skiprows=1,
                comments="#",
            )

            det_params = {}
            for var, data in zip(variables, loaded_data):
                if "beta" in var:
                    det_params[var] = np.deg2rad(np.array(data, dtype=np.float64))
                elif "freqs" in var:
                    det_params[var] = np.array(data, dtype=np.float64)
                else:
                    det_params[var] = np.array(data, dtype=np.complex128)

        else:  # TODO mueller_or_jones == "mueller"
            raise NotImplementedError(
                "band integration is only implemented for the Jones formalism."
            )

        # if not cur_det.bandpass:
        cmb2bb = _dBodTth(det_params["freqs"])

        # TODO: insert bandpass in detectorinfo so that we can apply the case where
        # each detector has a bandpass
        # elif self.bandpass:
        #    cur_det_params['freqs'], self.bandpass_profile = bandpass_profile(
        #        cur_det_params['freqs'], self.bandpass, self.include_beam_throughput
        #    )
        #
        #    self.cmb2bb = _dBodTth(cur_det_params['freqs']) * self.bandpass_profile

        # Normalize the band
        cmb2bb /= np.trapz(cmb2bb, det_params["freqs"])

        return [det_params, cmb2bb]

    def fill_tod(
        self,
        observations: Union[Observation, List[Observation]] = None,
        pointings: Union[np.ndarray, List[np.ndarray], None] = None,
        hwp_angle: Union[np.ndarray, List[np.ndarray], None] = None,
        input_map_in_galactic: bool = True,
        save_tod: bool = True,
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
                ``observations.tod``. If False, ``obs.tod`` gets deleted.

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
                if self.mueller_or_jones == "mueller":
                    if np.all(
                        cur_obs.mueller_hwp[idet] == np.diag([1.0, 1.0, -1.0, -1.0])
                    ):
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

                elif self.mueller_or_jones == "jones":
                    if cur_obs.jones_hwp[idet] is None:
                        cur_obs.jones_hwp[idet] = {
                            "0f": np.array([[0, 0], [0, 0]], dtype=np.float64),
                            "2f": np.array([[0, 0], [0, 0]], dtype=np.float64),
                        }

                    if cur_obs.jones_hwp_solver[idet] is None:
                        cur_obs.jones_hwp_solver[idet] = {
                            "0f": np.array([[0, 0], [0, 0]], dtype=np.float64),
                            "2f": np.array([[0, 0], [0, 0]], dtype=np.float64),
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

                xi = cur_obs.pol_angle_rad[idet]
                psi = cur_point[:, 2]

                phi = np.deg2rad(cur_obs.pointing_theta_phi_psi_deg[idet][1])

                cos2Xi2Phi = np.cos(2 * xi - 2 * phi)
                sin2Xi2Phi = np.sin(2 * xi - 2 * phi)

                if self.integrate_in_band:
                    cur_det_params, cur_det_cmb2bb = (
                        self.set_band_params_for_one_detector(idet)
                    )

                    input_T = np.array(
                        [input_T for i in range(len(cur_det_params["freqs"]))]
                    ).T
                    input_Q = np.array(
                        [input_Q for i in range(len(cur_det_params["freqs"]))]
                    ).T
                    input_U = np.array(
                        [input_U for i in range(len(cur_det_params["freqs"]))]
                    ).T

                    if self.mueller_or_jones == "mueller":
                        raise NotImplementedError(
                            "Band integration is only implemented for Jones Formalism"
                        )

                    elif self.mueller_or_jones == "jones":
                        deltas_j0f = np.zeros(
                            (len(cur_det_params["freqs"]), 2, 2), dtype=np.complex128
                        )
                        deltas_j2f = np.zeros(
                            (len(cur_det_params["freqs"]), 2, 2), dtype=np.complex128
                        )

                        for nu in range(len(cur_det_params["freqs"])):
                            deltas_j0f[nu] = np.array(
                                [
                                    [
                                        cur_det_params["h1_0f"][nu],
                                        cur_det_params["z1_0f"][nu],
                                    ],
                                    [
                                        cur_det_params["z2_0f"][nu],
                                        1
                                        - (1 + cur_det_params["h2_0f"][nu])
                                        * np.exp(cur_det_params["beta_0f"][nu] * 1j),
                                    ],
                                ],
                                dtype=np.complex128,
                            )
                            deltas_j2f[nu] = np.array(
                                [
                                    [
                                        cur_det_params["h1_2f"][nu],
                                        cur_det_params["z1_2f"][nu],
                                    ],
                                    [
                                        cur_det_params["z2_2f"][nu],
                                        1
                                        - (1 + cur_det_params["h2_2f"][nu])
                                        * np.exp(cur_det_params["beta_2f"][nu] * 1j),
                                    ],
                                ],
                                dtype=np.complex128,
                            )

                        jones_methods.integrate_inband_signal_for_one_detector(
                            tod_det=tod,
                            freqs=cur_det_params["freqs"],
                            band=cur_det_cmb2bb,
                            deltas_j0f=deltas_j0f,
                            deltas_j2f=deltas_j2f,
                            mapT=input_T,
                            mapQ=input_Q,
                            mapU=input_U,
                            rho=np.array(cur_hwp_angle, dtype=np.float64),
                            psi=np.array(psi, dtype=np.float64),
                            phi=phi,
                            cos2Xi2Phi=cos2Xi2Phi,
                            sin2Xi2Phi=sin2Xi2Phi,
                            apply_non_linearity=self.apply_non_linearity,
                            g_one_over_k=cur_obs.g_one_over_k[idet],
                            add_2f_hwpss=self.add_2f_hwpss,
                            amplitude_2f_k=cur_obs.amplitude_2f_k[idet],
                        )

                else:
                    if self.mueller_or_jones == "mueller":
                        mueller_methods.compute_signal_for_one_detector(
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

                    elif self.mueller_or_jones == "jones":
                        deltas_j0f = cur_obs.jones_hwp[idet]["0f"]
                        deltas_j2f = cur_obs.jones_hwp[idet]["2f"]

                        jones_methods.compute_signal_for_one_detector(
                            tod_det=tod,
                            deltas_j0f=deltas_j0f,
                            deltas_j2f=deltas_j2f,
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
                        )

                if self.build_map_on_the_fly:
                    if self.integrate_in_band_solver:
                        if self.mueller_or_jones == "mueller":
                            raise NotImplementedError(
                                "Band integration is only implemented for Jones Formalism"
                            )

                        elif self.mueller_or_jones == "jones":
                            deltas_j0f_solver = np.zeros(
                                (len(cur_det_params["freqs"]), 2, 2),
                                dtype=np.complex128,
                            )
                            deltas_j2f_solver = np.zeros(
                                (len(cur_det_params["freqs"]), 2, 2),
                                dtype=np.complex128,
                            )

                            for nu in range(len(cur_det_params["freqs"])):
                                deltas_j0f[nu] = np.array(
                                    [
                                        [
                                            cur_det_params["h1_0f_slv"][nu],
                                            cur_det_params["z1_0f_slv"][nu],
                                        ],
                                        [
                                            cur_det_params["z2_0f_slv"][nu],
                                            1
                                            - (1 + cur_det_params["h2_0f_slv"][nu])
                                            * np.exp(
                                                cur_det_params["beta_0f_slv"][nu] * 1j
                                            ),
                                        ],
                                    ],
                                    dtype=np.complex128,
                                )
                                deltas_j2f[nu] = np.array(
                                    [
                                        [
                                            cur_det_params["h1_2f_slv"][nu],
                                            cur_det_params["z1_2f_slv"][nu],
                                        ],
                                        [
                                            cur_det_params["z2_2f_slv"][nu],
                                            1
                                            - (1 + cur_det_params["h2_2f_slv"][nu])
                                            * np.exp(
                                                cur_det_params["beta_2f_slv"][nu] * 1j
                                            ),
                                        ],
                                    ],
                                    dtype=np.complex128,
                                )

                            jones_methods.integrate_inband_atd_ata_for_one_detector(
                                ata=self.ata,
                                atd=self.atd,
                                tod=tod,
                                freqs=cur_det_params["freqs"],
                                band=cur_det_cmb2bb,
                                deltas_j0f_solver=deltas_j0f_solver,
                                deltas_j2f_solver=deltas_j2f_solver,
                                pixel_ind=pix_out,
                                rho=np.array(cur_hwp_angle, dtype=np.float64),
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

                        elif self.mueller_or_jones == "jones":
                            deltas_j0fs = cur_obs.jones_hwp_solver[idet]["0f"]
                            deltas_j2fs = cur_obs.jones_hwp_solver[idet]["2f"]

                            jones_methods.compute_ata_atd_for_one_detector(
                                ata=self.ata,
                                atd=self.atd,
                                tod=tod,
                                deltas_j0f_solver=deltas_j0fs,
                                deltas_j2f_solver=deltas_j2fs,
                                pixel_ind=pix_out,
                                rho=np.array(cur_hwp_angle, dtype=np.float64),
                                psi=np.array(psi, dtype=np.float64),
                                phi=phi,
                                cos2Xi2Phi=cos2Xi2Phi,
                                sin2Xi2Phi=sin2Xi2Phi,
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
            print("before reducing", self.ata)
            self.comm.Allreduce(mpi.MPI.IN_PLACE, self.atd, mpi.MPI.SUM)
            self.comm.Allreduce(mpi.MPI.IN_PLACE, self.ata, mpi.MPI.SUM)
        else:
            raise NotImplementedError(
                "All observations must be distributed over the same MPI groups"
            )

        print("after reducing", self.ata)
        self.ata[:, 0, 1] = self.ata[:, 1, 0]
        self.ata[:, 0, 2] = self.ata[:, 2, 0]
        self.ata[:, 1, 2] = self.ata[:, 2, 1]

        cond = np.linalg.cond(self.ata)
        res = np.full_like(self.atd, hp.UNSEEN)
        mask = cond < COND_THRESHOLD
        res[mask] = np.linalg.solve(self.ata[mask], self.atd[mask])
        return res.T
