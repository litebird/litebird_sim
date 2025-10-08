# -*- encoding: utf-8 -*-
from typing import Union, List

import healpy as hp
import numpy as np
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from numba import njit

import litebird_sim as lbs
from ..coordinates import CoordinateSystem
from ..observations import Observation
from . import mueller_methods
from . import jones_methods
from ..pointings_in_obs import (
    _get_pointings_array,
)
from ..hwp import HWP, Calc

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


def set_band_params_for_one_detector(hwp, band_filenames, idet):
    if hwp.calc is Calc.JONES:
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
            band_filenames[idet],
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
    # elif bandpass:
    #    cur_det_params['freqs'], bandpass_profile = bandpass_profile(
    #        cur_det_params['freqs'], bandpass, include_beam_throughput
    #    )
    #
    #    cmb2bb = _dBodTth(cur_det_params['freqs']) * bandpass_profile

    # Normalize the band
    cmb2bb /= np.trapz(cmb2bb, det_params["freqs"])

    return [det_params, cmb2bb]


def fill_tod(
    hwp: HWP,
    observation: Observation,
    pointings,
    hwp_angle: np.ndarray | None = None,
    input_map_in_galactic: bool = True,
    save_tod: bool = True,
    pointings_dtype=np.float64,
    interpolation: str | None = "",
    maps: np.ndarray = None,
    apply_non_linearity: bool = False,
    add_2f_hwpss: bool = False,
    mueller_phases: dict | None = None,
    integrate_in_band: bool = False,
    band_filenames: List[str] = None,
):
    r"""Fill a TOD for one observation, using HWP rotation speed
    harmonics calculus.

    Parameters
    ----------
    observation : `Observation`
        `Observation` object containing detector names, pointings,
        and TOD data, to which the computed sky signal will be added.

    pointings : np.ndarray or callable
        Pointing information for each detector. If an array, it should have shape
        (n_detectors, n_samples, 2), where the last dimension contains (theta, phi) in radians.
        If a callable, it should return pointing data when passed a detector index.

    hwp_angle : np.ndarray or None, default=None
        Half-wave plate (HWP) angles of an external HWP object. If None, the HWP information
        is taken from the Observation.

    input_map_in_galactic : bool, default=True
        Whether the input sky maps are provided in Galactic coordinates. If False, they are
        assumed to be in Ecliptic coordinates.

    save_tod : bool, default=True
        If False, ``obs.tod`` gets deleted.

    pointings_dtype : dtype, optional
        Data type for pointings generated on the fly. If the pointing is passed or
        already precomputed this parameter is ineffective. Default is `np.float64`.

    interpolation : str or None, default=""
        Method for extracting values from the maps:
        - "" (default): Nearest-neighbor interpolation.
        - "linear": Linear interpolation using Healpix.

    maps : dict of str -> np.ndarray
        Dictionary containing Stokes parameter maps (T, Q, U) in Healpix format. The keys
        correspond to different sky components.

    aply_non_linearity : bool, default=False
        Whether to apply detector non-linearity effects to the TOD.

    add_2f_hwpss : bool, default=False
        Whether to add a 2f HWP synchronous signal to the TOD.

    mueller_phases : dict or None, default=None
        Dictionary containing phase shifts for the 2f and 4f harmonics in the Mueller
        matrix formalism. If None, default values from Patanchon et al. 2021 are used.

    integrate_in_band : bool, default=False
        Whether to integrate the signal over the detector's frequency band. Only implemented for the Jones formalism.

    band_filenames : list of str or None, default=None
        List of filenames containing bandpass information for each detector. Required if `integrate_in_band` is True.

    Raises
    ------
    NotImplementedError
        If `integrate_in_band` is True and the HWP calculus is set to Mueller.
    ValueError
        If an invalid interpolation method is specified.
    """

    if type(pointings) is np.ndarray:
        assert observation.tod.shape == pointings.shape[0:2]

    if integrate_in_band:
        band_filenames = band_filenames

    if mueller_phases is None:
        # (temporary solution) using phases from Patanchon et al 2021 as the default.
        mueller_phases = {
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

    if input_map_in_galactic:
        output_coordinate_system = CoordinateSystem.Galactic
    else:
        output_coordinate_system = CoordinateSystem.Ecliptic

    for idet in range(observation.n_detectors):
        cur_point, cur_hwp_angle = _get_pointings_array(
            detector_idx=idet,
            pointings=pointings,
            hwp_angle=hwp_angle,
            output_coordinate_system=output_coordinate_system,
            pointings_dtype=pointings_dtype,
        )

        tod = observation.tod[idet, :]
        nside = hp.npix2nside(maps.shape[1])

        # all observed pixels over time (for each sample),
        # i.e. len(pix)==len(times)
        if interpolation in ["", None]:
            pix = hp.ang2pix(nside, cur_point[:, 0], cur_point[:, 1])

        if interpolation in ["", None]:
            input_T = maps[0, pix]
            input_Q = maps[1, pix]
            input_U = maps[2, pix]

        elif interpolation == "linear":
            input_T = hp.get_interp_val(
                maps[0, :],
                cur_point[:, 0],
                cur_point[:, 1],
            )
            input_Q = hp.get_interp_val(
                maps[1, :],
                cur_point[:, 0],
                cur_point[:, 1],
            )
            input_U = hp.get_interp_val(
                maps[2, :],
                cur_point[:, 0],
                cur_point[:, 1],
            )
        else:
            raise ValueError(
                "Wrong value for interpolation. It should be one of the following:\n"
                + '- "" for no interpolation\n'
                + '- "linear" for linear interpolation\n'
            )

        xi = observation.pol_angle_rad[idet]
        psi = cur_point[:, 2]

        phi = np.deg2rad(observation.pointing_theta_phi_psi_deg[idet][1])

        cos2Xi2Phi = np.cos(2 * xi - 2 * phi)
        sin2Xi2Phi = np.sin(2 * xi - 2 * phi)

        if integrate_in_band:
            if hwp.calculus is Calc.MUELLER:
                raise NotImplementedError(
                    "Band integration is only implemented for Jones Formalism"
                )

            cur_det_params, cur_det_cmb2bb = set_band_params_for_one_detector(
                hwp, band_filenames, idet
            )

            input_T = np.array([input_T for i in range(len(cur_det_params["freqs"]))]).T
            input_Q = np.array([input_Q for i in range(len(cur_det_params["freqs"]))]).T
            input_U = np.array([input_U for i in range(len(cur_det_params["freqs"]))]).T

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
                    apply_non_linearity=apply_non_linearity,
                    g_one_over_k=observation.g_one_over_k[idet],
                    add_2f_hwpss=add_2f_hwpss,
                    amplitude_2f_k=observation.amplitude_2f_k[idet],
                )

        else:
            if hwp.calculus is Calc.MUELLER:
                mueller_methods.compute_signal_for_one_detector(
                    tod_det=tod,
                    m0f=observation.mueller_hwp[idet]["0f"],
                    m2f=observation.mueller_hwp[idet]["2f"],
                    m4f=observation.mueller_hwp[idet]["4f"],
                    rho=np.array(cur_hwp_angle, dtype=np.float64),
                    psi=np.array(psi, dtype=np.float64),
                    mapT=input_T,
                    mapQ=input_Q,
                    mapU=input_U,
                    cos2Xi2Phi=cos2Xi2Phi,
                    sin2Xi2Phi=sin2Xi2Phi,
                    phi=phi,
                    apply_non_linearity=apply_non_linearity,
                    g_one_over_k=observation.g_one_over_k[idet],
                    add_2f_hwpss=add_2f_hwpss,
                    amplitude_2f_k=observation.amplitude_2f_k[idet],
                    phases_2f=mueller_phases["2f"],
                    phases_4f=mueller_phases["4f"],
                )

            elif hwp.calculus is Calc.JONES:
                deltas_j0f = observation.jones_hwp[idet]["0f"]
                deltas_j2f = observation.jones_hwp[idet]["2f"]

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
                    apply_non_linearity=apply_non_linearity,
                    g_one_over_k=observation.g_one_over_k[idet],
                    add_2f_hwpss=add_2f_hwpss,
                    amplitude_2f_k=observation.amplitude_2f_k[idet],
                )

        observation.tod[idet] = tod

    if interpolation in ["", None]:
        del pix
    del input_T, input_Q, input_U
    if not save_tod:
        del observation.tod
