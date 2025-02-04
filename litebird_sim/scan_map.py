# -*- encoding: utf-8 -*-

import numpy as np
from numba import njit, prange

from ducc0.healpix import Healpix_Base
from typing import Union, List, Dict, Optional
from .observations import Observation
from .hwp import HWP, mueller_ideal_hwp
from .pointings import get_hwp_angle
from .coordinates import rotate_coordinates_e2g, CoordinateSystem
from .healpix import npix_to_nside
import logging
import healpy as hp


@njit
def vec_stokes(stokes, T, Q, U):
    stokes[0] = T
    stokes[1] = Q
    stokes[2] = U


@njit
def vec_polarimeter(angle, gamma):
    # (1,0,0,0) x Mpol x Rpol
    vec = np.empty(4, dtype=np.float64)
    vec[0] = 1
    vec[1] = gamma * np.cos(2 * angle)
    vec[2] = gamma * np.sin(2 * angle)
    vec[3] = 0
    return vec


@njit
def rot_matrix(mat, angle):
    ca = np.cos(2 * angle)
    sa = np.sin(2 * angle)
    mat[1, 1:3] = ca, sa
    mat[2, 1:3] = -sa, ca


@njit
def compute_signal_for_one_sample(T, Q, U, co, si, gamma):
    """Bolometric equation"""
    return T + gamma * (co * Q + si * U)


@njit(parallel=True)
def scan_map_for_one_detector(
    tod_det, input_T, input_Q, input_U, pol_angle_det, pol_eff_det
):
    for i in prange(len(tod_det)):
        tod_det[i] += compute_signal_for_one_sample(
            T=input_T[i],
            Q=input_Q[i],
            U=input_U[i],
            co=np.cos(2 * pol_angle_det[i]),
            si=np.sin(2 * pol_angle_det[i]),
            gamma=pol_eff_det,
        )


@njit
def compute_signal_generic_hwp_for_one_sample(Stokes, Vpol, Rhwp, Mhwp, Rtel):
    """Bolometric equation for generic HWP Mueller matrix
    (1,0,0,0) x Mpol x Rpol x Rhwp^T x Mhwp x Rhwp x Rtel x Stokes
    """
    return Vpol @ Rhwp.T @ Mhwp @ Rhwp @ Rtel @ Stokes


@njit
def scan_map_generic_hwp_for_one_detector(
    tod_det,
    input_T,
    input_Q,
    input_U,
    orientation_telescope,
    pol_angle_det,
    pol_eff_det,
    hwp_angle,
    mueller_hwp,
):
    polarimeter = vec_polarimeter(pol_angle_det, pol_eff_det)

    vec_S = np.zeros(4, dtype=np.float64)
    rot_hwp = np.eye(4, dtype=np.float64)
    rot_tel = np.eye(4, dtype=np.float64)

    for i in range(len(tod_det)):
        vec_stokes(vec_S, input_T[i], input_Q[i], input_U[i])
        rot_matrix(rot_hwp, hwp_angle[i])
        rot_matrix(rot_tel, orientation_telescope[i])

        tod_det[i] += compute_signal_generic_hwp_for_one_sample(
            Stokes=vec_S,
            Vpol=polarimeter,
            Rhwp=rot_hwp,
            Mhwp=mueller_hwp,
            Rtel=rot_tel,
        )


def scan_map(
    tod,
    pointings,
    maps: Dict[str, np.ndarray],
    pol_angle_detectors: Union[np.ndarray, None] = None,
    pol_eff_detectors: Union[np.ndarray, None] = None,
    hwp_angle: Union[np.ndarray, None] = None,
    mueller_hwp: Union[np.ndarray, None] = None,
    input_names: Union[str, None] = None,
    input_map_in_galactic: bool = True,
    interpolation: Union[str, None] = "",
):
    """Scan a map filling time-ordered data

    This function modifies the values in `tod` by adding the contribution of the
    bolometric equation given a list of TQU maps `maps`. The `pointings` argument
    must be a DxNx2 matrix containing the pointing information, where D is the number
    of detector for the current observation and N is the size of the `tod` array.
    `pol_angle` is the array of size DxN containing the polarization angle in radiants.
    `input_names` is an array containing the keywords that allow to select the proper
    input in `maps` for each detector in the TOD. If `input_map_in_galactic` is set to
    False the input map is assumed in ecliptic coordinates, default galactic. The
    `interpolation` argument specifies the type of TOD interpolation ("" for no
    interpolation, "linear" for linear interpolation)
    """

    n_detectors = tod.shape[0]

    if type(pointings) is np.ndarray:
        assert tod.shape == pointings.shape[0:2]

    if pol_angle_detectors is None:
        pol_angle_detectors = np.zeros(n_detectors)

    if pol_eff_detectors is None:
        pol_eff_detectors = np.ones(n_detectors)

    for detector_idx in range(n_detectors):
        if type(pointings) is np.ndarray:
            curr_pointings_det = pointings[detector_idx, :, :]
        else:
            curr_pointings_det, hwp_angle = pointings(detector_idx)

        if input_map_in_galactic:
            curr_pointings_det = rotate_coordinates_e2g(curr_pointings_det)

        if input_names is None:
            maps_det = maps
        else:
            maps_det = maps[input_names[detector_idx]]

        nside = npix_to_nside(maps_det.shape[1])

        if interpolation in ["", None]:
            hpx = Healpix_Base(nside, "RING")
            pixel_ind_det = hpx.ang2pix(curr_pointings_det[:, 0:2])
            input_T = maps_det[0, pixel_ind_det]
            input_Q = maps_det[1, pixel_ind_det]
            input_U = maps_det[2, pixel_ind_det]
        elif interpolation == "linear":
            input_T = hp.get_interp_val(
                maps_det[0, :], curr_pointings_det[:, 0], curr_pointings_det[:, 1]
            )
            input_Q = hp.get_interp_val(
                maps_det[1, :], curr_pointings_det[:, 0], curr_pointings_det[:, 1]
            )
            input_U = hp.get_interp_val(
                maps_det[2, :], curr_pointings_det[:, 0], curr_pointings_det[:, 1]
            )
        else:
            raise ValueError(
                "Wrong value for interpolation. It should be one of the following:\n"
                + '- "" for no interpolation\n'
                + '- "linear" for linear interpolation\n'
            )

        if (mueller_hwp[detector_idx] is None) or (
            (mueller_hwp[detector_idx] == mueller_ideal_hwp).all()
        ):
            # With HWP implements:
            # (T + Q ρ Cos[2 (2 α - θ + ψ])] + U ρ Sin[2 (2 α - θ + ψ)])
            # without
            # (T + Q ρ Cos[2 (θ + ψ])] + U ρ Sin[2 (θ + ψ)])
            # ρ: polarization efficiency
            # θ: polarization angle
            # ψ: angle of the telescope
            # α: HWP angle
            scan_map_for_one_detector(
                tod_det=tod[detector_idx],
                input_T=input_T,
                input_Q=input_Q,
                input_U=input_U,
                pol_angle_det=(
                    pol_angle_detectors[detector_idx] + curr_pointings_det[:, 2]
                    if mueller_hwp[detector_idx] is None
                    else 2 * hwp_angle
                    - pol_angle_detectors[detector_idx]
                    + curr_pointings_det[:, 2]
                ),
                pol_eff_det=pol_eff_detectors[detector_idx],
            )
        else:
            # This implements:
            # (1,0,0,0) x Mpol(ρ) x Rpol(θ) x Rhwp(α)^T x Mhwp x Rhwp(α) x Rtel(ψ) x Stokes
            scan_map_generic_hwp_for_one_detector(
                tod_det=tod[detector_idx],
                input_T=input_T,
                input_Q=input_Q,
                input_U=input_U,
                orientation_telescope=curr_pointings_det[:, 2],
                pol_angle_det=pol_angle_detectors[detector_idx],
                pol_eff_det=pol_eff_detectors[detector_idx],
                hwp_angle=hwp_angle,
                mueller_hwp=mueller_hwp[detector_idx],
            )


def scan_map_in_observations(
    observations: Union[Observation, List[Observation]],
    maps: Dict[str, np.ndarray],
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[HWP] = None,
    input_map_in_galactic: bool = True,
    component: str = "tod",
    interpolation: Union[str, None] = "",
):
    """Scan a map filling time-ordered data

    This is a wrapper around the :func:`.scan_map` function that applies to the TOD
    stored in `observations` and the pointings stored in `pointings`. The two types
    can either bed a :class:`.Observation` instance and a NumPy matrix, or a list
    of observations and a list of NumPy matrices; in the latter case, they must have
    the same number of elements.

    The field `maps` must either be a dictionary associating the name of each detector
    with a ``(3, NPIX)`` array containing the three I/Q/U maps or a plain ``(3, NPIX)``
    array. In the latter case, the I/Q/U maps will be used for all the detectors.

    The coordinate system is usually specified using the key `Coordinates` in the
    dictionary passed to the `maps` argument, and it must be an instance of
    the class :class:`.CoordinateSystem`. If you are using a plain NumPy array instead
    of a dictionary for `maps`, you should specify whether to use Ecliptic or Galactic
    coordinates through the parameter `input_map_in_galactic`. If
    ``maps["Coordinates"]`` is present, it must be consistent with the value for
    `input_map_in_galactic`; if not, the code prints a warning and uses the former.

    By default, the signal is added to ``Observation.tod``. If you want to add it to
    some other field of the :class:`.Observation` class, use `component`::

        for cur_obs in sim.observations:
            # Allocate a new TOD for the sky signal alone
            cur_obs.sky_tod = np.zeros_like(cur_obs.tod)

        # Ask `add_noise_to_observations` to store the noise
        # in `observations.sky_tod`
        scan_map_in_observations(sim.observations, …, component="sky_tod")

    """

    if pointings is None:
        if isinstance(observations, Observation):
            obs_list = [observations]
            if hasattr(observations, "pointing_matrix"):
                ptg_list = [observations.pointing_matrix]
            else:
                ptg_list = [observations.get_pointings]
        else:
            obs_list = observations
            ptg_list = []
            for ob in observations:
                if hasattr(ob, "pointing_matrix"):
                    ptg_list.append(ob.pointing_matrix)
                else:
                    ptg_list.append(ob.get_pointings)
    else:
        if isinstance(observations, Observation):
            assert isinstance(pointings, np.ndarray), (
                "You must pass a list of observations *and* a list "
                + "of pointing matrices to scan_map_in_observations"
            )
            obs_list = [observations]
            ptg_list = [pointings]
        else:
            assert isinstance(pointings, list), (
                "When you pass a list of observations to scan_map_in_observations, "
                + "you must do the same for `pointings`"
            )
            assert len(observations) == len(pointings), (
                f"The list of observations has {len(observations)} elements, but "
                + f"the list of pointings has {len(pointings)} elements"
            )
            obs_list = observations
            ptg_list = pointings

    for cur_obs, cur_ptg in zip(obs_list, ptg_list):
        if isinstance(maps, dict):
            if all(item in maps.keys() for item in cur_obs.name):
                input_names = cur_obs.name
            elif all(item in maps.keys() for item in cur_obs.channel):
                input_names = cur_obs.channel
            else:
                raise ValueError(
                    "The dictionary maps does not contain all the relevant"
                    + "keys, please check the list of detectors and channels"
                )
            if "Coordinates" in maps.keys():
                dict_input_map_in_galactic = (
                    maps["Coordinates"] is CoordinateSystem.Galactic
                )
                if dict_input_map_in_galactic != input_map_in_galactic:
                    logging.warning(
                        "input_map_in_galactic variable in scan_map_in_observations"
                        + " overwritten!"
                    )
                input_map_in_galactic = dict_input_map_in_galactic
        else:
            assert isinstance(maps, np.ndarray), (
                "maps must either a dictionary contaning keys for all the"
                + "channels/detectors, or be a numpy array of dim (3 x Npix)"
            )
            input_names = None

        if hwp is None:
            if cur_obs.has_hwp:
                if hasattr(cur_obs, "hwp_angle"):
                    hwp_angle = cur_obs.hwp_angle
                else:
                    hwp_angle = cur_obs.get_pointings()[1]
            else:
                assert all(m is None for m in cur_obs.mueller_hwp), (
                    "Detectors have been initialized with a mueller_hwp,"
                    "but no HWP is either passed or initilized in the pointing"
                )
                hwp_angle = None
        else:
            if type(cur_ptg) is np.ndarray:
                hwp_angle = get_hwp_angle(cur_obs, hwp)
            else:
                logging.warning(
                    "For using an external HWP object also pass a pre-calculated pointing"
                )

        scan_map(
            tod=getattr(cur_obs, component),
            pointings=cur_ptg,
            maps=maps,
            pol_angle_detectors=cur_obs.pol_angle_rad,
            pol_eff_detectors=cur_obs.pol_efficiency,
            hwp_angle=hwp_angle,
            mueller_hwp=cur_obs.mueller_hwp,
            input_names=input_names,
            input_map_in_galactic=input_map_in_galactic,
            interpolation=interpolation,
        )
