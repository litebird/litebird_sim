# -*- encoding: utf-8 -*-

import numpy as np
from numba import njit, prange

from ducc0.healpix import Healpix_Base
from typing import Union, List, Dict, Optional
from .observations import Observation
from .hwp import HWP, mueller_ideal_hwp
from .pointings_in_obs import (
    _get_hwp_angle,
    _get_pointings_array,
    _get_pol_angle,
    _normalize_observations_and_pointings,
)
from .coordinates import CoordinateSystem
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
    pointings_dtype=np.float64,
):
    """
    Scan a sky map and fill time-ordered data (TOD) based on detector observations.

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

    Parameters
    ----------
    tod : np.ndarray
        Time-ordered data (TOD) array of shape (n_detectors, n_samples) that will be filled
        with the simulated sky signal.

    pointings : np.ndarray or callable
        Pointing information for each detector. If an array, it should have shape
        (n_detectors, n_samples, 2), where the last dimension contains (theta, phi) in radians.
        If a callable, it should return pointing data when passed a detector index.

    maps : dict of str -> np.ndarray
        Dictionary containing Stokes parameter maps (T, Q, U) in Healpix format. The keys
        correspond to different sky components.

    pol_angle_detectors : np.ndarray or None, default=None
        Polarization angles of detectors in radians. If None, all angles are set to zero.

    pol_eff_detectors : np.ndarray or None, default=None
        Polarization efficiency of detectors. If None, all detectors have unit efficiency.

    hwp_angle : np.ndarray or None, default=None
        Half-wave plate (HWP) angles of an external HWP object. If None, the HWP information
        is taken from the Observation.

    mueller_hwp : np.ndarray or None, default=None
        Mueller matrices for the HWP. If None, a standard polarization response is used.

    input_names : str or None, default=None
        Names of the sky maps to use for each detector. If None, all detectors use the same map.

    input_map_in_galactic : bool, default=True
        Whether the input sky maps are provided in Galactic coordinates. If False, they are
        assumed to be in Ecliptic coordinates.

    interpolation : str or None, default=""
        Method for extracting values from the maps:
        - "" (default): Nearest-neighbor interpolation.
        - "linear": Linear interpolation using Healpix.

    pointings_dtype : dtype, optional
        Data type for pointings generated on the fly. If the pointing is passed or
        already precomputed this parameter is ineffective. Default is `np.float64`.

    Raises
    ------
    ValueError
        If an invalid interpolation method is provided.
    AssertionError
        If `tod` and `pointings` shapes are inconsistent.

    Notes
    -----
    - The function modifies `tod` in place by adding the scanned sky signal.
    - If `mueller_hwp` is provided, a full HWP Mueller matrix transformation is applied.
    - Polarization angles are corrected based on telescope orientation and HWP effects.
    - This function is crucial for simulating realistic observations in CMB and astrophysical
      experiments.
    """

    n_detectors = tod.shape[0]

    if type(pointings) is np.ndarray:
        assert tod.shape == pointings.shape[0:2]

    if pol_angle_detectors is None:
        pol_angle_detectors = np.zeros(n_detectors)

    if pol_eff_detectors is None:
        pol_eff_detectors = np.ones(n_detectors)

    for detector_idx in range(n_detectors):
        if input_map_in_galactic:
            output_coordinate_system = CoordinateSystem.Galactic
        else:
            output_coordinate_system = CoordinateSystem.Ecliptic

        curr_pointings_det, hwp_angle = _get_pointings_array(
            detector_idx=detector_idx,
            pointings=pointings,
            hwp_angle=hwp_angle,
            output_coordinate_system=output_coordinate_system,
            pointings_dtype=pointings_dtype,
        )

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
                pol_angle_det=_get_pol_angle(
                    curr_pointings_det, hwp_angle, pol_angle_detectors[detector_idx]
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
    maps: Union[np.ndarray, Dict[str, np.ndarray]],
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    hwp: Optional[HWP] = None,
    input_map_in_galactic: bool = True,
    component: str = "tod",
    interpolation: Optional[str] = "",
    pointings_dtype=np.float64,
):
    """

    Scan a sky map and fill time-ordered data (TOD) for a set of observations.

    This is a wrapper around the :func:`.scan_map` function that applies to the TOD
    stored in `observations` and the pointings stored in `pointings`. The two types
    can either bed a :class:`.Observation` instance and a NumPy matrix, or a list
    of observations and a list of NumPy matrices; in the latter case, they must have
    the same number of elements.

    The field `maps` must either be a (list of) dictionary associating the name of each
    detector with a ``(3, NPIX)`` array containing the three I/Q/U maps or a
    plain ``(3, NPIX)`` array. In the latter case, the I/Q/U maps will be used for all
    the detectors.

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

    Parameters
    ----------
    observations : Observation or list of Observation
        One or more `Observation` objects containing detector names, pointings,
        and TOD data, to which the computed sky signal will be added.

    maps : np.ndarray or dict of str -> np.ndarray
        Sky maps containing Stokes parameters (T, Q, U). If a dictionary, keys
        should match detector or channel names, and values should be arrays of shape (3, NPIX).
        If a single array is provided, the same map is used for all detectors.

    pointings : np.ndarray or list of np.ndarray, optional
        Pointing matrices associated with the observations. If None, the function
        extracts pointing information from the `Observation` objects.

    hwp : HWP, optional
        Half-wave plate (HWP) model. If None, HWP effects are ignored unless
        the `Observation` object contains HWP data.

    input_map_in_galactic : bool, default=True
        Whether the input sky maps are provided in Galactic coordinates. If False, they
        are assumed to be in Ecliptic coordinates.

    component : str, default="tod"
        The TOD component in the `Observation` object where the computed signal will be stored.

    interpolation : str, optional, default=""
        Method for extracting values from the sky maps:
        - "" (default): Nearest-neighbor interpolation.
        - "linear": Linear interpolation using Healpix.

    pointings_dtype : dtype, optional
        Data type for pointings generated on the fly. If the pointing is passed or
        already precomputed this parameter is ineffective. Default is `np.float64`.

    Raises
    ------
    ValueError
        If the dictionary `maps` does not contain the required detector or channel keys.
    AssertionError
        If the number of observations and pointings do not match.
        If `maps` is not a dictionary or a valid `(3, NPIX)` NumPy array.
        If `tod` and `pointings` shapes are inconsistent.

    Notes
    -----
    - This function modifies `observations` in place by adding the computed sky signal
      to the specified `component` field.
    - If `maps` is a dictionary, its `Coordinates` key (if present) must match
      `input_map_in_galactic`, otherwise a warning is issued.
    - If `pointings` is None, the function attempts to extract them from `Observation` objects.
      If the pointing is generated on the fly pointings_dtype specifies its type.
    - If an HWP model is provided, the function computes HWP angles and applies the
      corresponding Mueller matrices.
    - This function supports both single observations and lists of observations,
      handling each one separately.
    """

    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )

    if maps is None:
        try:
            maps = observations[0].sky
        except AttributeError:
            msg = "'maps' is None and nothing is found in the observation. You should either pass the maps here, or store them in the observations if 'mbs' is used."
            raise AttributeError(msg)
        assert maps["type"] == "maps", (
            "'maps' should be of type 'maps'. Disable 'store_alms' in 'MbsParameters' to make it so."
        )

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
            hwp_angle = None
        else:
            # If you pass an external HWP, get hwp_angle here, otherwise this is handled in scan_map
            hwp_angle = _get_hwp_angle(
                obs=cur_obs, hwp=hwp, pointing_dtype=pointings_dtype
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
            pointings_dtype=pointings_dtype,
        )
