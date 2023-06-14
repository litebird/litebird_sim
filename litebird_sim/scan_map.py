# -*- encoding: utf-8 -*-

from numba import njit
import numpy as np

from ducc0.healpix import Healpix_Base

from astropy.time import Time, TimeDelta

from typing import Union, List, Dict

from .observations import Observation

from .coordinates import rotate_coordinates_e2g, CoordinateSystem

from .healpix import npix_to_nside

import logging

import healpy as hp


@njit
def compute_signal_for_one_sample(T, Q, U, co, si):
    """Bolometric equation"""
    return T + co * Q + si * U


@njit
def scan_map_for_one_detector(tod_det, input_T, input_Q, input_U, pol_angle_det):
    for i in range(len(tod_det)):
        tod_det[i] += compute_signal_for_one_sample(
            T=input_T[i],
            Q=input_Q[i],
            U=input_U[i],
            co=np.cos(2 * pol_angle_det[i]),
            si=np.sin(2 * pol_angle_det[i]),
        )


def scan_map(
    tod,
    pointings,
    pol_angle,
    maps: Dict[str, np.ndarray],
    input_names,
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

    assert tod.shape == pointings.shape[0:2]

    for detector_idx in range(tod.shape[0]):
        if input_map_in_galactic:
            curr_pointings_det, curr_pol_angle_det = rotate_coordinates_e2g(
                pointings[detector_idx, :, :], pol_angle[detector_idx, :]
            )
        else:
            curr_pointings_det = pointings[detector_idx, :, :]
            curr_pol_angle_det = pol_angle[detector_idx, :]

        if input_names is None:
            maps_det = maps
        else:
            maps_det = maps[input_names[detector_idx]]

        nside = npix_to_nside(maps_det.shape[1])

        if interpolation in ["", None]:
            hpx = Healpix_Base(nside, "RING")
            pixel_ind_det = hpx.ang2pix(curr_pointings_det)

            scan_map_for_one_detector(
                tod_det=tod[detector_idx],
                input_T=maps_det[0, pixel_ind_det],
                input_Q=maps_det[1, pixel_ind_det],
                input_U=maps_det[2, pixel_ind_det],
                pol_angle_det=curr_pol_angle_det,
            )

        elif interpolation == "linear":
            scan_map_for_one_detector(
                tod_det=tod[detector_idx],
                input_T=hp.get_interp_val(
                    maps_det[0, :], curr_pointings_det[:, 0], curr_pointings_det[:, 1]
                ),
                input_Q=hp.get_interp_val(
                    maps_det[1, :], curr_pointings_det[:, 0], curr_pointings_det[:, 1]
                ),
                input_U=hp.get_interp_val(
                    maps_det[2, :], curr_pointings_det[:, 0], curr_pointings_det[:, 1]
                ),
                pol_angle_det=curr_pol_angle_det,
            )

        else:
            raise ValueError(
                "Wrong value for interpolation. It should be one of the following:\n"
                + '- "" for no interpolation\n'
                + '- "linear" for linear interpolation\n'
            )


def scan_map_in_observations(
    obs: Union[Observation, List[Observation]],
    maps: Dict[str, np.ndarray],
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    input_map_in_galactic: bool = True,
    component: str = "tod",
    interpolation: Union[str, None] = "",
):
    """Scan a map filling time-ordered data

    This is a wrapper around the :func:`.scan_map` function that applies to the TOD
    stored in `obs` and the pointings stored in `pointings`. The two types can either
    bed a :class:`.Observation` instance and a NumPy matrix, or a list
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
        # in `obs.sky_tod`
        scan_map_in_observations(sim.observations, …, component="sky_tod")

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

    for cur_obs, cur_ptg, cur_psi in zip(obs_list, ptg_list, psi_list):
        if type(maps) is dict:
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

        scan_map(
            tod=getattr(cur_obs, component),
            pointings=cur_ptg,
            pol_angle=cur_psi,
            maps=maps,
            input_names=input_names,
            input_map_in_galactic=input_map_in_galactic,
            interpolation=interpolation,
        )
