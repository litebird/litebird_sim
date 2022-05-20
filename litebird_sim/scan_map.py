# -*- encoding: utf-8 -*-

from numba import njit
import numpy as np
import healpy as hp

from astropy.time import Time, TimeDelta

from typing import Union, List, Dict

from .observations import Observation

from .coordinates import rotate_coordinates_e2g, CoordinateSystem


@njit
def compute_signal_for_one_sample(T, Q, U, co, si):
    """Bolometric equation"""
    return T + co * Q + si * U


@njit
def scan_map_for_one_detector(tod_det, pixel_ind_det, pol_angle_det, maps):

    for i in range(len(tod_det)):

        tod_det[i] += compute_signal_for_one_sample(
            T=maps[0, pixel_ind_det[i]],
            Q=maps[1, pixel_ind_det[i]],
            U=maps[2, pixel_ind_det[i]],
            co=np.cos(2 * pol_angle_det[i]),
            si=np.sin(2 * pol_angle_det[i]),
        )


def scan_map(
    tod,
    pointings,
    hwp_radpsec,
    maps: Dict[str, np.ndarray],
    input_names,
    start_time_s,
    delta_time_s,
    input_map_in_galactic,
    pol_angle: Union[np.ndarray, None] = None,
    pixel_ind: Union[np.ndarray, None] = None,
):
    """Scan a map filling time-ordered data

    This function modifies the values in `tod` by adding the contribution of the
    bolometric equation given a list of TQU maps `maps`. The `pointings` argument
    must be a N×3 matrix containing the pointing information, where N is the size
    of the `tod` array. `hwp_radpsec` is the hwp rotation speed in radiants per
    second.`input_names` is an array containing the keywords that allow to select
    the proper input in `maps` for each detector in the TOD. `start_time_s` and
    `delta_time_s` are respectively the start time of the TOD and the time step
    between two samples. If `input_map_in_galactic` is set to False the input map
    is assumed in ecliptic coordinates, default galactic. Optionally it can return
    the polarization angle `pol_angle` and the pixel index `pixel_ind` in arrays
    of size N.
    """

    assert tod.shape == pointings.shape[0:2]

    for detector_idx in range(tod.shape[0]):

        if input_map_in_galactic:
            curr_pointings = rotate_coordinates_e2g(pointings[detector_idx, :, :])
        else:
            curr_pointings = pointings[detector_idx, :, :]

        maps_det = maps[input_names[detector_idx]]
        nside = hp.npix2nside(maps_det.shape[1])

        n_samples = len(curr_pointings[:, 0])

        pixel_ind_det = hp.ang2pix(nside, curr_pointings[:, 0], curr_pointings[:, 1])
        pol_angle_det = np.mod(
            curr_pointings[:, 2]
            + 2 * (start_time_s + np.arange(n_samples) * delta_time_s) * hwp_radpsec,
            2 * np.pi,
        )

        scan_map_for_one_detector(
            tod_det=tod[detector_idx],
            pixel_ind_det=pixel_ind_det,
            pol_angle_det=pol_angle_det,
            maps=maps_det,
        )

        if pixel_ind is not None:
            pixel_ind[detector_idx] = pixel_ind_det

        if pol_angle is not None:
            pol_angle[detector_idx] = pol_angle_det


def scan_map_in_observations(
    obs: Union[Observation, List[Observation]],
    pointings: Union[np.ndarray, List[np.ndarray]],
    hwp_radpsec,
    maps: Dict[str, np.ndarray],
    input_map_in_galactic: bool = True,
    fill_psi_and_pixind_in_obs: bool = False,
):
    """Scan a map filling time-ordered data

    This is a wrapper around the :func:`.scan_map` function that applies to the TOD
    stored in `obs` and the pointings stored in `pointings`. The two types can either
    bed a :class:`.Observation` instance and a NumPy matrix, or a list
    of observations and a list of NumPy matrices; in the latter case, they must have
    the same number of elements.
    """

    if isinstance(obs, Observation):
        assert isinstance(pointings, np.ndarray), (
            "You must pass a list of observations *and* a list "
            + "of pointing matrices to scan_map_in_observations"
        )
        obs_list = [obs]
        ptg_list = [pointings]
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
        ptg_list = pointings

    for cur_obs, cur_ptg in zip(obs_list, ptg_list):

        if cur_obs.name[0] in maps:
            input_names = cur_obs.name
        else:
            input_names = cur_obs.channel

        if isinstance(cur_obs.start_time, Time):
            start_time_s = (cur_obs.start_time - cur_obs.start_time_global).sec
        else:
            start_time_s = cur_obs.start_time - cur_obs.start_time_global

        if isinstance(cur_obs.get_delta_time(), TimeDelta):
            delta_time_s = cur_obs.get_delta_time().value
        else:
            delta_time_s = cur_obs.get_delta_time()

        if fill_psi_and_pixind_in_obs:
            cur_obs.psi = np.empty_like(cur_obs.tod)
            cur_obs.pixind = np.empty_like(cur_obs.tod, dtype=np.int32)

            if input_map_in_galactic:
                cur_obs.psi_and_pixind_coords = CoordinateSystem.Galactic
            else:
                cur_obs.psi_and_pixind_coords = CoordinateSystem.Ecliptic

            scan_map(
                tod=cur_obs.tod,
                pointings=cur_ptg,
                hwp_radpsec=hwp_radpsec,
                maps=maps,
                input_names=input_names,
                start_time_s=start_time_s,
                delta_time_s=delta_time_s,
                input_map_in_galactic=input_map_in_galactic,
                pol_angle=cur_obs.psi,
                pixel_ind=cur_obs.pixind,
            )
        else:
            scan_map(
                tod=cur_obs.tod,
                pointings=cur_ptg,
                hwp_radpsec=hwp_radpsec,
                maps=maps,
                input_names=input_names,
                start_time_s=start_time_s,
                delta_time_s=delta_time_s,
                input_map_in_galactic=input_map_in_galactic,
            )
