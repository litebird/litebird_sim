# -*- encoding: utf-8 -*-

from numba import njit
import numpy as np
import healpy as hp

from astropy.time import Time, TimeDelta

from typing import Union, List

from .observations import Observation

from .coordinates import rotate_coordinates_e2g

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
    maps,
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
    between two samples. Optionally it can return the polarization angle `pol_angle`
    and the pixel index `pixel_ind` in arrays of size N.
    """

    assert tod.shape == pointings.shape[0:2]

    for detector_idx in range(tod.shape[0]):
        curr_pointings = pointings[detector_idx, :, :]

        if input_map_in_galactic:
            rotate_coordinates_e2g(curr_pointings)

        maps_det = maps[input_names[detector_idx]]
        nside = hp.npix2nside(maps_det.shape[1])

        n_samples = len(curr_pointings[ :, 0])

        pixel_ind_det = hp.ang2pix(
            nside, curr_pointings[ :, 0], curr_pointings[ :, 1]
        )
        pol_angle_det = np.mod(
            curr_pointings[ :, 2]
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
    pointings,
    hwp_radpsec,
    maps: List,
    input_map_in_galactic: bool = True,
    fill_psi_and_pixind_in_obs: bool = False,
):
    """Scan a map filling time-ordered data

    This is a wrapper around the :func:`.scan_map` function that applies to the TOD
    stored in `obs`, which can either be one :class:`.Observation` instance or a list
    of observations.
    """

    if isinstance(obs, Observation):
        obs_list = [obs]
    else:
        obs_list = obs

    for cur_obs in obs_list:

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
                cur_obs.psi_and_pixind_coords = "Galactic"
            else:
                cur_obs.psi_and_pixind_coords = "Ecliptic"                

            scan_map(
                tod=cur_obs.tod,
                pointings=pointings,
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
                pointings=pointings,
                hwp_radpsec=hwp_radpsec,
                maps=maps,
                input_names=input_names,
                start_time_s=start_time_s,
                delta_time_s=delta_time_s,
                input_map_in_galactic=input_map_in_galactic,
            )
