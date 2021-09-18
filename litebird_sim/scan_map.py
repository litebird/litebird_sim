# -*- encoding: utf-8 -*-

from numba import njit
import numpy as np
import healpy as hp

from astropy.time import Time

from typing import Union

from .observations import Observation


@njit
def compute_signal_for_one_sample(T, Q, U, co, si):
    return 0.5 * (T + co * Q + si * U)


@njit
def scan_map_for_one_detector(tod_det, pixel_ind_det, pol_angle_det, maps):

    for i in range(len(tod_det)):

        co = np.cos(2 * pol_angle_det[i])
        si = np.sin(2 * pol_angle_det[i])

        tod_det[i] += compute_signal_for_one_sample(
            T=maps[0, pixel_ind_det[i]],
            Q=maps[1, pixel_ind_det[i]],
            U=maps[2, pixel_ind_det[i]],
            co=co,
            si=si,
        )


def scan_map(
    tod,
    pointings,
    hwp_radpsec,
    maps,
    input_names,
    start_time_s,
    delta_time_s,
    pol_angle: Union[np.ndarray, None] = None,
    pixel_ind: Union[np.ndarray, None] = None,
):

    assert tod.shape == pointings.shape[0:2]

    for detector_idx in range(tod.shape[0]):

        maps_det = maps[input_names[detector_idx]]
        nside = hp.npix2nside(maps_det.shape[1])

        n_samples = len(pointings[detector_idx, :, 0])

        pixel_ind_det = hp.ang2pix(
            nside, pointings[detector_idx, :, 0], pointings[detector_idx, :, 1]
        )
        pol_angle_det = (
            pointings[detector_idx, :, 2]
            + 2 * (start_time_s + np.arange(n_samples) * delta_time_s) * hwp_radpsec
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
    fill_psi_and_pixel_in_obs: bool = False,
):
    if isinstance(obs, Observation):
        obs_list = [obs]
    else:
        obs_list = obs

    for cur_obs in obs_list:

        if cur_obs.name[0] in maps:
            input_names = cur_obs.name
        else:
            input_names = cur_obs.channel

        if isinstance(obs.start_time, Time):
            start_time_s = (cur_obs.start_time - cur_obs.start_time_global).sec
        else:
            start_time_s = cur_obs.start_time - cur_obs.start_time_global

        if fill_psi_and_pixind_in_obs:
            cur_obs.psi = np.empty_like(cur_obs.tod)
            cur_obs.pixind = np.empty_like(cur_obs.tod, dtype=np.int)

            scan_map(
                tod=cur_obs.tod,
                pointings=pointings,
                hwp_radpsec=hwp_radpsec,
                maps=maps,
                input_names=input_names,
                start_time_s=start_time_s,
                delta_time_s=cur_obs.get_delta_time().value,
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
                delta_time_s=cur_obs.get_delta_time().value,
            )            
