# -*- encoding: utf-8 -*-

from numba import njit
import numpy as np

from ducc0.healpix import Healpix_Base

from astropy.time import Time, TimeDelta

from typing import Union, List, Dict

from .observations import Observation

from .coordinates import rotate_coordinates_e2g, CoordinateSystem

from .healpix import npix_to_nside


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
    pol_angle,
    maps: Dict[str, np.ndarray],
    input_names,
    input_map_in_galactic,
):
    """Scan a map filling time-ordered data

    This function modifies the values in `tod` by adding the contribution of the
    bolometric equation given a list of TQU maps `maps`. The `pointings` argument
    must be a DxN×2 matrix containing the pointing information, where D is the number
    of detector for the current observation and N is the size of the `tod` array.
    `pol_angle` is the array of size DxN containing the polarization angle in radiants.
    `input_names` is an array containing the keywords that allow to select the proper
    input in `maps` for each detector in the TOD. If `input_map_in_galactic` is set to
    False the input map is assumed in ecliptic coordinates, default galactic.
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

        hpx = Healpix_Base(nside, "RING")
        pixel_ind_det = hpx.ang2pix(curr_pointings_det)

        scan_map_for_one_detector(
            tod_det=tod[detector_idx],
            pixel_ind_det=pixel_ind_det,
            pol_angle_det=curr_pol_angle_det,
            maps=maps_det,
        )


def scan_map_in_observations(
    obs: Union[Observation, List[Observation]],
    maps: Dict[str, np.ndarray],
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    input_map_in_galactic: bool = True,
):
    """Scan a map filling time-ordered data

    This is a wrapper around the :func:`.scan_map` function that applies to the TOD
    stored in `obs` and the pointings stored in `pointings`. The two types can either
    bed a :class:`.Observation` instance and a NumPy matrix, or a list
    of observations and a list of NumPy matrices; in the latter case, they must have
    the same number of elements.
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
        else:
            assert isinstance(maps, np.ndarray), (
                "maps must either a dictionary contaning keys for all the"
                + "channels/detectors, or be a numpy array of dim (3 x Npix)"
            )
            input_names = None

        scan_map(
            tod=cur_obs.tod,
            pointings=cur_ptg,
            pol_angle=cur_psi,
            maps=maps,
            input_names=input_names,
            input_map_in_galactic=input_map_in_galactic,
        )
