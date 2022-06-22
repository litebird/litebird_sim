# -*- encoding: utf-8 -*-

import numpy as np
from numba import njit
import healpy as hp

from typing import Union, List

from .observations import Observation

from .coordinates import rotate_coordinates_e2g

from . import mpi

from ducc0.healpix import Healpix_Base

from .healpix import nside_to_npix

COND_THRESHOLD = 1e10


@njit
def _accumulate_map_and_info(tod, pix, psi, weights, info):
    # Fill the upper triangle of the information matrix and use the lower
    # triangle for the RHS of the map-making equation
    assert tod.shape == pix.shape == psi.shape

    ndets = tod.shape[0]

    for idet in range(ndets):
        for d, p, a in zip(tod[idet], pix[idet], psi[idet]):
            one = 1.0 / np.sqrt(weights[idet])
            cos = np.cos(2 * a) / np.sqrt(weights[idet])
            sin = np.sin(2 * a) / np.sqrt(weights[idet])
            info_pix = info[p]
            info_pix[0, 0] += one * one
            info_pix[0, 1] += one * cos
            info_pix[0, 2] += one * sin
            info_pix[1, 1] += cos * cos
            info_pix[1, 2] += sin * cos
            info_pix[2, 2] += sin * sin
            info_pix[1, 0] += d * one * one
            info_pix[2, 0] += d * cos * one
            info_pix[2, 1] += d * sin * one


def _extract_map_and_fill_info(info):
    # Extract the RHS of the mapmaking equation from the lower triangle of info
    # and fill it in with the upper triangle
    ilr = np.array([1, 2, 2])
    ilc = np.array([0, 0, 1])
    rhs = info[:, ilr, ilc]
    info[:, ilr, ilc] = info[:, ilc, ilr]
    return rhs


def make_bin_map(
    obs: Union[Observation, List[Observation]],
    nside: int,
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    do_covariance: bool = False,
    output_map_in_galactic: bool = True,
):
    """Bin Map-maker

    Map a list of observations

    Args:
        obss (list of :class:`Observations`): observations to be mapped. They
            are required to have the following attributes as arrays

            * `tod`: the time-ordered data to be mapped
            * `pointings`: the pointing information (in radians) for each tod
               sample
            * `psi`: the polarization angle (in radians) for each tod sample

            If the observations are distributed over some communicator(s), they
            must share the same group processes.
            If pointings and psi are not included in the observations, they can
            be provided through an array (or a list of arrays) of dimension
            (Ndetectors x Nsamples x 3), containing theta, phi and psi
        nside (int): HEALPix nside of the output map
        pointings (array or list of arrays): optional, external pointing
            information, if not included in the observations
        do_covariance (bool): optional, if true it returns also covariance
        output_map_in_galactic (bool): optional, if true maps in Galactic
            coordinates

    Returns:
        array: T, Q, U maps (stacked). The shape is `(3, 12 * nside * nside)`.
            All the detectors of all the observations contribute to the map.
            If the observations are distributed over some communicator(s), all
            the processes (contribute and) hold a copy of the map.
            Optionally can return the covariance matrix in an array of shape
            `(12 * nside * nside, 3, 3)`
            Map and covariance are in Galactic coordinates unless
            output_map_in_galactic is set to False
    """

    hpx = Healpix_Base(nside, "RING")

    n_pix = nside_to_npix(nside)
    info = np.zeros((n_pix, 3, 3))

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
                "When you pass a list of observations to make_bin_map, "
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
        try:
            weights = cur_obs.sampling_rate_ghz * cur_obs.net_ukrts**2
        except AttributeError:
            weights = np.ones(cur_obs.n_detectors)

        ndets = cur_obs.tod.shape[0]
        pixidx_all = np.empty_like(cur_obs.tod, dtype=int)
        polang_all = np.empty_like(cur_obs.tod)

        for idet in range(ndets):
            if output_map_in_galactic:
                curr_pointings_det, curr_pol_angle_det = rotate_coordinates_e2g(
                    cur_ptg[idet, :, :], cur_psi[idet, :]
                )
            else:
                curr_pointings_det = cur_ptg[idet, :, :]
                curr_pol_angle_det = cur_psi[idet, :]

            pixidx_all[idet] = hpx.ang2pix(curr_pointings_det)
            polang_all[idet] = curr_pol_angle_det

        _accumulate_map_and_info(cur_obs.tod, pixidx_all, polang_all, weights, info)

    if all([obs.comm is None for obs in obs_list]) or not mpi.MPI_ENABLED:
        # Serial call
        pass
    elif all(
        [
            mpi.MPI.Comm.Compare(obs_list[i].comm, obs_list[i + 1].comm) < 2
            for i in range(len(obs_list) - 1)
        ]
    ):
        info = obs_list[0].comm.allreduce(info, mpi.MPI.SUM)
    else:
        raise NotImplementedError(
            "All observations must be distributed over the same MPI groups"
        )

    rhs = _extract_map_and_fill_info(info)
    try:
        res = np.linalg.solve(info, rhs)
    except np.linalg.LinAlgError:
        cond = np.linalg.cond(info)
        res = np.full_like(rhs, hp.UNSEEN)
        mask = cond < COND_THRESHOLD
        res[mask] = np.linalg.solve(info[mask], rhs[mask])

    if do_covariance:
        try:
            return res.T, np.linalg.inv(info)
        except np.linalg.LinAlgError:
            covmat = np.full_like(info, hp.UNSEEN)
            covmat[mask] = np.linalg.inv(info[mask])
            return res.T, covmat
    else:
        return res.T
