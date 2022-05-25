# -*- encoding: utf-8 -*-

import numpy as np
from numba import njit
import healpy as hp

from typing import Union, List

from .observations import Observation

from . import mpi


COND_THRESHOLD = 1e10


@njit
def _accumulate_map_and_info(tod, pix, psi, weights, info):
    # Fill the upper triangle of the information matrix and use the lower
    # triangle for the RHS of the map-making equation
    assert tod.shape == pix.shape == psi.shape

    ndets = tod.shape[0]
    nsamples = tod.shape[1]

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
    obss: Union[Observation, List[Observation]], nside, docovariance=False
):
    """Bin Map-maker

    Map a list of observations

    Args:
        obss (list of :class:`Observations`): observations to be mapped. They
            are required to have the following attributes as arrays of identical
            shapes

            * `tod`: the time-ordered data to be mapped
            * `pixind`: the index of the pixel observed for each tod sample in a
              HEALpix map at nside `nside`
            * `psi`: the polarization angle (in radians) for each tod sample

            If the observations are distributed over some communicator(s), they
            must share the same group processes.
        nside (int): HEALPix nside of the output map
    Returs:
        array: T, Q, U maps (stacked). The shape is `(3, 12 * nside * nside)`.
            All the detectors of all the observations contribute to the map.
            If the observations are distributed over some communicator(s), all
            the processes (contribute and) hold a copy of the map
    """
    n_pix = hp.nside2npix(nside)
    info = np.zeros((n_pix, 3, 3))

    if isinstance(obss, Observation):
        obs_list = [obss]
    else:
        obs_list = obss

    for obs in obs_list:
        try:
            weights = obs.sampling_rate_shz * obs.net_ukrts ** 2
        except:
            weights = np.ones(obs.n_detectors)

            _accumulate_map_and_info(obs.tod, obs.pixind, obs.psi, weights, info)

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

    if docovariance:
        try:
            return res, np.linalg.inv(info)
        except np.linalg.LinAlgError:
            covmat = np.full_like(info, hp.UNSEEN)
            covmat[mask] = np.linalg.inv(info[mask])
            return res, covmat
    else:
        return res
