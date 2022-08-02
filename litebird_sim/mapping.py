# -*- encoding: utf-8 -*-
from dataclasses import dataclass

import numpy as np
from numba import njit
import healpy as hp

from typing import Union, List, Any

from .observations import Observation

from .coordinates import rotate_coordinates_e2g, CoordinateSystem

from . import mpi

from ducc0.healpix import Healpix_Base

from .healpix import nside_to_npix

COND_THRESHOLD = 1e10


@dataclass
class DestriperParameters:
    """Parameters used by the destriper to produce a map.

    The list of fields in this dataclass is the following:

    - ``nside``: the NSIDE parameter used to create the maps

    - ``coordinate_system``: an instance of the :class:`.CoordinateSystem` enum.
      It specifies if the map must be created in ecliptic (default) or
      galactic coordinates.

    - ``nnz``: number of components per pixel. The default is 3 (I/Q/U).

    - ``baseline_length_s``: length of the baseline for 1/f noise in seconds

    - ``iter_max``: maximum number of iterations

    - ``output_file_prefix``: prefix to be used for the filenames of the
      Healpix FITS maps saved in the output directory

    The following Boolean flags specify which maps should be returned
    by the function :func:`.destripe`:

    - ``return_hit_map``: return the hit map (number of hits per
      pixel)

    - ``return_binned_map``: return the binned map (i.e., the map with
      no baselines removed).

    - ``return_destriped_map``: return the destriped map. If pure
      white noise is present in the timelines, this should be the same
      as the binned map.

    - ``return_npp``: return the map of the white noise covariance per
      pixel. It contains the following fields: ``II``, ``IQ``, ``IU``,
      ``QQ``, ``QU``, and ``UU`` (in this order).

    - ``return_invnpp``: return the map of the inverse covariance per
      pixel. It contains the following fields: ``II``, ``IQ``, ``IU``,
      ``QQ``, ``QU``, and ``UU`` (in this order).

    - ``return_rcond``: return the map of condition numbers.

    The default is to only return the destriped map.

    """

    nside: int = 512
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic
    nnz: int = 3
    baseline_length_s: float = 60.0
    iter_max: int = 100
    output_file_prefix: str = "lbs_"
    return_hit_map: bool = False
    return_binned_map: bool = False
    return_destriped_map: bool = True
    return_npp: bool = False
    return_invnpp: bool = False
    return_rcond: bool = False


@dataclass
class DestriperResult:
    """Result of a call to :func:`.destripe`

    This dataclass has the following fields:

    - ``hit_map``: Healpix map containing the number of hit counts
      (integer values) per pixel

    - ``binned_map``: Healpix map containing the binned value for each pixel

    - ``destriped_map``: destriped Healpix mapmaker

    - ``npp``: covariance matrix elements for each pixel in the map

    - ``invnpp``: inverse of the covariance matrix element for each
      pixel in the map

    - ``rcond``: pixel condition number, stored as an Healpix map

    """

    hit_map: Any = None
    binned_map: Any = None
    destriped_map: Any = None
    npp: Any = None
    invnpp: Any = None
    rcond: Any = None
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic


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
