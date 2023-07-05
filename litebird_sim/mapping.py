# -*- encoding: utf-8 -*-

# The implementation of the destriping algorithm provided here is based on the paper
# «Destriping CMB temperature and polarization maps» by Kurki-Suonio et al. 2009,
# A&A 506, 1511–1539 (2009), https://dx.doi.org/10.1051/0004-6361/200912361
#
# It is important to have that paper at hand while reading this code, as many
# functions and variable defined here use the same letters and symbols of that
# paper. We refer to it in code comments and docstrings as "KurkiSuonio2009".

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
class MapMakerResult:
    """Result of a call to any of the map-making functions, like :func:`.destripe`

    This dataclass has the following fields:

    - ``hit_map``: Healpix map containing the number of hit counts
      (integer values) per pixel

    - ``binned_map``: Healpix map containing the binned value for each pixel

    - ``destriped_map``: destriped Healpix map (if a destriper has been used)

    - ``npp``: covariance matrix elements for each pixel in the map

    - ``invnpp``: inverse of the covariance matrix element for each
      pixel in the map

    - ``rcond``: pixel condition number, stored as an Healpix map

    - ``coordinate_system``: the coordinate system of the output maps
      (a :class:`.CoordinateSistem` object)
    """

    hit_map: Any = None
    binned_map: Any = None
    destriped_map: Any = None
    npp: Any = None
    invnpp: Any = None
    rcond: Any = None
    coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic


@njit
def _solve_mapmaking(ata, atd):
    # Sove the map-making equation

    # Expected shape:
    # - `ata`: (N, 3, 3) is an array of N 3×3 matrices, where N is the number of pixels
    # - `atd`: (N, 3)
    npix = atd.shape[0]

    for ipix in range(npix):
        if np.linalg.cond(ata[ipix]) < COND_THRESHOLD:
            atd[ipix] = np.linalg.solve(ata[ipix], atd[ipix])
            ata[ipix] = np.linalg.inv(ata[ipix])
        else:
            ata[ipix].fill(hp.UNSEEN)
            atd[ipix].fill(hp.UNSEEN)


@njit
def _accumulate_map_and_info(tod, pix, psi, weights, info, additional_component: bool):
    # Fill the upper triangle of the information matrix and use the lower
    # triangle for the RHS of the map-making equation:
    #
    # 1. The upper triangle and the diagonal contains the coefficients in
    #    Eq. (10) of KurkiSuonio2009. This must be set just once, as it only
    #    depends on the pointing information The flag `additional_component`
    #    tells if this part must be calculated (``false``) or not
    # 2. The lower triangle contains the weighted sum of all the TOD components.
    assert tod.shape == pix.shape == psi.shape

    num_of_detectors = tod.shape[0]

    for idet in range(num_of_detectors):
        inv_sigma = 1.0 / np.sqrt(weights[idet])
        inv_sigma2 = inv_sigma * inv_sigma

        if not additional_component:
            # Fill the upper triangle
            for cur_pix_idx, cur_psi in zip(pix[idet], psi[idet]):
                cos_over_sigma = np.cos(2 * cur_psi) * inv_sigma
                sin_over_sigma = np.sin(2 * cur_psi) * inv_sigma
                info_pix = info[cur_pix_idx]

                # Upper triangle
                info_pix[0, 0] += inv_sigma2
                info_pix[0, 1] += inv_sigma * cos_over_sigma
                info_pix[0, 2] += inv_sigma * sin_over_sigma
                info_pix[1, 1] += cos_over_sigma * cos_over_sigma
                info_pix[1, 2] += sin_over_sigma * cos_over_sigma
                info_pix[2, 2] += sin_over_sigma * sin_over_sigma

        # Fill the lower triangle
        for cur_sample, cur_pix_idx, cur_psi in zip(tod[idet], pix[idet], psi[idet]):
            cos_over_sigma = np.cos(2 * cur_psi) * inv_sigma
            sin_over_sigma = np.sin(2 * cur_psi) * inv_sigma
            info_pix = info[cur_pix_idx]

            info_pix[1, 0] += cur_sample * inv_sigma2
            info_pix[2, 0] += cur_sample * cos_over_sigma * inv_sigma
            info_pix[2, 1] += cur_sample * sin_over_sigma * inv_sigma


@njit
def _numba_extract_map_and_fill_info2(info, rhs):
    # This is used internally by _extract_map_and_fill_info
    for idx in range(info.shape[0]):
        rhs[idx, 0] = info[idx, 1, 0]
        rhs[idx, 1] = info[idx, 2, 0]
        rhs[idx, 2] = info[idx, 2, 1]

        info[idx, 1, 0] = info[idx, 0, 1]
        info[idx, 2, 0] = info[idx, 0, 2]
        info[idx, 2, 1] = info[idx, 1, 2]


def _extract_map_and_fill_info(info):
    # Extract the RHS of the mapmaking equation from the lower triangle of info
    # and fill the lower triangle with the upper triangle, thus making each
    # matrix in "info" symmetric
    rhs = np.empty((info.shape[0], 3), dtype=info.dtype)

    # The implementation in Numba of this code is ~5 times faster than the older
    # implementation that used NumPy
    _numba_extract_map_and_fill_info2(info, rhs)

    return rhs


def make_bin_map(
    obs: Union[Observation, List[Observation]],
    nside: int,
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    do_covariance: bool = False,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    components: List[str] = ["tod"],
) -> MapMakerResult:
    """Bin Map-maker

    Map a list of observations

    Args:
        obs (list of :class:`Observations`): observations to be mapped. They
            are required to have the following attributes as arrays

            * `pointings`: the pointing information (in radians) for each tod
               sample
            * `psi`: the polarization angle (in radians) for each tod sample
            * any attribute listed in `components` (by default, `tod`) and
              containing the TOD(s) to be binned together

            If the observations are distributed over some communicator(s), they
            must share the same group processes.
            If pointings and psi are not included in the observations, they can
            be provided through an array (or a list of arrays) of dimension
            (Ndetectors x Nsamples x 3), containing theta, phi and psi
        nside (int): HEALPix nside of the output map
        pointings (array or list of arrays): optional, external pointing
            information, if not included in the observations
        do_covariance (bool): optional, if true it returns also covariance
        output_coordinate_system (:class:`.CoordinateSystem`): the coordinates
            to use for the output map
        components (list[str]): list of components to include in the map-making.
            The default is just to use the field ``tod`` of each
            :class:`.Observation` object.

    Returns:
        array: T, Q, U maps (stacked). The shape is `(3, 12 * nside * nside)`.
            All the detectors of all the observations contribute to the map.
            If the observations are distributed over some communicator(s), all
            the processes (contribute and) hold a copy of the map.
            The covariance matrix is an array of shape
            `(12 * nside * nside, 3, 3)`
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
            weights = cur_obs.sampling_rate_hz * cur_obs.net_ukrts**2
        except AttributeError:
            weights = np.ones(cur_obs.n_detectors)

        first_component = getattr(cur_obs, components[0])
        ndets = first_component.shape[0]
        pixidx_all = np.empty_like(first_component, dtype=int)
        polang_all = np.empty_like(first_component)

        for idet in range(ndets):
            if output_coordinate_system == CoordinateSystem.Galactic:
                curr_pointings_det, polang_all[idet] = rotate_coordinates_e2g(
                    cur_ptg[idet, :, :], cur_psi[idet, :]
                )
            else:
                curr_pointings_det = cur_ptg[idet, :, :]
                polang_all[idet] = cur_psi[idet, :]

            pixidx_all[idet] = hpx.ang2pix(curr_pointings_det)

        if output_coordinate_system == CoordinateSystem.Galactic:
            # free curr_pointings_det in case of output_map_in_galactic
            del curr_pointings_det

        for idx, cur_component_name in enumerate(components):
            cur_component = getattr(cur_obs, cur_component_name)
            assert (
                cur_component.shape == first_component.shape
            ), 'The two TODs "{}" and "{}" do not have a matching shape'.format(
                components[0], cur_component_name
            )
            _accumulate_map_and_info(
                cur_component,
                pixidx_all,
                polang_all,
                weights,
                info,
                additional_component=idx > 0,
            )

        del pixidx_all, polang_all

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

    _solve_mapmaking(info, rhs)

    return MapMakerResult(
        binned_map=rhs.T,
        invnpp=info,
        coordinate_system=output_coordinate_system,
    )


def _split_items_into_n_segments(n: int, num_of_segments: int) -> List[int]:
    """Divide a quantity `length` into chunks, each roughly of the same length

    This low-level function is used to determine how many samples in a TOD should be
    collected by the destriper within the same baseline.

    .. testsetup::

        from litebird_sim.mapping import _split_into_n

    .. testcode::

        # Divide 10 items into 4 groups, so that each of them will
        # have roughly the same number of items
        print(split_into_n(10, 4))

    .. testoutput::

        [2 3 2 3]
    """
    assert num_of_segments > 0, f"num_of_segments={num_of_segments} is not positive"
    assert (
        n >= num_of_segments
    ), f"n={n} is smaller than num_of_segments={num_of_segments}"

    start_positions = np.array(
        [int(i * n / num_of_segments) for i in range(num_of_segments + 1)],
        dtype="int",
    )
    return start_positions[1:] - start_positions[0:-1]


def split_items_evenly(n: int, sub_n: int) -> List[int]:
    """Evenly split `n` of items into groups, each with roughly `sublength` elements

    .. testsetup::

        from litebird_sim.mapping import split

    .. testcode::

        # Divide 10 items into groups, so that each of them will contain
        # roughly 4 items
        print(split(10, 4))

    .. testoutput::

        [3 3 4]

    """
    assert sub_n > 0, "sub_n={0} is not positive".format(sub_n)
    assert sub_n < n, "sub_n={0} is not smaller than n={1}".format(sub_n, n)

    return _split_items_into_n_segments(n=n, num_of_segments=int(np.ceil(n / sub_n)))
