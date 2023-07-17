# -*- encoding: utf-8 -*-

# The implementation of the destriping algorithm provided here is based on the paper
# «Destriping CMB temperature and polarization maps» by Kurki-Suonio et al. 2009,
# A&A 506, 1511–1539 (2009), https://dx.doi.org/10.1051/0004-6361/200912361
#
# It is important to have that paper at hand while reading this code, as many
# functions and variable defined here use the same letters and symbols of that
# paper. We refer to it in code comments and docstrings as "KurkiSuonio2009".

from dataclasses import dataclass
import gc

import mpi4py.MPI
import numpy as np
import numpy.typing as npt
from ducc0.healpix import Healpix_Base
from numba import njit
import healpy as hp

from litebird_sim.mpi import MPI_ENABLED, MPI_COMM_WORLD
from typing import Union, List, Any, Optional
from litebird_sim.observations import Observation
from litebird_sim.coordinates import rotate_coordinates_e2g, CoordinateSystem

from .common import (
    _compute_pixel_indices,
    _normalize_observations_and_pointings,
    COND_THRESHOLD,
    get_map_making_weights,
    cholesky,
    estimate_cond_number,
)


def _split_items_into_n_segments(n: int, num_of_segments: int) -> List[int]:
    """Divide a quantity `length` into chunks, each roughly of the same length

    This low-level function is used to determine how many samples in a TOD should be
    collected by the toast_destriper within the same baseline.

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

        from litebird_sim.mapping import split_items_evenly

    .. testcode::

        # Divide 10 items into groups, so that each of them will contain
        # roughly 4 items
        print(split_items_evenly(10, 4))

    .. testoutput::

        [3 3 4]

    """
    assert sub_n > 0, "sub_n={0} is not positive".format(sub_n)
    assert sub_n < n, "sub_n={0} is not smaller than n={1}".format(sub_n, n)

    return _split_items_into_n_segments(n=n, num_of_segments=int(np.ceil(n / sub_n)))


@dataclass
class NobsMatrix:
    # Shape: (Npix, 6)
    nobs_matrix: npt.NDArray
    # Shape: (Npix,) (array of Boolean flags)
    valid_pixel: npt.NDArray
    # True if `nobs_matrix` contains the Cholesky decomposition of M_i
    is_cholesky: bool


@dataclass
class DestriperParameters:
    nside: int = 256
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic
    nnz: int = 3
    samples_per_baseline: Union[int, List[int]] = 100
    iter_max: int = 100
    threshold: float = 1e-7


@dataclass
class DestriperResult:
    params: DestriperParameters
    hit_map: npt.ArrayLike
    binned_map: npt.ArrayLike
    nobs_matrix_cholesky: npt.ArrayLike
    coordinate_system: CoordinateSystem
    # The following fields are filled only if the CG algorithm was used
    baselines: Optional[npt.ArrayLike]
    baseline_lengths: Optional[npt.ArrayLike]
    stopping_factors: Optional[npt.ArrayLike]
    destriped_map: Optional[npt.ArrayLike]


# @njit
def _solve_map_making(nobs_matrix, atd):
    # Apply M⁻¹ to `atd`
    #
    # The parameter `nobs_matrix` is the M matrix (eq. 9 in KurkiSuonio2009).
    #
    # This is the same as _solve_binning (see above), but as it needs to be
    # called iteratively it does not alter `nobs_matrix`.
    #
    # Expected shape:
    # - `nobs_matrix`: (N_p, 3, 3) is an array of N_p 3×3 matrices, where
    #   N_p is the number of pixels in the map
    # - `atd`: (N_p, 3)
    npix = atd.shape[0]

    for ipix in range(npix):
        # TODO: we should save the value of `cond` somewhere instead of calculating it
        #       again every time
        if np.linalg.cond(nobs_matrix[ipix]) < COND_THRESHOLD:
            atd[ipix] = np.linalg.solve(nobs_matrix[ipix], atd[ipix])
        else:
            atd[ipix].fill(hp.UNSEEN)


# @njit
def _accumulate_nobs_matrix(
    pix_idx: npt.ArrayLike,  # Shape: (Ndet, 1)
    psi_angle_rad: npt.ArrayLike,  # Shape: (Ndet, 1)
    weights: npt.ArrayLike,  # Shape: (N_det,)
    nobs_matrix: npt.ArrayLike,  # Shape: (N_pix, 6)
) -> None:
    # Fill the upper triangle of the N_obs matrix following Eq. (10)
    # of KurkiSuonio2009. This must be set just once during destriping,
    # as it only depends on the pointing information
    #
    # Note that nobs_matrix must have been set to 0 before starting
    # calling this function!

    assert pix_idx.shape == psi_angle_rad.shape

    num_of_detectors = pix_idx.shape[0]

    for det_idx in range(num_of_detectors):
        inv_sigma = 1.0 / np.sqrt(weights[det_idx])
        inv_sigma2 = inv_sigma * inv_sigma

        # Fill the lower triangle of M_i only for i = 1…N_pix
        for cur_pix_idx, cur_psi in zip(pix_idx[det_idx], psi_angle_rad[det_idx]):
            cos_over_sigma = np.cos(2 * cur_psi) * inv_sigma
            sin_over_sigma = np.sin(2 * cur_psi) * inv_sigma
            cur_matrix = nobs_matrix[cur_pix_idx]

            cur_matrix[0] += inv_sigma2
            cur_matrix[1] += cos_over_sigma * inv_sigma
            cur_matrix[2] += cos_over_sigma * cos_over_sigma
            cur_matrix[3] += sin_over_sigma * inv_sigma
            cur_matrix[4] += sin_over_sigma * cos_over_sigma
            cur_matrix[5] += sin_over_sigma * sin_over_sigma


# @njit
def _nobs_matrix_to_cholesky(
    nobs_matrix: npt.ArrayLike,  # Shape: (N_pix, 6)
    dest_valid_pixel: npt.ArrayLike,  # Shape: (N_pix,)
    dest_nobs_matrix_cholesky: npt.ArrayLike,  # Shape: (N_pix, 6)
) -> None:
    # Apply `cholesky` iteratively on all the input maps in `nobs_matrix`
    # and save each result in `nobs_matrix_cholesky`
    for i in range(nobs_matrix.shape[0]):
        cur_nobs_matrix = nobs_matrix[i]
        (cond_number, flag) = estimate_cond_number(
            a00=cur_nobs_matrix[0],
            a10=cur_nobs_matrix[1],
            a11=cur_nobs_matrix[2],
            a20=cur_nobs_matrix[3],
            a21=cur_nobs_matrix[4],
            a22=cur_nobs_matrix[5],
        )
        dest_valid_pixel[i] = flag and (cond_number < COND_THRESHOLD)
        if dest_valid_pixel[i]:
            cholesky(
                a00=cur_nobs_matrix[0],
                a10=cur_nobs_matrix[1],
                a11=cur_nobs_matrix[2],
                a20=cur_nobs_matrix[3],
                a21=cur_nobs_matrix[4],
                a22=cur_nobs_matrix[5],
                dest_L=dest_nobs_matrix_cholesky[i],
            )
        # There is no `else`: we assume that
        # `nobs_matrix_cholesky` was already initialized
        # to zero everywhere


def _store_pixel_idx_and_pol_angle_in_obs(
    hpx: Healpix_Base,
    obs_list: List[Observation],
    ptg_list: List[npt.ArrayLike],
    psi_list: List[npt.ArrayLike],
    output_coordinate_system: CoordinateSystem,
):
    for cur_obs, cur_ptg, cur_psi in zip(obs_list, ptg_list, psi_list):
        cur_obs.destriper_weights = get_map_making_weights(cur_obs, check=True)

        (
            cur_obs.destriper_pixel_idx,
            cur_obs.destriper_pol_angle_rad,
        ) = _compute_pixel_indices(
            hpx=hpx,
            pointings=cur_ptg,
            psi=cur_psi,
            output_coordinate_system=output_coordinate_system,
        )


def _build_nobs_matrix(
    hpx: Healpix_Base,
    obs_list: List[Observation],
    ptg_list: List[npt.ArrayLike],
    psi_list: List[npt.ArrayLike],
) -> NobsMatrix:
    # Instead of a shape like (Npix, 3, 3), i.e., one 3×3 matrix per each
    # pixel, we only store the lower triangular part in a 6-element array.
    # In this way we reduce the memory usage by ~30% and the code is faster too.
    nobs_matrix = np.zeros((hpx.npix(), 6))  # Do not use np.empty() here!

    for cur_obs, cur_ptg, cur_psi in zip(obs_list, ptg_list, psi_list):
        _accumulate_nobs_matrix(
            pix_idx=cur_obs.destriper_pixel_idx,
            psi_angle_rad=cur_obs.destriper_pol_angle_rad,
            weights=cur_obs.destriper_weights,
            nobs_matrix=nobs_matrix,
        )

    # Now we must accumulate the result of every MPI process
    if MPI_ENABLED:
        MPI_COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE, nobs_matrix, op=mpi4py.MPI.SUM)

    # `nobs_matrix_cholesky` will *not* contain the M_i maps shown in
    # Eq. 9 of KurkiSuonio2009, but its Cholesky Decomposition, i.e.,
    # a lower-triangular matrix L such that M_i = L_i · L_i†.
    nobs_matrix_cholesky = np.zeros((hpx.npix(), 6))  # Do not use np.empty() here!
    valid_pixel = np.empty(hpx.npix(), dtype=np.bool_)
    _nobs_matrix_to_cholesky(
        nobs_matrix=nobs_matrix,
        dest_valid_pixel=valid_pixel,
        dest_nobs_matrix_cholesky=nobs_matrix_cholesky,
    )

    # We can get rid of all the M_i, as we are going to use their
    # Cholesky's decompositions from now on
    del nobs_matrix

    return NobsMatrix(
        nobs_matrix=nobs_matrix_cholesky,
        valid_pixel=valid_pixel,
        is_cholesky=True,
    )


# @njit
def _update_binned_map_local(
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    tod: npt.ArrayLike,
    pol_angle_rad: npt.ArrayLike,
    pixel_idx: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> None:
    for det_idx in range(tod.shape[0]):
        cur_weight = weights[det_idx]

        for i in range(tod.shape[1]):
            sin_term = np.sin(2 * pol_angle_rad[det_idx, i])
            cos_term = np.cos(2 * pol_angle_rad[det_idx, i])

            cur_pix = pixel_idx[det_idx, i]
            cur_sample = tod[det_idx, i]
            sky_map[0, cur_pix] += cur_sample / cur_weight
            sky_map[1, cur_pix] += cur_sample * cos_term / cur_weight
            sky_map[2, cur_pix] += cur_sample * sin_term / cur_weight
            hit_map[cur_pix] += 1.0 / cur_weight


def _update_binned_map(
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    tod: npt.ArrayLike,
    pol_angle_rad: npt.ArrayLike,
    pixel_idx: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> None:
    _update_binned_map_local(
        sky_map=sky_map,
        hit_map=hit_map,
        tod=tod,
        pol_angle_rad=pol_angle_rad,
        pixel_idx=pixel_idx,
        weights=weights,
    )

    if MPI_ENABLED:
        MPI_COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE, sky_map, op=mpi4py.MPI.SUM)
        MPI_COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE, hit_map, op=mpi4py.MPI.SUM)


def _reset_maps_for_destriper(params: DestriperResult) -> None:
    params.hit_map[:] = 0
    params.binned_map[:] = 0.0

    if params.destriped_map is not None:
        params.destriped_map[:] = 0.0


def make_destriped_map(
    obs: Union[Observation, List[Observation]],
    pointings: Optional[Union[npt.ArrayLike, List[npt.ArrayLike]]],
    params=DestriperParameters(),
    components: Optional[List[str]] = None,
    keep_weights: bool = False,
    keep_pixel_idx: bool = False,
    keep_pol_angle_rad: bool = False,
) -> DestriperResult:
    if not components:
        components = ["tod"]

    obs_list, ptg_list, psi_list = _normalize_observations_and_pointings(
        obs=obs, pointings=pointings
    )

    nobs_matrix_cholesky = _build_nobs_matrix(
        nside=params.nside,
        obs_list=obs_list,
        ptg_list=ptg_list,
        psi_list=psi_list,
        output_coordinate_system=params.output_coordinate_system,
    )

    if not keep_weights:
        for cur_obs in obs_list:
            del cur_obs.destriper_weights

    if not keep_pixel_idx:
        for cur_obs in obs_list:
            del cur_obs.destriper_pixel_idx

    if not keep_pol_angle_rad:
        for cur_obs in obs_list:
            del cur_obs.destriper_pol_angle_rad

    gc.collect()

    return DestriperResult(
        params=params,
        hit_map=np.zeros(1),
        binned_map=np.zeros((3, 1)),
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        coordinate_system=params.output_coordinate_system,
        # The following fields are filled only if the CG algorithm was used
        baselines=np.zeros(1),
        baseline_lengths=None,
        stopping_factors=None,
        destriped_map=np.zeros((3, 1)),
    )
