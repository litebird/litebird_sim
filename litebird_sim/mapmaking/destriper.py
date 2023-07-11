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


@dataclass
class DestriperResult:
    params: DestriperParameters
    baselines: npt.ArrayLike
    binned_map: npt.ArrayLike
    destriped_map: npt.ArrayLike
    coordinate_system: CoordinateSystem


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


def _build_nobs_matrix(
    nside: int,
    obs_list: List[Observation],
    ptg_list: List[npt.ArrayLike],
    psi_list: List[npt.ArrayLike],
    output_coordinate_system: CoordinateSystem,
) -> NobsMatrix:
    hpx = Healpix_Base(nside=nside, scheme="RING")
    # Instead of a shape like (Npix, 3, 3), i.e., one 3×3 matrix per each
    # pixel, we only store the lower triangular part in a 6-element array.
    # In this way we reduce the memory usage by ~30% and the code is faster too.
    nobs_matrix = np.zeros((hpx.npix(), 6))  # Do not use np.empty() here!

    for cur_obs, cur_ptg, cur_psi in zip(obs_list, ptg_list, psi_list):
        cur_weights = get_map_making_weights(cur_obs, check=True)

        pixidx_all, polang_all = _compute_pixel_indices(
            hpx=hpx,
            pointings=cur_ptg,
            psi=cur_psi,
            output_coordinate_system=output_coordinate_system,
        )

        _accumulate_nobs_matrix(
            pixidx_all,
            polang_all,
            cur_weights,
            nobs_matrix,
        )

        del pixidx_all, polang_all

    # Now we must accumulate the result of every MPI process
    if MPI_ENABLED:
        if MPI_COMM_WORLD.rank == 0:
            with open("/home/tomasi/test.txt", "wt") as f:
                print("Going to call allreduce…", file=f)
        nobs_matrix = MPI_COMM_WORLD.allreduce(nobs_matrix)
    else:
        with open("/home/tomasi/test.txt", "wt") as f:
            print("MPI is not enabled", file=f)

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


def make_destriped_map(
    obs: Union[Observation, List[Observation]],
    pointings: Optional[Union[npt.ArrayLike, List[npt.ArrayLike]]],
    params=DestriperParameters(),
    components: Optional[List[str]] = None,
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

    print(nobs_matrix_cholesky)

    return DestriperResult(
        params=params,
        baselines=np.zeros(1),
        binned_map=np.zeros((3, 1)),
        destriped_map=np.zeros((3, 1)),
        coordinate_system=params.output_coordinate_system,
    )
