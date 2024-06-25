# -*- encoding: utf-8 -*-
import logging
import time

# The implementation of the destriping algorithm provided here is based on the paper
# «Destriping CMB temperature and polarization maps» by Kurki-Suonio et al. 2009,
# A&A 506, 1511–1539 (2009), https://dx.doi.org/10.1051/0004-6361/200912361
#
# It is important to have that paper at hand while reading this code, as many
# functions and variable defined here use the same letters and symbols of that
# paper. We refer to it in code comments and docstrings as "KurkiSuonio2009".

from dataclasses import dataclass
import gc
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ducc0.healpix import Healpix_Base
from numba import njit, prange
import healpy as hp

from litebird_sim.mpi import MPI_ENABLED, MPI_COMM_WORLD
from typing import Callable, Union, List, Optional, Tuple, Any, Dict
from litebird_sim.hwp import HWP
from litebird_sim.observations import Observation
from litebird_sim.pointings import get_hwp_angle
from litebird_sim.coordinates import CoordinateSystem, coord_sys_to_healpix_string

from .common import (
    _compute_pixel_indices,
    _normalize_observations_and_pointings,
    COND_THRESHOLD,
    get_map_making_weights,
    cholesky,
    solve_cholesky,
    estimate_cond_number,
    _build_mask_detector_split,
    _build_mask_time_split,
)

if MPI_ENABLED:
    import mpi4py.MPI


__DESTRIPER_RESULTS_FILE_NAME = "destriper_results.fits"
__BASELINES_FILE_NAME = f"baselines_mpi{MPI_COMM_WORLD.rank:04d}.fits"


def _split_items_into_n_segments(n: int, num_of_segments: int) -> List[int]:
    """Divide a quantity `length` into chunks, each roughly of the same length

    This low-level function is used to determine how many samples in a TOD should be
    collected by the toast_destriper within the same baseline.

    .. testsetup::

        from litebird_sim.mapping import _split_items_into_n_segments

    .. testcode::

        # Divide 10 items into 4 groups, so that each of them will
        # have roughly the same number of items
        print(_split_items_into_n_segments(10, 4))

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
    """Evenly split `n` items into groups, each with roughly `sub_n` elements

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


@njit(parallel=True)
def _get_invnpp(
    nobs_matrix_cholesky: npt.NDArray,
    valid_pixel: npt.ArrayLike,
    result: npt.NDArray,
):
    npix = nobs_matrix_cholesky.shape[0]

    for ipix in range(npix):
        if not valid_pixel[ipix]:
            result[ipix, :, :] = 0.0
            continue

        # We associate each coefficient of the i-th matrix in
        # `nobs_matrix_cholesky` to the following variables:
        #
        #     | a 0 0 |
        # L = | b c 0 |
        #     | d e f |
        a, b, c, d, e, f = nobs_matrix_cholesky[ipix, :]

        # Computing the inverse of L is trivial, the result is
        #
        #       |  1/a             0      0  |
        # L⁻¹ = | −b/ac           1/c     0  |
        #       | (be − cd)/acf  −e/cf   1/f |
        #
        # We assign its coefficients to the following letters:
        #
        #       | g 0 0 |
        # L⁻¹ = | h i 0 |
        #       | j k m |
        g = 1 / a
        h = -b / (a * c)
        i = 1 / c
        j = (b * e - c * d) / (a * c * f)
        k = -e / (c * f)
        m = 1 / f

        # Now it's time for us to compute M⁻¹. We can do this quickly:
        #
        # M⁻¹ = (L L^T)⁻¹ = (L^T)⁻¹ L⁻¹ = (L⁻¹)^T L⁻¹
        #
        # where we used the fact that (AB)⁻¹ = B⁻¹ A⁻¹, and that
        # (L^T)⁻¹ = (L⁻¹)^T. The product of (L⁻¹)^T and L⁻¹ is
        #
        # | g h j |   | g 0 0 |   | g²+h²+j²  hi+jk  jm |
        # | 0 i k | · | h i 0 | = |  hi+jk    i²+k²  km |
        # | 0 0 m |   | j k m |   |   jm       km    m² |
        result[ipix, 0, 0] = g**2 + h**2 + j**2
        result[ipix, 1, 0] = h * i + j * k
        result[ipix, 0, 1] = result[ipix, 1, 0]
        result[ipix, 1, 1] = i**2 + k**2
        result[ipix, 2, 0] = j * m
        result[ipix, 0, 2] = result[ipix, 2, 0]
        result[ipix, 2, 1] = k * m
        result[ipix, 1, 2] = result[ipix, 2, 1]
        result[ipix, 2, 2] = m**2


@dataclass
class NobsMatrix:
    """
    A class containing the N_obs matrix described in Kurki-Suonio et al. (2009)

    The matrix is used to “solve” for the value of the Stokes parameters I/Q/U
    per each pixel given a set of intensity measurements for that pixel.

    The N_obs matrix is a block-diagonal matrix where each 3×3 block corresponds
    to one pixel. (See Eq. 10 in Kurki-Suonio et al.) However, since the matrix
    is symmetric, this class only stores its lower-triangular part, and the
    ``nobs_matrix`` field is thus a 2D array of shape ``(6, N_pix)``, with
    ``N_pix`` the number of pixels in the map.

    The array `valid_pixel` is an array of ``N_pix`` booleans, telling whether
    the corresponding pixel was observed enough times to make the problem of
    reconstructing the Stokes I/Q/U components solvable or not.

    If `is_cholesky` is true, the `nobs_matrix` contains the coefficients of
    the lower triangular L such that L·L^t = M, where M is the 3×3 submatrix
    in Eq. (10). This is extremely efficient, as in this way it is trivial
    to invert matrix M to solve for the three Stokes parameter. And since L
    is lower-triangular, 6 elements are still enough to keep it.
    """

    # Shape: (Npix, 6)
    nobs_matrix: npt.NDArray
    # Shape: (Npix,) (array of Boolean flags)
    valid_pixel: npt.NDArray
    # True if `nobs_matrix` contains the Cholesky decomposition of M_i
    is_cholesky: bool

    @property
    def nbytes(self) -> int:
        """Return the number of bytes used by this object"""
        return self.nobs_matrix.nbytes + self.valid_pixel.nbytes

    def get_invnpp(self) -> npt.NDArray:
        """Return the inverse noise covariance matrix per pixel

        This method returns a (N_pix, 3, 3) array containing the
        3×3 matrices associated with each pixel that contain the
        estimate noise per pixel. Each of the N_pix matrices
        corresponds to Mᵢ⁻¹, the inverse of Mᵢ in Eq. (10) of
        Kurki-Suonio et al. (2009).

        A null matrix is associated with each invalid pixel.
        """

        assert self.is_cholesky, (
            "You can use NobsMatrix.get_invnpp only after "
            "having called the binner/destriper"
        )

        npix = self.nobs_matrix.shape[0]
        result = np.empty((npix, 3, 3), dtype=np.float64)
        _get_invnpp(
            nobs_matrix_cholesky=self.nobs_matrix,
            valid_pixel=self.valid_pixel,
            result=result,
        )
        return result


@dataclass
class DestriperParameters:
    """
    Parameters used by the function :func:`.make_destriped_map`

    The class is used to tell the destriper how to solve the map-making
    equation. The fields are:

    - ``nside`` (integer, power of two): resolution of the output map
    - ``output_coordinate_system`` (:class:`.CoordinateSystem`): the
      coordinate system of the output map
    - ``samples_per_baseline`` (integer): how many consecutive samples
      in the TOD must be assigned to the same baseline. If ``None``,
      the destriper algorithm is skipped and a simple binning will
      be done.
    - ``iter_max`` (integer): maximum number of iterations for the
      Conjugate Gradient
    - ``threshold`` (float): minimum value of the discrepancy between
      the estimated baselines and the baselines deduced by the
      destriped map. The lower this value, the better the map
      produced by the destriper
    - ``use_preconditioner`` (Boolean): if ``True``, use the preconditioned
      conjugate gradient algorithm. If ``False`` do not use the
      preconditioner. (The latter is probably useful only to debug the
      code.)
    """

    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic
    samples_per_baseline: Optional[Union[int, List[npt.ArrayLike]]] = 100
    iter_max: int = 100
    threshold: float = 1e-7
    use_preconditioner: bool = True


@dataclass
class DestriperResult:
    """
    Result of a call to :func:`.make_destriped_map`

    The fields `baselines`, `baseline_lengths`, `stopping_factors`,
    `destriped_map`, and `converged` are only relevant if you actually
    used the destriper; otherwise, they will be set to ``None``.

    If you are running several MPI processes, keep in mind that the fields
    `hit_map`, `binned_map`, `destriped_map`, `nobs_matrix_cholesky`,
    and `bytes_in_temporary_buffers` contain the *global* map, but
    `baselines` and `baseline_lengths` only refer to the TODs within
    the *current* MPI process.

    List of fields:

    - ``params``: an instance of the class :class:`.DestriperParameters`
    - ``nside``: the resolution of the Healpix maps.
    - ``hit_map``: a map of scalar values, each representing the sum
      of weights per pixel (normalized over the σ of each detector).
    - ``binned_map``: the binned map (i.e., *without* subtracting the
      1/f baselines estimated by the destriper).
    - ``nobs_matrix_cholesky``: an instance of the class
      :class:`.NobsMatrix`.
    - ``coordinate_system``: the coordinate system of the maps
      ``hit_map``, ``binned_map``, and ``destriped_map``.
    - ``baselines``: a Python list of NumPy arrays (one per each
      observation passed to the destriper) containing the value
      of the baselines.
    - ``baseline_errors``: a Python list of NumPy arrays (one per
      each observation passed to the destriper) containing an
      optimistic estimate of the error per each baseline. This error
      is estimated assuming that there is no correlation between
      baselines, which is only a rough approximation.
    - ``baseline_lengths``: a Python list of NumPy arrays (one per
      each observation passed to the destriper) containing the
      number of TOD samples per each baseline.
    - ``stopping_factor``: the maximum residual for the destriping
      solution stored in the field `baselines`. It is an assessment
      of the quality of the solution found by the destriper: the
      lower, the better.
    - ``history_of_stopping_factors``: list of stopping factors
      computed by the iterative algorithm. This list should ideally
      be monothonically decreasing, as this means that the destriping
      algorithm was able to converge to the correct solution.
    - ``converged``: a Boolean flag telling whether the destriper
      was able to achieve the desired accuracy (the value of
      ``params.threshold``).
    - ``elapsed_time_s``: the elapsed time spent by the function
      :func:`.make_destriped_map`.
    - ``bytes_in_temporary_buffers``: the number of bytes allocated
      internally by the destriper for temporary buffers.
    """

    params: DestriperParameters
    nside: int
    components: List[str]
    hit_map: npt.ArrayLike
    binned_map: npt.ArrayLike
    nobs_matrix_cholesky: NobsMatrix
    coordinate_system: CoordinateSystem
    # The following fields are filled only if the CG algorithm was used
    baselines: Optional[npt.ArrayLike]
    baseline_errors: Optional[npt.ArrayLike]
    baseline_lengths: Optional[npt.ArrayLike]
    stopping_factor: Optional[float]
    history_of_stopping_factors: Optional[List[float]]
    destriped_map: Optional[npt.ArrayLike]
    converged: Union[bool, str]
    elapsed_time_s: float
    bytes_in_temporary_buffers: int
    detector_split: Optional[str] = "full"
    time_split: Optional[str] = "full"


def _sum_components_into_obs(
    obs_list: List[Observation],
    target: str,
    other_components: List[str],
    factor: float,
) -> None:
    """Sum all the TOD components into the first one

    If `target` is “tod” and `other_components` is the list ``["sky", "noise"]``,
    this function will do the operation ``tod += factor * (sky + noise)``.
    The value of `factor` is usually ±1, which means that you can either sum
    or subtract the sky and the noise in the above example.
    """
    for cur_obs in obs_list:
        target_tod = getattr(cur_obs, target)

        for other_component in other_components:
            target_tod += factor * getattr(cur_obs, other_component)


@njit
def _accumulate_nobs_matrix(
    pix_idx: npt.ArrayLike,  # Shape: (Ndet, 1)
    psi_angle_rad: npt.ArrayLike,  # Shape: (Ndet, 1)
    weights: npt.ArrayLike,  # Shape: (N_det,)
    d_mask: npt.ArrayLike,
    t_mask: npt.ArrayLike,
    nobs_matrix: npt.ArrayLike,  # Shape: (N_pix, 6)
) -> None:
    """
    Fill the upper triangle of the N_obs matrix following Eq. (10)
    of KurkiSuonio2009. This must be set just once during destriping,
    as it only depends on the pointing information

    Note that `nobs_matrix` must have been set to 0 before starting
    calling this function!
    """

    assert pix_idx.shape == psi_angle_rad.shape

    assert pix_idx.shape[0] == d_mask.shape[0]

    num_of_detectors = pix_idx.shape[0]

    for det_idx in range(num_of_detectors):
        if not d_mask[det_idx]:
            continue

        inv_sigma = 1.0 / np.sqrt(weights[det_idx])
        inv_sigma2 = inv_sigma * inv_sigma

        # Fill the lower triangle of M_i only for i = 1…N_pix
        for cur_pix_idx, cur_psi, cur_t_mask in zip(
            pix_idx[det_idx], psi_angle_rad[det_idx], t_mask
        ):
            if cur_t_mask:
                cos_over_sigma = np.cos(2 * cur_psi) * inv_sigma
                sin_over_sigma = np.sin(2 * cur_psi) * inv_sigma
                cur_matrix = nobs_matrix[cur_pix_idx]

                cur_matrix[0] += inv_sigma2
                cur_matrix[1] += cos_over_sigma * inv_sigma
                cur_matrix[2] += cos_over_sigma * cos_over_sigma
                cur_matrix[3] += sin_over_sigma * inv_sigma
                cur_matrix[4] += sin_over_sigma * cos_over_sigma
                cur_matrix[5] += sin_over_sigma * sin_over_sigma


@njit(parallel=True)
def _nobs_matrix_to_cholesky(
    nobs_matrix: npt.ArrayLike,  # Shape: (N_pix, 6)
    dest_valid_pixel: npt.ArrayLike,  # Shape: (N_pix,)
    dest_nobs_matrix_cholesky: npt.ArrayLike,  # Shape: (N_pix, 6)
) -> None:
    """Apply `cholesky` iteratively on all the input maps in `nobs_matrix`
    and save each result in `nobs_matrix_cholesky`"""

    for pixel_idx in prange(nobs_matrix.shape[0]):
        cur_nobs_matrix = nobs_matrix[pixel_idx]
        (cond_number, flag) = estimate_cond_number(
            a00=cur_nobs_matrix[0],
            a10=cur_nobs_matrix[1],
            a11=cur_nobs_matrix[2],
            a20=cur_nobs_matrix[3],
            a21=cur_nobs_matrix[4],
            a22=cur_nobs_matrix[5],
        )
        dest_valid_pixel[pixel_idx] = flag and (cond_number < COND_THRESHOLD)
        if dest_valid_pixel[pixel_idx]:
            cholesky(
                a00=cur_nobs_matrix[0],
                a10=cur_nobs_matrix[1],
                a11=cur_nobs_matrix[2],
                a20=cur_nobs_matrix[3],
                a21=cur_nobs_matrix[4],
                a22=cur_nobs_matrix[5],
                dest_L=dest_nobs_matrix_cholesky[pixel_idx],
            )
        # There is no `else`: we assume that
        # `nobs_matrix_cholesky` was already initialized
        # to zero everywhere


def _store_pixel_idx_and_pol_angle_in_obs(
    hpx: Healpix_Base,
    obs_list: List[Observation],
    ptg_list: Union[List[npt.ArrayLike], List[Callable]],
    hwp: Union[HWP, None],
    output_coordinate_system: CoordinateSystem,
):
    for cur_obs, cur_ptg in zip(obs_list, ptg_list):
        cur_obs.destriper_weights = get_map_making_weights(cur_obs, check=True)

        if hwp is None:
            if hasattr(cur_obs, "hwp_angle"):
                hwp_angle = cur_obs.hwp_angle
            else:
                hwp_angle = None
        else:
            if type(cur_ptg) is np.ndarray:
                hwp_angle = get_hwp_angle(cur_obs, hwp)
            else:
                logging.warning(
                    "For using an external HWP object also pass a pre-calculated pointing"
                )

        (
            cur_obs.destriper_pixel_idx,
            cur_obs.destriper_pol_angle_rad,
        ) = _compute_pixel_indices(
            hpx=hpx,
            pointings=cur_ptg,
            num_of_detectors=cur_obs.n_detectors,
            num_of_samples=cur_obs.n_samples,
            hwp_angle=hwp_angle,
            output_coordinate_system=output_coordinate_system,
        )


def _build_nobs_matrix(
    hpx: Healpix_Base,
    obs_list: List[Observation],
    dm_list: List[npt.ArrayLike],
    tm_list: List[npt.ArrayLike],
) -> NobsMatrix:
    # Instead of a shape like (Npix, 3, 3), i.e., one 3×3 matrix per each
    # pixel, we only store the lower triangular part in a 6-element array.
    # In this way we reduce the memory usage by ~30% and the code is faster too.
    nobs_matrix = np.zeros((hpx.npix(), 6))  # Do not use np.empty() here!

    for cur_obs, cur_d_mask, cur_t_mask in zip(obs_list, dm_list, tm_list):
        _accumulate_nobs_matrix(
            pix_idx=cur_obs.destriper_pixel_idx,
            psi_angle_rad=cur_obs.destriper_pol_angle_rad,
            weights=cur_obs.destriper_weights,
            nobs_matrix=nobs_matrix,
            d_mask=cur_d_mask,
            t_mask=cur_t_mask,
        )

    # Now we must accumulate the result of every MPI process
    if MPI_ENABLED:
        MPI_COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE, nobs_matrix, op=mpi4py.MPI.SUM)

    # `nobs_matrix_cholesky` will *not* contain the M_i maps shown in
    # Eq. 9 of KurkiSuonio2009, but its Cholesky decomposition, i.e.,
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


@njit
def _step_over_baseline(baseline_idx, samples_in_this_baseline, baseline_length):
    samples_in_this_baseline += 1
    if samples_in_this_baseline >= baseline_length[baseline_idx]:
        baseline_idx += 1
        samples_in_this_baseline = 0

    return (baseline_idx, samples_in_this_baseline)


@njit
def _sum_map_contribution_from_one_sample(
    pol_angle_rad: float, sample: float, weight: float, dest_array: npt.ArrayLike
) -> None:
    "This code implements Eqq. (18)–(20)"

    dest_array[0] += sample / weight
    dest_array[1] += sample * np.cos(2 * pol_angle_rad) / weight
    dest_array[2] += sample * np.sin(2 * pol_angle_rad) / weight


@njit
def _update_sum_map_with_tod(
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    tod: npt.ArrayLike,
    pol_angle_rad: npt.ArrayLike,
    pixel_idx: npt.ArrayLike,
    weights: npt.ArrayLike,
    d_mask: npt.ArrayLike,
    t_mask: npt.ArrayLike,
    baseline_lengths: npt.ArrayLike,  # Number of samples per baseline
) -> None:
    """
    Compute the sum map within the current MPI process for TOD y

    This function implements the calculation of the operator P^t C_w⁻¹
    (Eqq. 18–20), the so-called “sum map”. Note that the summation
    is done on the samples of the current MPI process; the overall
    summation is done by `_compute_binned_map`, which solves the
    three I/Q/U Stokes parameters.

    """

    for det_idx in range(pixel_idx.shape[0]):
        if not d_mask[det_idx]:
            continue
        cur_weight = weights[det_idx]

        baseline_idx = 0
        samples_in_this_baseline = 0

        for sample_idx in range(tod.shape[1]):
            if not t_mask[sample_idx]:
                (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                    baseline_idx, samples_in_this_baseline, baseline_lengths
                )
                continue

            cur_pix = pixel_idx[det_idx, sample_idx]
            _sum_map_contribution_from_one_sample(
                pol_angle_rad=pol_angle_rad[det_idx, sample_idx],
                sample=tod[det_idx, sample_idx],
                dest_array=sky_map[:, cur_pix],
                weight=cur_weight,
            )
            hit_map[cur_pix] += 1.0 / cur_weight

            (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                baseline_idx, samples_in_this_baseline, baseline_lengths
            )


@njit
def _update_sum_map_with_baseline(
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    pol_angle_rad: npt.ArrayLike,
    pixel_idx: npt.ArrayLike,
    weights: npt.ArrayLike,
    d_mask: npt.ArrayLike,
    t_mask: npt.ArrayLike,
    baselines: npt.ArrayLike,  # Value of each baseline
    baseline_lengths: npt.ArrayLike,  # Number of samples per baseline
) -> None:
    """
    Compute the sum map within the current MPI process for baselines Fa

    This function is the same as `_update_sum_map_with_tod`, but it
    sums the baselines unrolled over the TOD instead of the TOD itself.

    (Note that we could have avoided code duplication between these
    two functions, had we used some more advanced language like Julia! ☹)
    """

    for det_idx in range(pixel_idx.shape[0]):
        if not d_mask[det_idx]:
            continue
        cur_weight = weights[det_idx]

        baseline_idx = 0
        samples_in_this_baseline = 0

        for sample_idx in range(pixel_idx.shape[1]):
            if not t_mask[sample_idx]:
                (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                    baseline_idx, samples_in_this_baseline, baseline_lengths
                )
                continue

            cur_pix = pixel_idx[det_idx, sample_idx]
            _sum_map_contribution_from_one_sample(
                pol_angle_rad=pol_angle_rad[det_idx, sample_idx],
                sample=baselines[det_idx, baseline_idx],
                dest_array=sky_map[:, cur_pix],
                weight=cur_weight,
            )
            hit_map[cur_pix] += 1.0 / cur_weight

            (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                baseline_idx, samples_in_this_baseline, baseline_lengths
            )


@njit
def _sum_map_to_binned_map(
    sky_map: npt.ArrayLike,
    nobs_matrix_cholesky: npt.ArrayLike,
    valid_pixels: npt.ArrayLike,
) -> None:
    """Convert a “sum map” into a “binned map” using the N_obs matrix"""

    for cur_pix in range(sky_map.shape[1]):
        if valid_pixels[cur_pix]:
            cur_i, cur_q, cur_u = solve_cholesky(
                L=nobs_matrix_cholesky[cur_pix, :],
                v0=sky_map[0, cur_pix],
                v1=sky_map[1, cur_pix],
                v2=sky_map[2, cur_pix],
            )

            sky_map[0, cur_pix] = cur_i
            sky_map[1, cur_pix] = cur_q
            sky_map[2, cur_pix] = cur_u
        else:
            sky_map[0, cur_pix] = np.nan
            sky_map[1, cur_pix] = np.nan
            sky_map[2, cur_pix] = np.nan


def _compute_binned_map(
    obs_list: List[Observation],
    nobs_matrix_cholesky: NobsMatrix,
    baselines_list: Optional[List[npt.ArrayLike]],
    baseline_lengths_list: List[npt.ArrayLike],
    dm_list: List[npt.ArrayLike],
    tm_list: List[npt.ArrayLike],
    component: Optional[str],
    output_hit_map: npt.ArrayLike,
    output_sky_map: npt.ArrayLike,
) -> None:
    """
    Compute the global binned map

    This function computes the B≡M⁻¹·P^t·C_w⁻¹ operator (Eq. 21),
    which “bins” the TODs contained in `obs_list` in a sum map
    (using `_update_sum_map_local`), reduces the sums from all the
    MPI processes, and then solves for the I/Q/U parameters.

    This is applied either to `y` or to `Fa`, depending on whether
    the parameter `baselines_list` is ``None`` or not, respectively.
    (You *must* pass one of them, but not both.)

    The result is saved in `sky_map` (a 3,N_p tensor) and `hit_map`
    (a N_p vector).
    """

    assert nobs_matrix_cholesky.is_cholesky, (
        "The parameter nobs_matrix_cholesky should already "
        "contain the Cholesky decompositions of the 3×3 M_i matrices"
    )

    assert ((baselines_list is not None) and (component is None)) or (
        (baselines_list is None) and (component is not None)
    ), (
        "To call _compute_binned_map you must either provide "
        "the baselines or the TOD component, but not both"
    )

    # Step 1: compute the “sum map” (Eqq. 18–20)
    output_sky_map[:] = 0
    output_hit_map[:] = 0

    for obs_idx, (cur_obs, cur_baseline_lengths, cur_d_mask, cur_t_mask) in enumerate(
        zip(obs_list, baseline_lengths_list, dm_list, tm_list)
    ):
        if baselines_list is None:
            _update_sum_map_with_tod(
                sky_map=output_sky_map,
                hit_map=output_hit_map,
                tod=getattr(cur_obs, component),
                pol_angle_rad=cur_obs.destriper_pol_angle_rad,
                pixel_idx=cur_obs.destriper_pixel_idx,
                weights=cur_obs.destriper_weights,
                d_mask=cur_d_mask,
                t_mask=cur_t_mask,
                baseline_lengths=cur_baseline_lengths,
            )
        else:
            cur_baselines = baselines_list[obs_idx]
            _update_sum_map_with_baseline(
                sky_map=output_sky_map,
                hit_map=output_hit_map,
                pol_angle_rad=cur_obs.destriper_pol_angle_rad,
                pixel_idx=cur_obs.destriper_pixel_idx,
                weights=cur_obs.destriper_weights,
                d_mask=cur_d_mask,
                t_mask=cur_t_mask,
                baselines=cur_baselines,
                baseline_lengths=cur_baseline_lengths,
            )

    if MPI_ENABLED:
        MPI_COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE, output_sky_map, op=mpi4py.MPI.SUM)
        MPI_COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE, output_hit_map, op=mpi4py.MPI.SUM)

    # Step 2: compute the “binned map” (Eq. 21)
    _sum_map_to_binned_map(
        sky_map=output_sky_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky.nobs_matrix,
        valid_pixels=nobs_matrix_cholesky.valid_pixel,
    )


@njit
def estimate_sample_from_map(
    cur_pixel: int, cur_psi: float, sky_map: npt.ArrayLike
) -> float:
    cur_i = sky_map[0, cur_pixel]
    cur_q = sky_map[1, cur_pixel]
    cur_u = sky_map[2, cur_pixel]

    return cur_i + cur_q * np.cos(2 * cur_psi) + cur_u * np.sin(2 * cur_psi)


@njit
def _compute_tod_sums_for_one_component(
    weights: npt.ArrayLike,
    tod: npt.ArrayLike,
    pixel_idx: npt.ArrayLike,
    psi_angle_rad: npt.ArrayLike,
    sky_map: npt.ArrayLike,
    d_mask: npt.ArrayLike,
    t_mask: npt.ArrayLike,
    baseline_length: npt.ArrayLike,
    output_sums: npt.ArrayLike,
) -> None:
    """
    Compute F^t C_w⁻¹ Z over TOD y

    :param weights: The detector weights (array of N_det elements)
    :param tod: The vector `y` (a NumPy array with N_samp elements)
    :param pixel_idx: A NumPy array of N_samp Healpix indexes
    :param psi_angle_rad: Values of the polarization angles (N_samp elements)
    :param sky_map: The sky map used to compute operator Z
    :param baseline_length: Array of N_base integers (the number
        of samples per baseline)
    :param output_sums: An array of N_base elements that will contain the result
    """
    output_sums[:] = 0

    for det_idx, cur_weight in enumerate(weights):
        if not d_mask[det_idx]:
            continue
        det_pixel_idx = pixel_idx[det_idx, :]
        det_psi_angle_rad = psi_angle_rad[det_idx, :]

        baseline_idx = 0
        samples_in_this_baseline = 0

        for sample_idx in range(len(det_pixel_idx)):
            if not t_mask[sample_idx]:
                (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                    baseline_idx, samples_in_this_baseline, baseline_length
                )
                continue

            map_value = estimate_sample_from_map(
                cur_pixel=det_pixel_idx[sample_idx],
                cur_psi=det_psi_angle_rad[sample_idx],
                sky_map=sky_map,
            )
            value_to_add = (tod[det_idx, sample_idx] - map_value) / cur_weight
            if np.isfinite(value_to_add):
                output_sums[det_idx, baseline_idx] += value_to_add

            (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                baseline_idx, samples_in_this_baseline, baseline_length
            )


@njit
def _compute_baseline_sums_for_one_component(
    weights: npt.ArrayLike,
    pixel_idx: npt.ArrayLike,
    psi_angle_rad: npt.ArrayLike,
    sky_map: npt.ArrayLike,
    d_mask: npt.ArrayLike,
    t_mask: npt.ArrayLike,
    baselines: npt.ArrayLike,
    baseline_length: npt.ArrayLike,
    output_sums: npt.ArrayLike,
) -> None:
    """
    Compute F^t C_w⁻¹ Z over TOD Fa (baselines projected in TOD space)

    :param weights: The detector weights (array of N_det elements)
    :param pixel_idx: A NumPy array of N_samp Healpix indexes
    :param psi_angle_rad: Values of the polarization angles (N_samp elements)
    :param sky_map: The sky map used to compute operator Z
    :param baselines: Array of N_base numbers (the value of each baseline)
    :param baseline_length: Array of N_base integers (the number
        of samples per baseline)
    :param output_sums: An array of N_base elements that will contain the result
    """
    output_sums[:] = 0

    for det_idx, cur_weight in enumerate(weights):
        if not d_mask[det_idx]:
            continue
        det_pixel_idx = pixel_idx[det_idx, :]
        det_psi_angle_rad = psi_angle_rad[det_idx, :]

        baseline_idx = 0
        samples_in_this_baseline = 0

        for sample_idx in range(len(det_pixel_idx)):
            if not t_mask[sample_idx]:
                (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                    baseline_idx, samples_in_this_baseline, baseline_length
                )
                continue

            map_value = estimate_sample_from_map(
                cur_pixel=det_pixel_idx[sample_idx],
                cur_psi=det_psi_angle_rad[sample_idx],
                sky_map=sky_map,
            )
            cur_value = (baselines[det_idx, baseline_idx] - map_value) / cur_weight
            if np.isfinite(cur_value):
                output_sums[det_idx, baseline_idx] += cur_value

            (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                baseline_idx, samples_in_this_baseline, baseline_length
            )


def _compute_baseline_sums(
    obs_list: List[Observation],
    sky_map: npt.ArrayLike,
    baselines_list: Optional[List[npt.ArrayLike]],
    baseline_lengths_list: List[npt.ArrayLike],
    component: Optional[str],
    dm_list: List[npt.ArrayLike],
    tm_list: List[npt.ArrayLike],
    output_sums_list: List[npt.ArrayLike],
):
    """
    Compute F^t C_w⁻¹ Z either on the TOD Fa (baselines) or y (TOD)

    The matrix F “spreads” the baseline values “a” over the TOD space,
    while “y” is the TOD owned by each :class:`.Observation` object
    contained in `obs_list`. If `baselines_list` is not None,
    the operator is applied on Fa, otherwise on y, where the name of
    the field of the :class:`.Observation` class holding the TOD
    is specified by `component`.

    The field `baselines_list` (if specified) and `baseline_lengths_list`
    are lists of NumPy arrays; they are lists with the same length as
    `obs_list` and must contain the input value of the baselines and
    their lengths in terms of number of TOD samples, respectively.
    When using MPI, the baselines must refer to the TOD samples handled
    by the TOD owned by the current MPI process.

    The result is saved in `output_sums_list`, which must have already
    been allocated. Note that you *cannot* make this field point to
    the same memory location as `baselines_list`, although the two
    objects share the same shape.
    """

    assert len(baseline_lengths_list) == len(obs_list), (
        f"The baselines have been specified for {len(baseline_lengths_list)} "
        f"observations, but there are {len(obs_list)} observation(s) available"
    )
    assert len(output_sums_list) == len(obs_list), (
        f"There are {len(output_sums_list)} buffers for the output but "
        f"{len(obs_list)} observation(s)"
    )

    # Compute the value of the F^t C_w⁻¹ Z operator
    for obs_idx, (
        cur_obs,
        cur_baseline_lengths,
        cur_sums,
        cur_d_mask,
        cur_t_mask,
    ) in enumerate(
        zip(obs_list, baseline_lengths_list, output_sums_list, dm_list, tm_list)
    ):
        assert len(cur_baseline_lengths) == cur_sums.shape[1], (
            f"The output buffer for observation {obs_idx=} "
            f"has room for {cur_sums.shape[1]} elements, but there"
            f"are {len(cur_baseline_lengths)=} baselines in this observation"
        )

        if baselines_list is not None:
            cur_baseline = baselines_list[obs_idx]
            assert cur_baseline is not cur_sums, (
                "The input and output arrays used to hold the baselines "
                "must be different"
            )

            _compute_baseline_sums_for_one_component(
                weights=cur_obs.destriper_weights,
                pixel_idx=cur_obs.destriper_pixel_idx,
                psi_angle_rad=cur_obs.destriper_pol_angle_rad,
                sky_map=sky_map,
                d_mask=cur_d_mask,
                t_mask=cur_t_mask,
                baselines=cur_baseline,
                baseline_length=cur_baseline_lengths,
                output_sums=cur_sums,
            )
        else:
            _compute_tod_sums_for_one_component(
                weights=cur_obs.destriper_weights,
                tod=getattr(cur_obs, component),
                pixel_idx=cur_obs.destriper_pixel_idx,
                psi_angle_rad=cur_obs.destriper_pol_angle_rad,
                sky_map=sky_map,
                d_mask=cur_d_mask,
                t_mask=cur_t_mask,
                baseline_length=cur_baseline_lengths,
                output_sums=cur_sums,
            )


def _mpi_dot(a: List[npt.ArrayLike], b: List[npt.ArrayLike]) -> float:
    """Compute a dot product between lists of vectors using MPI

    This function is a glorified version of ``numpy.dot``. It assumes
    that `a` and `b` are lists of vectors spread among several
    observations, and it computes their inner product Σ aᵢ·bᵢ.
    If the code uses MPI, it makes the additional assumptions that
    the vectors aᵢ and bᵢ have been split among the MPI processes,
    so it computes the local dot product and then sums the contribution
    from every MPI process.
    """

    # As both x1 and x2 are 2D arrays with shape (N_detectors, N_baselines),
    # we call “flatten” to make them 1D and produce *one* scalar out of
    # the dot product
    local_result = sum([np.dot(x1.flatten(), x2.flatten()) for (x1, x2) in zip(a, b)])
    if MPI_ENABLED:
        return MPI_COMM_WORLD.allreduce(local_result, op=mpi4py.MPI.SUM)
    else:
        return local_result


def _get_stopping_factor(residual: List[npt.ArrayLike]) -> float:
    """Given a list of baseline residuals, estimate the stopping factor

    Our assumption here is that the stopping factor is the maximum absolute
    value of all the residuals. This is unlike other implementations of the
    CG algorithm, which just consider the value of ‖v‖ or ‖v‖²; we choose
    to use max(vᵢ) because this is stricter: it prevents to get low
    stopping factors for solutions where nearly all the baselines are ok
    but a few of them are significantly off.
    """
    local_result = np.max(np.abs(residual))
    if MPI_ENABLED:
        return MPI_COMM_WORLD.allreduce(local_result, op=mpi4py.MPI.MAX)
    else:
        return local_result


def _compute_b_or_Ax(
    obs_list: List[Observation],
    nobs_matrix_cholesky: NobsMatrix,
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    dm_list: List[npt.ArrayLike],
    tm_list: List[npt.ArrayLike],
    baselines_list: Optional[List[npt.ArrayLike]],
    baseline_lengths_list: List[npt.ArrayLike],
    component: Optional[str],
    result: List[npt.ArrayLike],
):
    """Either compute `Ax` or `b` in the map-making equation `Ax=b`

    The two terms `Ax` and `b` are similar, as `Ax` applies the
    `F^t·C_w⁻¹·Z·F` operator to the baselines `a`, while `b` is
    the vector `F^t·C_w⁻¹·Z·y`.
    """

    _compute_binned_map(
        obs_list=obs_list,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        dm_list=dm_list,
        tm_list=tm_list,
        component=component,
        output_sky_map=sky_map,
        output_hit_map=hit_map,
    )

    _compute_baseline_sums(
        obs_list=obs_list,
        sky_map=sky_map,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component=component,
        dm_list=dm_list,
        tm_list=tm_list,
        output_sums_list=result,
    )


def compute_b(
    obs_list: List[Observation],
    nobs_matrix_cholesky: NobsMatrix,
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    baseline_lengths_list: List[npt.ArrayLike],
    component: str,
    dm_list: List[npt.ArrayLike],
    tm_list: List[npt.ArrayLike],
    result: List[npt.ArrayLike],
) -> None:
    """
    Compute `F^t·C_w⁻¹·Z·y

    The value of `y` is the TOD component taken from the list
    of observations `obs_list` with name `component`.
    """

    _compute_b_or_Ax(
        obs_list=obs_list,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        sky_map=sky_map,
        hit_map=hit_map,
        dm_list=dm_list,
        tm_list=tm_list,
        baselines_list=None,
        baseline_lengths_list=baseline_lengths_list,
        component=component,
        result=result,
    )


def compute_Ax(
    obs_list: List[Observation],
    nobs_matrix_cholesky: NobsMatrix,
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    dm_list: List[npt.ArrayLike],
    tm_list: List[npt.ArrayLike],
    baselines_list: List[npt.ArrayLike],
    baseline_lengths_list: List[npt.ArrayLike],
    result: List[npt.ArrayLike],
) -> None:
    """
    Compute `F^t·C_w⁻¹·Z·F·a

    The value of `a` is the list of baselines in the
    parameter `baselines_list`, each with length
    `baseline_lengths_list`.
    """

    _compute_b_or_Ax(
        obs_list=obs_list,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        sky_map=sky_map,
        hit_map=hit_map,
        dm_list=dm_list,
        tm_list=tm_list,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component=None,
        result=result,
    )


@njit
def compute_weighted_baseline_length(
    lengths: npt.ArrayLike, weights: npt.ArrayLike, result: npt.ArrayLike
) -> None:
    """Compute Σ Nᵢ/σᵢ, where the summation is done over the detectors

    This quantity is used both to estimate the error bar for each baseline
    and to implement a simple preconditioner for the Conjugate Gradient
    algorithm.
    """

    for baseline_idx in range(len(result)):
        result[baseline_idx] = 0.0
        for cur_weight in weights:
            result[baseline_idx] += lengths[baseline_idx] / cur_weight

        # We store the *inverse*, i.e., the diagonal of matrix M⁻¹
        result[baseline_idx] = 1.0 / result[baseline_idx]


def _create_preconditioner(obs_list, baseline_lengths_list) -> List[npt.ArrayLike]:
    """
    We just compute (F^T·C_w⁻¹·F)⁻¹, which is a diagonal matrix containing
    the number of elements in each baseline divided by σ². (Remember
    that the field `destriper_weights` already contains σ².)

    This is the most common choice, but it's not necessarily the best one.
    See for instance these papers:

    1. A fast map-making preconditioner for regular scanning patterns
       (Naess & al., 2014)

    2. Accelerating the cosmic microwave background map-making procedure
       through preconditioning (Szydlarski, 2014)

    Our choice corresponds to Eq. (10) in Szydlarski's paper.
    """

    assert len(obs_list) == len(baseline_lengths_list), (
        f"The number of observations is {len(obs_list)}, but the baseline "
        f"lengths are enough for {len(baseline_lengths_list)} elements"
    )

    result = [np.empty(len(len_k)) for len_k in baseline_lengths_list]
    for obs_k, lengths_k, result_k in zip(obs_list, baseline_lengths_list, result):
        compute_weighted_baseline_length(
            weights=obs_k.destriper_weights, lengths=lengths_k, result=result_k
        )

    return result


def _apply_preconditioner(precond: List[npt.ArrayLike], z: List[npt.ArrayLike]) -> None:
    for precond_k, z_k in zip(precond, z):
        z_k *= precond_k


def _compute_num_of_bytes_in_list(x: List[npt.ArrayLike]) -> int:
    return sum([elem.nbytes for elem in x])


def _run_destriper(
    obs_list: List[Observation],
    nobs_matrix_cholesky: NobsMatrix,
    binned_map: npt.ArrayLike,
    destriped_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    baseline_lengths_list: List[npt.ArrayLike],
    baselines_list_start: List[npt.ArrayLike],
    dm_list: List[npt.ArrayLike],
    tm_list: List[npt.ArrayLike],
    component: str,
    threshold: float,
    max_steps: int,
    use_preconditioner: bool,
    callback: Any,
    callback_kwargs: Dict[Any, Any],
    recycled_convergence: bool,
    recycle_baselines: Optional[bool] = False,
) -> Tuple[
    List[npt.ArrayLike],  # The solution, i.e., the list of baselines
    List[npt.ArrayLike],  # The error bars of the baselines
    List[float],  # The list of stopping factors
    float,  # The best stopping factor found during the iterations
    bool,  # Has the destriper converged to a solution?
    int,  # Number of bytes used by temporary buffers
]:
    """Apply the Conjugate Gradient (CG) algorithm to find the solution for Ax=b"""

    # To understand how the PCG algorithm works, you should read the paper
    # “Painless conjugate gradient” by Shewchuk (1994), which tells you everything
    # you need to grasp the geometrical meaning of these operations.
    #
    # Be aware that in our case the matrix A is singular (there are infinite
    # solutions, as the baselines can be shifted by an arbitrary additive offset).
    # But this is not a problem for the CG algorithm, as it is able to work with
    # positive-semidefinite matrices as well.
    #
    # We use short variable names because we wanted to match the description
    # of the algorithm provided by Wikipedia [1]. Keep in mind these points:
    #
    # - Instead of using subscripts to denote the k-th and the (k+1)-th element,
    #   we prepend “new_” to the (k+1)-th element. Thus, r_k is called `r` and
    #   r_{k+1} is called `new_r`.
    #
    # - The `x` term is the array of baselines (`a` in the paper by Kurki-Suonio)
    #
    # [1] https://en.wikipedia.org/wiki/Conjugate_gradient_method

    bytes_in_temporary_buffers = 0

    assert nobs_matrix_cholesky.is_cholesky, (
        "_run_destriper requires that `nobs_matrix_cholesky` "
        "already contains the Cholesky transforms"
    )

    assert len(obs_list) == len(baselines_list_start), (
        f"There are {len(obs_list)} observations, but `baselines_list_start` "
        f"only contains {len(baselines_list_start)} elements"
    )

    # We allocate all the memory in advance: this ensures the code is fast
    # and prevents memory fragmentation

    # Preallocate memory for the baselines at the k-th step
    x = [np.copy(cur_baseline) for cur_baseline in baselines_list_start]
    bytes_in_temporary_buffers += _compute_num_of_bytes_in_list(x)

    # This is the “best” baseline found during the iterations. Normally,
    # the best one is the last one, unless the loop was cut short because
    # the maximum number of iterations was reached.
    best_x = [np.copy(cur_baseline) for cur_baseline in baselines_list_start]
    bytes_in_temporary_buffers += _compute_num_of_bytes_in_list(best_x)
    best_stopping_factor = None

    # The `b` value used by the Wikipedia article corresponds to
    # Kurki-Suonio's F^t·C_w⁻¹·Z·F·a
    b = [
        np.empty((getattr(cur_obs, component).shape[0], len(cur_baseline_lengths)))
        for (cur_obs, cur_baseline_lengths) in zip(obs_list, baseline_lengths_list)
    ]
    bytes_in_temporary_buffers += _compute_num_of_bytes_in_list(b)

    # A·x corresponds to F^t·C_w⁻¹·Z·y
    Ax = [np.empty_like(b_k) for b_k in b]
    bytes_in_temporary_buffers += _compute_num_of_bytes_in_list(Ax)

    # This is the residual b−A·x; ideally, if we already had the correct solution,
    # it should be zero.
    r = [np.empty_like(b_k) for b_k in b]
    bytes_in_temporary_buffers += _compute_num_of_bytes_in_list(r)

    # Initialize r_k
    compute_b(
        obs_list=obs_list,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        sky_map=destriped_map,
        hit_map=hit_map,
        dm_list=dm_list,
        tm_list=tm_list,
        baseline_lengths_list=baseline_lengths_list,
        component=component,
        result=b,
    )
    compute_Ax(
        obs_list=obs_list,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        sky_map=destriped_map,
        hit_map=hit_map,
        dm_list=dm_list,
        tm_list=tm_list,
        baselines_list=x,
        baseline_lengths_list=baseline_lengths_list,
        result=Ax,
    )
    for r_k, b_k, A_k in zip(r, b, Ax):
        r_k[:] = b_k - A_k

    new_r = [np.copy(r_k) for r_k in r]
    bytes_in_temporary_buffers += _compute_num_of_bytes_in_list(new_r)

    z = [np.copy(r_k) for r_k in r]
    bytes_in_temporary_buffers += _compute_num_of_bytes_in_list(z)

    precond = _create_preconditioner(
        obs_list=obs_list, baseline_lengths_list=baseline_lengths_list
    )
    bytes_in_temporary_buffers += _compute_num_of_bytes_in_list(precond)

    if use_preconditioner:
        _apply_preconditioner(precond, z)
    k = 0

    old_r_dot = _mpi_dot(z, r)

    history_of_stopping_factors = [_get_stopping_factor(r)]  # type: List[float]
    if callback:
        callback(
            stopping_factor=history_of_stopping_factors[-1],
            step_number=k,
            max_steps=max_steps,
            **callback_kwargs,
        )

    while True:
        k += 1
        if k >= max_steps:
            converged = False
            break
        if recycle_baselines:
            converged = f"Recycled baselines with 'converged = {recycled_convergence}' were used!"
            break

        compute_Ax(
            obs_list=obs_list,
            nobs_matrix_cholesky=nobs_matrix_cholesky,
            sky_map=destriped_map,
            hit_map=hit_map,
            dm_list=dm_list,
            tm_list=tm_list,
            baselines_list=z,
            baseline_lengths_list=baseline_lengths_list,
            result=Ax,
        )
        α = old_r_dot / _mpi_dot(z, Ax)

        for x_k, z_k in zip(x, z):
            x_k += α * z_k

        for new_r_k, r_k, Ax_k in zip(new_r, r, Ax):
            new_r_k[:] = r_k - α * Ax_k

        cur_stopping_factor = _get_stopping_factor(new_r)
        if (not best_stopping_factor) or cur_stopping_factor < best_stopping_factor:
            best_stopping_factor = cur_stopping_factor
            for cur_best_x_k, x_k in zip(best_x, x):
                cur_best_x_k[:] = x_k

        history_of_stopping_factors.append(cur_stopping_factor)
        if callback:
            callback(
                stopping_factor=history_of_stopping_factors[-1],
                step_number=k,
                max_steps=max_steps,
                **callback_kwargs,
            )

        if cur_stopping_factor < threshold:
            converged = True
            break

        for z_k, r_k in zip(z, r):
            z_k[:] = r_k[:]
        if use_preconditioner:
            _apply_preconditioner(precond, z)

        new_r_dot = _mpi_dot(new_r, z)
        for z_k, r_k, new_r_k in zip(z, r, new_r):
            z_k[:] = new_r_k + (new_r_dot / old_r_dot) * z_k

        old_r_dot = new_r_dot
        r = new_r

    # Redo the binned and destriped map with the best solution found so far

    # First, compute the binned map by passing `baselines_list=None`…
    _compute_binned_map(
        obs_list=obs_list,
        output_sky_map=binned_map,
        output_hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        component=component,
        dm_list=dm_list,
        tm_list=tm_list,
        baselines_list=None,
        baseline_lengths_list=baseline_lengths_list,
    )

    # …then compute the map from the “unrolled” baselines F·a…
    _compute_binned_map(
        obs_list=obs_list,
        output_sky_map=destriped_map,
        output_hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        dm_list=dm_list,
        tm_list=tm_list,
        component=None,
        baselines_list=best_x,
        baseline_lengths_list=baseline_lengths_list,
    )

    # …and finally get the destriped map from their difference
    # (y - F·a, as in Eq. (17))
    destriped_map[:] = binned_map - destriped_map

    # Remove the mean value from I, as it is meaningless
    mask = np.isfinite(destriped_map[0, :])
    destriped_map[0, mask] -= np.mean(destriped_map[0, mask])
    bytes_in_temporary_buffers += mask.nbytes

    if MPI_ENABLED:
        bytes_in_temporary_buffers = MPI_COMM_WORLD.allreduce(
            bytes_in_temporary_buffers,
            op=mpi4py.MPI.SUM,
        )

    return (
        best_x,
        precond,
        history_of_stopping_factors,
        best_stopping_factor,
        converged,
        bytes_in_temporary_buffers,
    )


def destriper_log_callback(
    stopping_factor: float, step_number: int, max_steps: int
) -> None:
    """A function called by :func:`.make_destriped_map` for every CG iteration

    This function has the purpose to produce a visual feedback while the destriper
    is running the (potentially long) Conjugated Gradient iteration.
    """
    if MPI_COMM_WORLD.rank == 0:
        logging.info(
            f"Destriper CG iteration {step_number + 1}/{max_steps}, "
            f"stopping factor: {stopping_factor:.3e}"
        )


def make_destriped_map(
    nside: int,
    observations: Union[Observation, List[Observation]],
    pointings: Optional[Union[npt.ArrayLike, List[npt.ArrayLike]]] = None,
    hwp: Optional[HWP] = None,
    params: DestriperParameters = DestriperParameters(),
    components: Optional[List[str]] = None,
    keep_weights: bool = False,
    keep_pixel_idx: bool = False,
    keep_pol_angle_rad: bool = False,
    detector_split: str = "full",
    time_split: str = "full",
    baselines_list: Optional[List[npt.ArrayLike]] = None,
    recycled_convergence: bool = False,
    callback: Any = destriper_log_callback,
    callback_kwargs: Optional[Dict[Any, Any]] = None,
) -> DestriperResult:
    """
    Applies the destriping algorithm to produce a map out from a TOD

    This function takes the samples stored in the list of
    :class:`.Observation` objects `observations` and produces a destriped map.
    Pointings can either be embedded in the `observations` objects or provided
    through the parameter `pointings`.

    The `params` argument is an instance of the class
    :class:`.DestriperParameters`, and it is used to tune the
    way the destriper should solve the map-making equation.
    Have a look at the documentation for this class to get
    more information.

    The samples in the TOD can be saved in several fields within each
    :class:`.Observation` object. For instance, you could have
    generated a noise component in `observations.noise_tod`, the dipole in
    `observations.dipole_tod`, etc., and you want to produce a map containing
    the *sum* of all these components. In this case, you can pass
    a list of strings containing the name of the fields as the
    parameter `components`, and the destriper will add them together
    before creating the map.

    To show a few clues about the progress of the destriping algorithm,
    the function accepts the `callback` argument. This must be a function
    with the following prototype::

        def callback(
            stopping_factor: float, step_number: int, max_steps: int
        ):
            ...

    The parameter `stopping_factor` is the current stopping factor, i.e.,
    the maximum residual between the current estimate of the baselines
    and the result of applying the binning to the destriped TOD. The
    parameter `step_number` is an integer between 0 and ``max_steps - 1``
    and it is increased by 1 each time the callback is called. The
    callback function can accept more parameters; in this case, their
    value must be passed using the `callback_kwargs` parameter. For
    instance, you might be running the destriper within a graphical
    program and you would show a progress bar dialog; in this case, one
    of the additional arguments might be an handle to the progress
    window::

        def my_gui_callback(
            stopping_factor: float, step_number: int, max_steps: int,
            window_handle: GuiWindow,
        ):
            window_handle.set_progress(step_number / (max_steps - 1))
            window_handle.text_label.set(
                f"Stopping factor is {stopping_factor:.2e}"
            )

    To use it, you must call ``make_destriped_map`` as follows::

        my_window_handle = CreateProgressWindow(...)
        make_destriped_map(
            observations=obs_list,
            params=params,
            callback=my_gui_callback,
            callback_kwargs={"window_handle": my_window_handle},
        )

    :param observations: an instance of the class :class:`.Observation`,
       or a list of objects of this kind
    :param pointings: a 3×N array containing the values of the
       θ,φ,ψ angles (in radians), or a list if `observations` was a
       list. If no pointings are specified, they will be
       taken from `observations` (the most common situation)
    :param params: an instance of the :class:`.DestriperParameters` class
    :param components: a list of components to extract from
       the TOD and sum together. The default is to use `observations.tod`.
    :param keep_weights: the destriper adds a `destriper_weights`
       field to each :class:`.Observation` object in `observations`, and
       it deletes it once the map has been produced. Setting
       this parameter to ``True`` prevents the field from being
       deleted. (Useful for debugging.)
    :param keep_pixel_idx: same as `keep_weights`, but the
       field to be kept from the :class:`.Observation`
       classes is `destriper_pixel_idx`
    :param keep_pol_angle_rad: same as `keep_weights`, but the
       field to be kept from the :class:`.Observation`
       classes is `destriper_pol_angle_rad`
    :param callback: a function that is called during each
       iteration of the CG routine. It is meant as a way to
       provide some visual feedback to the user. The default
       implementation uses the ``logging`` library to print
       a line. You can provide a custom callback, if you
       want (see above for an example).
    :param callback_kwargs: additional keyword arguments to be
       passed to the `callback` function
    :return: an instance of the :class:`.DestriperResult`
       containing the destriped map and other useful information
    """
    elapsed_time_s = time.monotonic()

    if not components:
        components = ["tod"]

    do_destriping = params.samples_per_baseline is not None

    obs_list, ptg_list = _normalize_observations_and_pointings(
        observations=observations, pointings=pointings
    )

    hpx = Healpix_Base(nside=nside, scheme="RING")

    # Convert pointings and ψ angles according to the coordinate system,
    # convert them into Healpix indices and save the result into
    # each Observation object (don't worry, we will delete them
    # later)
    _store_pixel_idx_and_pol_angle_in_obs(
        hpx=hpx,
        obs_list=obs_list,
        ptg_list=ptg_list,
        hwp=hwp,
        output_coordinate_system=params.output_coordinate_system,
    )

    if len(components) > 1:
        # It is often the case that one asks to create a map out of a
        # sum of TODs (e.g., “CMB”, “synchrotron”, “1/f noise”, “white
        # noise”, etc.). It is a burden for the destriping code to take
        # into account the sum of all the components in each function,
        # so we sum all the TODs into the first one and only use that
        # in the map-making. We'll revert the change later
        _sum_components_into_obs(
            obs_list=obs_list,
            target=components[0],
            other_components=components[1:],
            factor=+1.0,
        )

    detector_mask_list = _build_mask_detector_split(detector_split, obs_list)

    time_mask_list = _build_mask_time_split(time_split, obs_list)

    nobs_matrix_cholesky = _build_nobs_matrix(
        hpx=hpx,
        obs_list=obs_list,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
    )

    number_of_pixels = hpx.npix()
    binned_map = np.empty((3, number_of_pixels))
    hit_map = np.empty(number_of_pixels)

    if do_destriping:
        try:
            # This will fail if the parameter is a scalar
            len(params.samples_per_baseline)

            baseline_lengths_list = params.samples_per_baseline
            assert len(baseline_lengths_list) == len(obs_list), (
                f"The list baseline_lengths_list has {len(baseline_lengths_list)} "
                f"elements, but there are {len(obs_list)} observations"
            )
        except TypeError:
            # Ok, params.samples_per_baseline is a scalar, so we must
            # figure out the number of samples in each baseline within
            # each observation
            baseline_lengths_list = [
                split_items_evenly(
                    n=getattr(cur_obs, components[0]).shape[1],
                    sub_n=int(params.samples_per_baseline),
                )
                for cur_obs in obs_list
            ]

        # Each element of this list is a 2D array with shape (N_det, N_baselines),
        # where N_det is the number of detectors in the i-th Observation object
        recycle_baselines = False
        if baselines_list is None:
            baselines_list = [
                np.zeros((getattr(cur_obs, components[0]).shape[0], len(cur_baseline)))
                for (cur_obs, cur_baseline) in zip(obs_list, baseline_lengths_list)
            ]
        else:
            recycle_baselines = True

        destriped_map = np.empty((3, number_of_pixels))
        (
            baselines_list,
            baseline_errors_list,
            history_of_stopping_factors,
            best_stopping_factor,
            converged,
            bytes_in_temporary_buffers,
        ) = _run_destriper(
            obs_list=obs_list,
            nobs_matrix_cholesky=nobs_matrix_cholesky,
            binned_map=binned_map,
            destriped_map=destriped_map,
            hit_map=hit_map,
            baseline_lengths_list=baseline_lengths_list,
            baselines_list_start=baselines_list,
            recycle_baselines=recycle_baselines,
            recycled_convergence=recycled_convergence,
            dm_list=detector_mask_list,
            tm_list=time_mask_list,
            component=components[0],
            threshold=params.threshold,
            max_steps=params.iter_max,
            use_preconditioner=params.use_preconditioner,
            callback=callback,
            callback_kwargs=callback_kwargs if callback_kwargs else {},
        )

        if MPI_ENABLED:
            bytes_in_temporary_buffers = MPI_COMM_WORLD.allreduce(
                bytes_in_temporary_buffers,
                op=mpi4py.MPI.SUM,
            )
    else:
        # No need to run the destriping, just compute the binned map with
        # one single baseline set to zero
        _compute_binned_map(
            obs_list=obs_list,
            output_sky_map=binned_map,
            output_hit_map=hit_map,
            nobs_matrix_cholesky=nobs_matrix_cholesky,
            component=components[0],
            dm_list=detector_mask_list,
            tm_list=time_mask_list,
            baselines_list=None,
            baseline_lengths_list=[
                np.array([getattr(cur_obs, components[0]).shape[1]], dtype=int)
                for cur_obs in obs_list
            ],
        )
        bytes_in_temporary_buffers = 0

        destriped_map = None
        baseline_lengths_list = None
        baselines_list = None
        baseline_errors_list = None
        history_of_stopping_factors = None
        best_stopping_factor = None
        converged = True

    # Add the temporary memory that was allocated *before* calling the destriper
    bytes_in_temporary_buffers += sum(
        [
            cur_obs.destriper_weights.nbytes
            + cur_obs.destriper_pixel_idx.nbytes
            + cur_obs.destriper_pol_angle_rad.nbytes
            for cur_obs in obs_list
        ]
    )

    # We're nearly done! Let's clean up some stuff…
    if not keep_weights:
        for cur_obs in obs_list:
            del cur_obs.destriper_weights
    if not keep_pixel_idx:
        for cur_obs in obs_list:
            del cur_obs.destriper_pixel_idx
    if not keep_pol_angle_rad:
        for cur_obs in obs_list:
            del cur_obs.destriper_pol_angle_rad

    # Make sure that the `del` statements above are applied immediately
    gc.collect()

    # Revert the value of the first TOD component, if other
    # components were added to it
    if len(components) > 1:
        _sum_components_into_obs(
            obs_list=obs_list,
            target=components[0],
            other_components=components[1:],
            factor=-1.0,
        )

    elapsed_time_s = time.monotonic() - elapsed_time_s
    return DestriperResult(
        params=params,
        nside=nside,
        components=components,
        hit_map=hit_map,
        binned_map=binned_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        coordinate_system=params.output_coordinate_system,
        detector_split=detector_split,
        time_split=time_split,
        # The following fields are filled only if the CG algorithm was used
        baselines=baselines_list,
        baseline_errors=baseline_errors_list,
        baseline_lengths=baseline_lengths_list,
        history_of_stopping_factors=history_of_stopping_factors,
        stopping_factor=best_stopping_factor,
        destriped_map=destriped_map,
        converged=converged,
        elapsed_time_s=elapsed_time_s,
        bytes_in_temporary_buffers=bytes_in_temporary_buffers,
    )


@njit
def _remove_baselines(
    tod: npt.ArrayLike, baselines: npt.ArrayLike, baseline_lengths: npt.ArrayLike
):
    num_of_detectors, num_of_samples = tod.shape
    for det_idx in range(num_of_detectors):
        baseline_idx = 0
        samples_in_this_baseline = 0
        for sample_idx in range(num_of_samples):
            tod[det_idx, sample_idx] -= baselines[det_idx, baseline_idx]
            (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                baseline_idx, samples_in_this_baseline, baseline_lengths
            )


def remove_baselines_from_tod(
    obs_list: List[Observation],
    baselines: List[npt.ArrayLike],
    baseline_lengths: List[npt.ArrayLike],
    component: str = "tod",
) -> None:
    """
    Subtract 1/f baselines from the TODs in `obs_list`

    This functions subtracts the baselines in `baselines` from the
    `component` in the :class:`.Observation` objects listed within
    the argument `obs_list`. The two lists `baselines` and
    `baseline_lengths` must have the same number of elements as
    `obs_list`.

    :param obs_list: A list of :class:`.Observation` objects
        containing the TOD to be cleaned from baselines
    :param baselines: A list of NumPy arrays of floats (one per
        each item in `obs_list`): each value is a baseline to
        subtract
    :param baseline_lengths: A list of NumPy arrays of integers
        (one per each item in `obs_list`). Each array contains
        the number of samples within each baseline in the
        argument `baselines`; the sum of all the values within
        each array must be equal to the number of samples in
        the TOD within the corresponding element in `obs_list`.
    :param component: The name of the TOD component to clean
        from the baselines. The default is ``"tod"``.
    """

    for cur_obs, cur_baseline, cur_baseline_lengths in zip(
        obs_list, baselines, baseline_lengths
    ):
        cur_tod = getattr(cur_obs, component)
        samples_in_baselines = np.sum(cur_baseline_lengths)
        samples_in_tod = cur_tod.shape[1]
        assert samples_in_baselines == samples_in_tod, (
            f"There are {samples_in_tod} samples in the observation, but "
            f"baselines cover {samples_in_baselines} samples"
        )
        _remove_baselines(
            tod=cur_tod,
            baselines=cur_baseline,
            baseline_lengths=cur_baseline_lengths,
        )


def remove_destriper_baselines_from_tod(
    obs_list: List[Observation],
    destriper_result: DestriperResult,
    component: str = "tod",
):
    """
    A wrapper around :func:`.remove_baselines_from_tod`

    This method removes the baselines computed by :func:`.make_destriped_map` from
    the TOD in the observations `obs_list`. The TOD component is specified
    by the string `component`, which is ``tod`` by default.

    :param obs_list: A list of :class:`.Observation` objects
        containing the TOD to be cleaned from baselines
    :param destriper_result: The result of a call to
        :func:`.make_destriped_map`
    :param component: The name of the TOD component to clean.
        The default is ``"tod"``.
    """

    remove_baselines_from_tod(
        obs_list=obs_list,
        baselines=destriper_result.baselines,
        baseline_lengths=destriper_result.baseline_lengths,
        component=component,
    )


def _save_rank0_destriper_results(results: DestriperResult, output_file: Path) -> None:
    from astropy.io import fits
    from datetime import datetime
    from litebird_sim import write_healpix_map_to_hdu

    destriping_flag = results.destriped_map is not None

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header.add_comment(
        f"Created by the LiteBIRD Simulation Framework on {datetime.utcnow()}"
    )
    primary_hdu.header["MPISIZE"] = (
        MPI_COMM_WORLD.size,
        "Number of MPI processes used in the computation",
    )
    primary_hdu.header["TMPBUFS"] = (
        results.bytes_in_temporary_buffers,
        "Size of temporary buffers [bytes]",
    )
    primary_hdu.header["ELAPSEDT"] = (results.elapsed_time_s, "Wall clock time [s]")

    primary_hdu.header["DSPLIT"] = (results.detector_split, "Detector split")

    primary_hdu.header["TSPLIT"] = (results.detector_split, "Time split")

    hdu_list = [primary_hdu]

    if destriping_flag:
        hdu_list.append(
            write_healpix_map_to_hdu(
                pixels=results.destriped_map,
                coord=coord_sys_to_healpix_string(results.coordinate_system),
                column_names=("I", "Q", "U"),
                column_units=["K"] * 3,
                name="DESTRMAP",
            ),
        )

    hdu_list.append(
        write_healpix_map_to_hdu(
            pixels=results.binned_map,
            coord=coord_sys_to_healpix_string(results.coordinate_system),
            column_names=("I", "Q", "U"),
            column_units=["K"] * 3,
            name="BINMAP",
        ),
    )

    hdu_list.append(
        write_healpix_map_to_hdu(
            pixels=results.hit_map,
            coord=coord_sys_to_healpix_string(results.coordinate_system),
            column_names=["HITS"],
            column_units=["1/K^2"],
            name="HITMAP",
        ),
    )

    components_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="COMPNAME",
                array=np.array(results.components),
                format="{}A".format(max([len(x) for x in results.components])),
            )
        ]
    )
    components_hdu.name = "COMPS"
    hdu_list.append(components_hdu)

    nobs_matrix_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="COEFFS",
                array=results.nobs_matrix_cholesky.nobs_matrix,
                format="6D",
            )
        ]
    )
    nobs_matrix_hdu.name = "NOBSMATR"
    nobs_matrix_hdu.header["ISCHOL"] = (
        results.nobs_matrix_cholesky.is_cholesky,
        "Does the matrix contain Cholesky transforms?",
    )
    hdu_list.append(nobs_matrix_hdu)

    visibility_hdu = write_healpix_map_to_hdu(
        pixels=results.nobs_matrix_cholesky.valid_pixel,
        coord=coord_sys_to_healpix_string(results.coordinate_system),
        dtype=bool,
        column_names=["VALID"],
        column_units=[""],
        name="MASK",
    )
    hdu_list.append(visibility_hdu)

    if destriping_flag:
        history_hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="RZ", array=results.history_of_stopping_factors, format="1D"
                )
            ]
        )
        history_hdu.name = "HISTORY"

        history_hdu.header["ITERMAX"] = (
            results.params.iter_max,
            "Upper limit on the number of iterations",
        )
        history_hdu.header["THRESHLD"] = (
            results.params.threshold,
            "Threshold to stop the iterations",
        )
        history_hdu.header["STOPFACT"] = (
            results.stopping_factor,
            "Actual value of the stopping factor",
        )
        history_hdu.header["CONVERG"] = (
            results.converged,
            "Has the destriper converged?",
        )
        history_hdu.header["PRECOND"] = (
            results.params.use_preconditioner,
            "Was the preconditioner used?",
        )

        hdu_list.append(history_hdu)

    with output_file.open("wb") as outf:
        fits.HDUList(hdu_list).writeto(outf, overwrite=True)


def _save_baselines(results: DestriperResult, output_file: Path) -> None:
    from astropy.io import fits

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["MPIRANK"] = (
        MPI_COMM_WORLD.rank,
        "The rank of the MPI process that wrote this file",
    )
    primary_hdu.header["MPISIZE"] = (
        MPI_COMM_WORLD.size,
        "The number of MPI processes used in the computation",
    )

    hdu_list = [primary_hdu]

    idx = 0
    for cur_baseline, cur_error, cur_lengths in zip(
        results.baselines, results.baseline_errors, results.baseline_lengths
    ):
        baseline_hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name=f"BSL{det_idx:05d}",
                    array=cur_baseline[det_idx, :],
                    format="1E",
                    unit="K",
                )
                for det_idx in range(cur_baseline.shape[0])
            ]
        )
        baseline_hdu.header["NUMDETS"] = (cur_baseline.shape[0], "Number of detectors")

        error_hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name=f"ERR{idx:05d}",
                    array=cur_error,
                    format="1E",
                    unit="K",
                ),
            ]
        )

        length_hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name="LENGTH", array=cur_lengths, unit="", format="1J")]
        )

        for hdu_base_name, this_hdu in (
            ("BSL", baseline_hdu),
            ("ERR", error_hdu),
            ("LEN", length_hdu),
        ):
            this_hdu.header["OBSIDX"] = (
                idx,
                "Index of the Observation object within this MPI process",
            )
            this_hdu.header["NUMSAMP"] = (
                np.sum(cur_lengths),
                "Number of samples covered by these baselines",
            )
            this_hdu.name = f"{hdu_base_name}{idx:05d}"

        hdu_list += [baseline_hdu, error_hdu, length_hdu]
        idx += 1

    hdu_list[0].header["NUMOBS"] = (idx, "Number of observations")

    with output_file.open("wb") as outf:
        fits.HDUList(hdu_list).writeto(outf, overwrite=True)


def save_destriper_results(
    results: DestriperResult,
    output_folder: Path,
    custom_dest_file: Optional[str] = None,
    custom_base_file: Optional[str] = None,
) -> None:
    """
    Save the results of a call to :func:`.make_destriped_map` to disk

    The results are saved in a set of FITS files:

    1. A FITS file containing the maps (destriped, binned, hits), the N_obs matrix,
       and general information about the convergence
    2. A set of FITS files containing the baselines. Each MPI process writes *one*
       file containing its baselines.

    The only parameter that is not saved is the field
    ``results.params.samples_per_baseline``: its type is very versatile
    and would not fit well in the FITS file format. Moreover, the necessary
    information is already available in the baseline lengths that are
    saved in the files (see point 2. above).

    To load the results from the files, use :func:`.load_destriper_results`.

    :param results: The result of the call to :func:`.make_destriped_map`

    :param output_folder: The folder where to save all the FITS files
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Only MPI process #0 saves the file with the maps
    if MPI_COMM_WORLD.rank == 0:
        if custom_dest_file:
            _save_rank0_destriper_results(
                results=results, output_file=output_folder / custom_dest_file
            )
        else:
            _save_rank0_destriper_results(
                results=results,
                output_file=output_folder / __DESTRIPER_RESULTS_FILE_NAME,
            )

    # Now let's save the baselines: one per each observation
    if results.destriped_map is not None:
        if custom_base_file:
            _save_baselines(results, output_file=output_folder / custom_base_file)
        else:
            _save_baselines(results, output_file=output_folder / __BASELINES_FILE_NAME)


def _load_rank0_destriper_results(file_path: Path) -> DestriperResult:
    from astropy.io import fits

    with fits.open(file_path) as inpf:
        nobs_matrix = NobsMatrix(
            nobs_matrix=inpf["NOBSMATR"].data.field("COEFFS"),
            valid_pixel=np.array(inpf["MASK"].data.field("VALID"), dtype=bool),
            is_cholesky=bool(inpf["NOBSMATR"].header["ISCHOL"]),
        )
        nside = hp.npix2nside(len(nobs_matrix.valid_pixel))

        if "GAL" in str(inpf["HITMAP"].header["COORDSYS"]).upper():
            coord_sys = CoordinateSystem.Galactic
        else:
            coord_sys = CoordinateSystem.Ecliptic

        result = DestriperResult(
            nside=nside,
            params=DestriperParameters(
                output_coordinate_system=CoordinateSystem.Galactic,
                samples_per_baseline=0,
                iter_max=0,
                threshold=0,
                use_preconditioner=False,
            ),
            components=[x.strip() for x in inpf["COMPS"].data.field("COMPNAME")],
            nobs_matrix_cholesky=nobs_matrix,
            hit_map=inpf["HITMAP"].data.field("HITS"),
            binned_map=np.array(
                [inpf["BINMAP"].data.field(comp) for comp in ("I", "Q", "U")]
            ),
            coordinate_system=coord_sys,
            detector_split=inpf[0].header["DSPLIT"],
            time_split=inpf[0].header["TSPLIT"],
            history_of_stopping_factors=[],
            elapsed_time_s=inpf[0].header["ELAPSEDT"],
            destriped_map=None,
            converged=False,
            stopping_factor=None,
            baselines=None,
            baseline_lengths=None,
            baseline_errors=None,
            bytes_in_temporary_buffers=inpf[0].header["TMPBUFS"],
        )

        if "DESTRMAP" in inpf:
            result.destriped_map = np.array(
                [inpf["DESTRMAP"].data.field(comp) for comp in ("I", "Q", "U")]
            )
            result.converged = inpf["HISTORY"].header["CONVERG"]
            result.stopping_factor = inpf["HISTORY"].header["STOPFACT"]

            result.params.iter_max = int(inpf["HISTORY"].header["ITERMAX"])
            result.params.threshold = float(inpf["HISTORY"].header["THRESHLD"])
            result.params.use_preconditioner = bool(inpf["HISTORY"].header["PRECOND"])
            result.params.output_coordinate_system = coord_sys

            result.history_of_stopping_factors = [
                float(x) for x in inpf["HISTORY"].data.field("RZ")
            ]

    return result


def load_destriper_results(
    folder: Path,
    custom_dest_file: Optional[str] = None,
    custom_base_file: Optional[str] = None,
) -> DestriperResult:
    """
    Load the results of a call to :func:`.make_destriped_map` from disk

    This function complements :func:`.save_destriper_results`.
    It re-creates an object of type :class:`.DestriperResult` from a
    set of FITS files. If you are calling this function from multiple
    MPI processes, you must ensure that it is called by *every* MPI
    process at the same time, and that the number of MPI processes is
    the same that was used when the data were saved. (The function checks
    for this and halts its execution if the condition is not satisfied.)

    :param folder: The folder containing the FITS files to load
    :return: A new :class:`.DestriperResult` object
    """

    from astropy.io import fits

    # We run this on *all* the MPI processes, as it might be that each of them
    # needs this information!
    if custom_dest_file:
        result = _load_rank0_destriper_results(folder / custom_dest_file)
    else:
        result = _load_rank0_destriper_results(folder / __DESTRIPER_RESULTS_FILE_NAME)

    if result.destriped_map is not None:
        result.baselines = []
        result.baseline_errors = []
        result.baseline_lengths = []

        if custom_base_file:
            baselines_file_name = folder / custom_base_file
        else:
            baselines_file_name = folder / __BASELINES_FILE_NAME

        with fits.open(baselines_file_name) as inpf:
            assert MPI_COMM_WORLD.rank == inpf[0].header["MPIRANK"], (
                "You must call load_destriper_results using the "
                "same MPI layout that was used for save_destriper_results "
            )
            assert MPI_COMM_WORLD.size == inpf[0].header["MPISIZE"], (
                "You must call load_destriper_results using the "
                "same MPI layout that was used for save_destriper_results"
            )

            num_of_obs = int(inpf[0].header["NUMOBS"])
            for obs_idx in range(num_of_obs):
                baselines_hdu = inpf[f"BSL{obs_idx:05d}"]
                num_of_detectors = baselines_hdu.header["NUMDETS"]
                result.baselines.append(
                    np.array(
                        [
                            baselines_hdu.data.field(f"BSL{det_idx:05d}")
                            for det_idx in range(num_of_detectors)
                        ]
                    ),
                )
                result.baseline_errors.append(
                    np.array(inpf[f"ERR{obs_idx:05d}"].data.field(f"ERR{obs_idx:05d}")),
                )
                result.baseline_lengths.append(
                    inpf[f"LEN{obs_idx:05d}"].data.field("LENGTH")
                )

    return result
