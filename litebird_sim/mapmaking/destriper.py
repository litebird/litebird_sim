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

import numpy as np
import numpy.typing as npt
from ducc0.healpix import Healpix_Base
from numba import njit
import healpy as hp

from litebird_sim.mpi import MPI_ENABLED, MPI_COMM_WORLD
from typing import Union, List, Optional, Tuple
from litebird_sim.observations import Observation
from litebird_sim.coordinates import CoordinateSystem

from .common import (
    _compute_pixel_indices,
    _normalize_observations_and_pointings,
    COND_THRESHOLD,
    get_map_making_weights,
    cholesky,
    solve_cholesky,
    estimate_cond_number,
)

if MPI_ENABLED:
    import mpi4py.MPI


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
    `hit_map`, `binned_map`, `destriped_map`, and `nobs_matrix_cholesky`
    contain the *global* map, but `baselines` and `baseline_lengths` only
    refer to the TODs within the *current* MPI process.
    """

    params: DestriperParameters
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
    converged: bool


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


@njit
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


@njit
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

    sin_term = np.sin(2 * pol_angle_rad)
    cos_term = np.cos(2 * pol_angle_rad)

    dest_array[0] += sample / weight
    dest_array[1] += sample * cos_term / weight
    dest_array[2] += sample * sin_term / weight


@njit
def _update_sum_map_with_tod(
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    tod: npt.ArrayLike,
    pol_angle_rad: npt.ArrayLike,
    pixel_idx: npt.ArrayLike,
    weights: npt.ArrayLike,
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
        cur_weight = weights[det_idx]

        baseline_idx = 0
        samples_in_this_baseline = 0

        for sample_idx in range(tod.shape[1]):
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
        cur_weight = weights[det_idx]

        baseline_idx = 0
        samples_in_this_baseline = 0

        for sample_idx in range(pixel_idx.shape[1]):
            cur_pix = pixel_idx[det_idx, sample_idx]
            _sum_map_contribution_from_one_sample(
                pol_angle_rad=pol_angle_rad[det_idx, sample_idx],
                sample=baselines[baseline_idx],
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

    The result is saved in `sky_map` (a 3,N_p tensor) and `hit_map`
    (a N_p vector).
    """

    assert nobs_matrix_cholesky.is_cholesky, (
        "The parameter nobs_matrix_cholesky should already "
        "contain the Cholesky decompositions of the 3×3 M_i matrices"
    )

    assert (baselines_list is not None) or (component is not None), (
        "To call _compute_binned_map you must either provide "
        "the baselines or the TOD component"
    )

    # Step 1: compute the “sum map” (Eqq. 18–20)
    output_sky_map[:] = 0
    output_hit_map[:] = 0

    for obs_idx, (cur_obs, cur_baseline_lengths) in enumerate(
        zip(obs_list, baseline_lengths_list)
    ):
        if baselines_list is None:
            _update_sum_map_with_tod(
                sky_map=output_sky_map,
                hit_map=output_hit_map,
                tod=getattr(cur_obs, component),
                pol_angle_rad=cur_obs.destriper_pol_angle_rad,
                pixel_idx=cur_obs.destriper_pixel_idx,
                weights=cur_obs.destriper_weights,
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
def estimate_sample_from_map(cur_pixel, cur_psi, sky_map) -> float:
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

    for (det_idx, cur_weight) in enumerate(weights):
        det_pixel_idx = pixel_idx[det_idx, :]
        det_psi_angle_rad = psi_angle_rad[det_idx, :]

        baseline_idx = 0
        samples_in_this_baseline = 0

        for sample_idx in range(len(det_pixel_idx)):
            map_value = estimate_sample_from_map(
                cur_pixel=det_pixel_idx[sample_idx],
                cur_psi=det_psi_angle_rad[sample_idx],
                sky_map=sky_map,
            )
            output_sums[baseline_idx] += (
                tod[det_idx, sample_idx] - map_value
            ) / cur_weight

            (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                baseline_idx, samples_in_this_baseline, baseline_length
            )


@njit
def _compute_baseline_sums_for_one_component(
    weights: npt.ArrayLike,
    pixel_idx: npt.ArrayLike,
    psi_angle_rad: npt.ArrayLike,
    sky_map: npt.ArrayLike,
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

    for (det_idx, cur_weight) in enumerate(weights):
        det_pixel_idx = pixel_idx[det_idx, :]
        det_psi_angle_rad = psi_angle_rad[det_idx, :]

        baseline_idx = 0
        samples_in_this_baseline = 0

        for sample_idx in range(len(det_pixel_idx)):
            map_value = estimate_sample_from_map(
                cur_pixel=det_pixel_idx[sample_idx],
                cur_psi=det_psi_angle_rad[sample_idx],
                sky_map=sky_map,
            )
            output_sums[baseline_idx] += (
                baselines[baseline_idx] - map_value
            ) / cur_weight

            (baseline_idx, samples_in_this_baseline) = _step_over_baseline(
                baseline_idx, samples_in_this_baseline, baseline_length
            )


def _compute_baseline_sums(
    obs_list: List[Observation],
    sky_map: npt.ArrayLike,
    baselines_list: Optional[List[npt.ArrayLike]],
    baseline_lengths_list: List[npt.ArrayLike],
    component: Optional[str],
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
    for obs_idx, (cur_obs, cur_baseline_lengths, cur_sums) in enumerate(
        zip(obs_list, baseline_lengths_list, output_sums_list)
    ):
        assert len(cur_baseline_lengths) == len(cur_sums), (
            f"The output buffer for observation {obs_idx=} "
            f"has room for {len(cur_sums)=} elements, but there"
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
                baseline_length=cur_baseline_lengths,
                output_sums=cur_sums,
            )


def _mpi_dot(a: List[npt.ArrayLike], b: List[npt.ArrayLike]):
    local_result = sum([np.dot(x1, x2) for (x1, x2) in zip(a, b)])
    if MPI_ENABLED:
        return MPI_COMM_WORLD.allreduce(local_result, op=mpi4py.MPI.SUM)
    else:
        return local_result


def _get_stopping_factor(residual: List[npt.ArrayLike]) -> float:
    local_result = max([np.max(np.abs(cur_baseline)) for cur_baseline in residual])
    if MPI_ENABLED:
        return MPI_COMM_WORLD.allreduce(local_result, op=mpi4py.MPI.MAX)
    else:
        return local_result


def _compute_b_or_Ax(
    obs_list: List[Observation],
    nobs_matrix_cholesky: NobsMatrix,
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
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
        output_sky_map=sky_map,
        output_hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component=component,
    )

    _compute_baseline_sums(
        obs_list=obs_list,
        sky_map=sky_map,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component=component,
        output_sums_list=result,
    )


def compute_b(
    obs_list: List[Observation],
    nobs_matrix_cholesky: NobsMatrix,
    sky_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    baseline_lengths_list: List[npt.ArrayLike],
    component: str,
    result: List[npt.ArrayLike],
) -> None:
    _compute_b_or_Ax(
        obs_list=obs_list,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        sky_map=sky_map,
        hit_map=hit_map,
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
    baselines_list: List[npt.ArrayLike],
    baseline_lengths_list: List[npt.ArrayLike],
    result: List[npt.ArrayLike],
) -> None:
    _compute_b_or_Ax(
        obs_list=obs_list,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        sky_map=sky_map,
        hit_map=hit_map,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component=None,
        result=result,
    )


def _compute_preconditioner(obs_list, baseline_lengths_list) -> List[npt.ArrayLike]:
    # We just compute (F^T·C_w⁻¹·F)⁻¹, which is a diagonal matrix containing
    # the number of elements in each baseline divided by σ². (Remember
    # that the field `destriper_weights` already contain σ².)
    #
    # This is the most common choice, but it's not necessarily the best one.
    # See for instance these papers:
    #
    # 1. A fast map-making preconditioner for regular scanning patterns
    #    (Naess & al., 2014)
    #
    # 2. Accelerating the cosmic microwave background map-making procedure
    #    through preconditioning (Szydlarski, 2014)
    #
    # Our choice corresponds to Eq. (10) in Szydlarski's paper.

    return [
        cur_obs.destriper_weights / cur_baseline_lengths
        for cur_obs, cur_baseline_lengths in zip(obs_list, baseline_lengths_list)
    ]


def _apply_preconditioner(precond: List[npt.ArrayLike], z: List[npt.ArrayLike]):
    for precond_k, z_k in zip(precond, z):
        precond_k *= z_k


def _run_destriper(
    obs_list: List[Observation],
    nobs_matrix_cholesky: NobsMatrix,
    binned_map: npt.ArrayLike,
    destriped_map: npt.ArrayLike,
    hit_map: npt.ArrayLike,
    baseline_lengths_list: List[npt.ArrayLike],
    baselines_list_start: List[npt.ArrayLike],
    component: str,
    threshold: float,
    max_steps: int,
    use_preconditioner: bool,
) -> Tuple[List[npt.ArrayLike], List[npt.ArrayLike], List[float], float, bool]:
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

    assert nobs_matrix_cholesky.is_cholesky, (
        "_run_destriper requires that `nobs_matrix_cholesky` "
        "already contains the Cholesky transforms"
    )

    assert len(obs_list) == len(baselines_list_start)

    # We allocate all the memory in advance: this ensures the code is fast
    # and prevents memory fragmentation

    # Preallocate memory for the baselines at the k-th step
    x = [np.copy(cur_baseline) for cur_baseline in baselines_list_start]

    # This is the “best” baseline found during the iterations. Normally,
    # the best one is the last one, unless the loop was cut short because
    # the maximum number of iterations was reached.
    best_x = [np.copy(cur_baseline) for cur_baseline in baselines_list_start]
    best_stopping_factor = None

    # The `b` value used by the Wikipedia article corresponds to
    # Kurki-Suonio's F^t·C_w⁻¹·Z·F·a
    b = [
        np.empty(len(cur_baseline_lengths))
        for cur_baseline_lengths in baseline_lengths_list
    ]
    # A·x corresponds to F^t·C_w⁻¹·Z·y
    Ax = [
        np.empty(len(cur_baseline_lengths))
        for cur_baseline_lengths in baseline_lengths_list
    ]
    # This is the residual b−A·x; ideally, if we already had the correct solution,
    # it should be zero.
    r = [
        np.empty(len(cur_baseline_lengths))
        for cur_baseline_lengths in baseline_lengths_list
    ]

    # Initialize r_k
    compute_b(
        obs_list=obs_list,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        sky_map=destriped_map,
        hit_map=hit_map,
        baseline_lengths_list=baseline_lengths_list,
        component=component,
        result=b,
    )
    compute_Ax(
        obs_list=obs_list,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        sky_map=destriped_map,
        hit_map=hit_map,
        baselines_list=x,
        baseline_lengths_list=baseline_lengths_list,
        result=Ax,
    )
    for (r_k, b_k, A_k) in zip(r, b, Ax):
        r_k[:] = b_k - A_k

    new_r = [np.copy(r_k) for r_k in r]

    z = [np.copy(r_k) for r_k in r]
    precond = _compute_preconditioner(
        obs_list=obs_list, baseline_lengths_list=baseline_lengths_list
    )
    if use_preconditioner:
        _apply_preconditioner(precond, z)
    k = 0

    old_r_dot = _mpi_dot(z, r)

    history_of_stopping_factors = [_get_stopping_factor(r)]  # type: List[float]
    while True:
        k += 1
        if k >= max_steps:
            converged = False
            break

        compute_Ax(
            obs_list=obs_list,
            nobs_matrix_cholesky=nobs_matrix_cholesky,
            sky_map=destriped_map,
            hit_map=hit_map,
            baselines_list=z,
            baseline_lengths_list=baseline_lengths_list,
            result=Ax,
        )
        α = old_r_dot / _mpi_dot(z, Ax)

        for x_k, z_k in zip(x, z):
            x_k += α * z_k

        for (new_r_k, r_k, Ax_k) in zip(new_r, r, Ax):
            new_r_k[:] = r_k - α * Ax_k

        cur_stopping_factor = _get_stopping_factor(new_r)
        if (not best_stopping_factor) or cur_stopping_factor < best_stopping_factor:
            best_stopping_factor = cur_stopping_factor
            for cur_best_x_k, x_k in zip(best_x, x):
                cur_best_x_k[:] = x_k

        history_of_stopping_factors.append(cur_stopping_factor)
        if cur_stopping_factor < threshold:
            converged = True
            break

        for (z_k, r_k) in zip(z, r):
            z_k[:] = r_k[:]
        if use_preconditioner:
            _apply_preconditioner(z)

        new_r_dot = _mpi_dot(new_r, z)
        for z_k, r_k, new_r_k in zip(z, r, new_r):
            z_k[:] = new_r_k + (new_r_dot / old_r_dot) * z_k

        old_r_dot = new_r_dot
        r = new_r

    # Redo the binned and destriped map with the best solution found so far

    # First, compute the binned map by passing `baselines_list=None`…
    binned_map = np.empty_like(destriped_map)
    _compute_binned_map(
        obs_list=obs_list,
        output_sky_map=binned_map,
        output_hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        component=component,
        baselines_list=None,
        baseline_lengths_list=baseline_lengths_list,
    )

    # …then compute the map from the “unrolled” baselines F·a…
    _compute_binned_map(
        obs_list=obs_list,
        output_sky_map=destriped_map,
        output_hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        component=component,
        baselines_list=best_x,
        baseline_lengths_list=baseline_lengths_list,
    )

    # …and finally get the destriped map from their difference
    # (y - F·a, as in Eq. (17))
    destriped_map[:] = binned_map - destriped_map

    # Remove the mean value from I, as it is meaningless
    mask = np.isfinite(destriped_map[0, :])
    destriped_map[0, mask] -= np.mean(destriped_map[0, mask])

    return best_x, precond, history_of_stopping_factors, best_stopping_factor, converged


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

    do_destriping = params.samples_per_baseline is not None

    obs_list, ptg_list, psi_list = _normalize_observations_and_pointings(
        obs=obs, pointings=pointings
    )

    hpx = Healpix_Base(nside=params.nside, scheme="RING")

    # Convert pointings and ψ angles according to the coordinate system,
    # convert them into Healpix indices and save the result into
    # each Observation object (don't worry, we will delete them
    # later)
    _store_pixel_idx_and_pol_angle_in_obs(
        hpx=hpx,
        obs_list=obs_list,
        ptg_list=ptg_list,
        psi_list=psi_list,
        output_coordinate_system=CoordinateSystem.Ecliptic,
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

    nobs_matrix_cholesky = _build_nobs_matrix(
        hpx=hpx,
        obs_list=obs_list,
        ptg_list=ptg_list,
        psi_list=psi_list,
    )

    number_of_pixels = hpx.npix()
    binned_map = np.empty((3, number_of_pixels))
    hit_map = np.empty(number_of_pixels)

    if do_destriping:
        if type(params.samples_per_baseline) is int:
            baseline_lengths_list = [
                split_items_evenly(
                    n=getattr(cur_obs, components[0]).shape[1],
                    sub_n=params.samples_per_baseline,
                )
                for cur_obs in obs_list
            ]
        else:
            baseline_lengths_list = params.samples_per_baseline

        baselines_list = [
            np.zeros(len(cur_baseline)) for cur_baseline in baseline_lengths_list
        ]

        destriped_map = np.empty((3, number_of_pixels))
        (
            baselines_list,
            baseline_errors_list,
            history_of_stopping_factors,
            best_stopping_factor,
            converged,
        ) = _run_destriper(
            obs_list=obs_list,
            nobs_matrix_cholesky=nobs_matrix_cholesky,
            binned_map=binned_map,
            destriped_map=destriped_map,
            hit_map=hit_map,
            baseline_lengths_list=baseline_lengths_list,
            baselines_list_start=baselines_list,
            component=components[0],
            threshold=params.threshold,
            max_steps=params.iter_max,
            use_preconditioner=params.use_preconditioner,
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
            baselines_list=None,
            baseline_lengths_list=[
                np.array([getattr(cur_obs, components[0]).shape[1]], dtype=int)
                for cur_obs in obs_list
            ],
        )

        destriped_map = None
        baseline_lengths_list = None
        baselines_list = None
        baseline_errors_list = None
        history_of_stopping_factors = None
        best_stopping_factor = None
        converged = True

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

    # Revert the value of the first TOD component
    if len(components) > 1:
        _sum_components_into_obs(
            obs_list=obs_list,
            target=components[0],
            other_components=components[1:],
            factor=-1.0,
        )

    return DestriperResult(
        params=params,
        hit_map=np.zeros(1),
        binned_map=binned_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        coordinate_system=params.output_coordinate_system,
        # The following fields are filled only if the CG algorithm was used
        baselines=baselines_list,
        baseline_errors=baseline_errors_list,
        baseline_lengths=baseline_lengths_list,
        history_of_stopping_factors=history_of_stopping_factors,
        stopping_factor=best_stopping_factor,
        destriped_map=destriped_map,
        converged=converged,
    )
