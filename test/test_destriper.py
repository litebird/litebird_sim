# -*- encoding: utf-8 -*-

# In this file, we test our own implementation of the destriping algorithm.
# We consider a simple TOD with very few samples, so that it's possible to
# write the matrices used to express the destriping problem in full form.
# After having created the matrices in memory and having inverted them,
# we check that the analytical solution is close enough to the output
# of `make_destriped_map`.
#
# The map is made by just 2 pixels, and the TOD contains 7 samples;
# there are two baselines that either contain 3 and 4 samples or
# vice versa

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, List

import astropy.time
import numpy as np
import numpy.typing as npt
from ducc0.healpix import Healpix_Base

import litebird_sim as lbs
from litebird_sim import CoordinateSystem, MPI_COMM_WORLD, MPI_ENABLED
from litebird_sim.mapmaking.destriper import (
    NobsMatrix,
    remove_destriper_baselines_from_tod,
    save_destriper_results,
    load_destriper_results,
)

if MPI_ENABLED:
    pass


# We define the simplest quantities directly, as global variables

# We assume that our TOD contains 7 elements and observes a map of just
# 2 pixels. We only use 2 baselines.
NUM_OF_PIXELS = 2
NUM_OF_SAMPLES = 7
NUM_OF_BASELINES = 2

INPUT_MAPS = np.array(
    [
        1.0,  # First pixel: I
        0.5,  # First pixel: Q
        0.2,  # First pixel: U
        5.0,  # Second pixel: I
        4.0,  # Second pixel: Q
        2.0,  # Second pixel: U
    ]
)

# These seven samples should represent the white-noise component. We pick them
# small enough not to alter the result of the destriping problem too much
NOISE_SAMPLES = np.array(
    [
        -2.23302888e-08,
        1.78939315e-07,
        1.27574038e-07,
        -3.98914056e-09,
        -8.57761576e-08,
        -8.53485713e-08,
        -1.29047524e-07,
    ]
)
assert len(NOISE_SAMPLES) == NUM_OF_SAMPLES

# These are the pixel observed by each of the seven samples in the TOD
# NOTE: if you change this, be sure there are no gaps. For instance,
# an array like [0, 3, 3, 0, 1] would be invalid, as pixel #2 is missing.
PIXEL_INDEXES = np.array(
    [
        0,
        0,
        1,
        1,
        1,
        1,
        0,
    ],
    dtype="int",
)
assert len(PIXEL_INDEXES) == NUM_OF_SAMPLES


# These are the attack angles for the seven samples in the TOD
# If you change these, be sure to properly sample each pixel
# listed in PIXEL_INDEXES (at least three non-degenerate angles
# per pixel must be provided).
#
# Since we want to test the code with 2 MPI processes and we
# rely on the ability of the Observation class to split TODs
# among processes, we cannot tell how many samples should go
# in the first baseline and how many in the second one: either
# it will be 4 in rank #0 and 3 in rank #1, or vice versa.
# Therefore, we choose the angles so that the N_obs matrix
# is non-singular for the two possible baseline lengths
# [4, 3] and [3, 4].
PSI_ANGLES = np.array(
    [
        np.deg2rad(90.0),  # Pixel #1  --*-------
        np.deg2rad(30.0),  # Pixel #1  --*-------
        np.deg2rad(90.0),  # Pixel #2  -------*--
        np.deg2rad(15.0),  # Pixel #2  -------*--
        np.deg2rad(30.0),  # Pixel #2  -------*--
        np.deg2rad(45.0),  # Pixel #2  -------*--
        np.deg2rad(60.0),  # Pixel #1  --*-------
    ]
)
assert len(PSI_ANGLES) == NUM_OF_SAMPLES


# Note that the average of these values is *not* zero. Thus, the
# destriping equation is going to retrieve different numbers, because
# the output of a destriper always produces zero-mean baselines.
BASELINE_VALUES = np.array([10.0, 12.0])


# This class is used to hold information about those matrices used in
# the analytical solution that are too complex to write down explicitly
# and thus need to be calculated.
@dataclass
class AnalyticalSolution:
    num_of_samples_in_tod: int = NUM_OF_SAMPLES
    num_of_baselines: int = NUM_OF_BASELINES
    num_of_observed_pixels: int = NUM_OF_PIXELS
    sigma: float = 1.0
    # Array of N integers, where N is the number of baselines. Each
    # value in the array is the number of TOD samples falling in
    # that baseline. Of course, sum(baseline_runs) == NUM_OF_SAMPLES
    baseline_runs: Any = None
    # Baseline indexes: array of NUM_OF_SAMPLES elements, where each
    # element is the index in the BASELINE_VALUES array
    baseline_indexes: Any = None
    # Pointing matrix (Eq. 2)
    P: Any = None
    # Noise covariance matrix (Eq. 5)
    Cw: Any = None
    # Cw⁻¹
    invCw: Any = None
    # Baseline→TOD projection matrix (Eq. 3)
    F: Any = None
    # N_obs matrix
    M: Any = None
    # Estimate of the noise part of the TOD
    Z: Any = None
    # Baseline noise estimator matrix
    D: Any = None
    # The sky signal, P·m (Eq. 2)
    sky_signal: Any = None
    # The baselines projected in TOD space, F·a (Eq. 3)
    baseline_signal: Any = None
    # The white-noise term `w` (Eq. 3)
    noise_signal: Any = None
    # The input TOD, sum of `sky_signal`, `baseline_signal`, and `noise_signal`
    y: Any = None
    # The input baselines (exact)
    input_a: Any = None
    # The input I/Q/U maps (exact)
    input_maps: Any = None
    # The expected binned map
    expected_binned_map: Any = None
    # The estimated baselines
    estimated_a: Any = None
    # The estimated I/Q/U maps
    estimated_maps: Any = None


def lower_triangular_to_matrix(coefficients):
    return np.array(
        [
            [coefficients[0], 0.0, 0.0],
            [coefficients[1], coefficients[2], 0.0],
            [coefficients[3], coefficients[4], coefficients[5]],
        ],
        dtype=np.float64,
    )


def create_analytical_solution(
    baseline_runs,
    sigma: float = 0.1,
    add_baselines: bool = True,
) -> AnalyticalSolution:
    # We assume that we have *two* baselines!
    assert (
        len(baseline_runs) == 2
    ), "The test code requires that *two* baselines be provided"

    # Beware, the `*` and `+` operators used here work on Python arrays, not
    # on NumPy objects! Their semantics differ! Here we just want
    # to create an array with shape [0, 0, 0, … , 0, 1, 1, 1, … , 1, 1]
    baseline_indexes = np.array([0] * baseline_runs[0] + [1] * baseline_runs[1])

    # Build the pointing matrix P (Eq. 2 of KurkiSuonio2009)
    P = np.zeros((NUM_OF_SAMPLES, 3 * NUM_OF_PIXELS))

    for i, pix_idx, psi in zip(range(NUM_OF_SAMPLES), PIXEL_INDEXES, PSI_ANGLES):
        P[i, 3 * pix_idx] = 1
        P[i, 3 * pix_idx + 1] = np.cos(2 * psi)
        P[i, 3 * pix_idx + 2] = np.sin(2 * psi)

    # Build the noise matrix
    Cw = np.eye(NUM_OF_SAMPLES) * (sigma**2)
    invCw = np.eye(NUM_OF_SAMPLES) / (sigma**2)

    # Build the baseline-projection matrix F (Eq. 3 of KurkiSuonio2009)
    F = np.zeros((NUM_OF_SAMPLES, NUM_OF_BASELINES))
    for idx, baseline_idx in enumerate(baseline_indexes):
        F[idx, baseline_idx] = 1

    # Build the TOD (Eq. 2+3 of KurkiSuonio2009)
    sky_signal = (P @ INPUT_MAPS).reshape(-1)
    baseline_signal = (F @ BASELINE_VALUES).reshape(-1)
    if not add_baselines:
        baseline_signal *= 0.0  # Turn off baselines (no 1/f noise)

    y = sky_signal + baseline_signal + NOISE_SAMPLES

    # Eq. 9 of KurkiSuonio2009
    M = np.transpose(P) @ invCw @ P
    invM = np.linalg.inv(M)

    # Eq. 12 of KurkiSuonio2009
    Z = np.eye(NUM_OF_SAMPLES) - P @ invM @ np.transpose(P) @ invCw

    # Eq. 15 of KurkiSuonio2009
    D = np.transpose(F) @ invCw @ Z @ F

    # Here are the solutions to the destriping problem:
    # Eq. 16 and 17 of KurkiSuonio2009
    # Since here we're looking for an analytical solution, we just
    # compute D⁻¹ and get the solution, but in our destriper we use
    # the conjugate gradient to retrieve an estimate for `a` (the
    # baselines)
    estimated_a = np.linalg.inv(D) @ np.transpose(F) @ invCw @ Z @ y
    estimated_maps = invM @ np.transpose(P) @ invCw @ (y - F @ BASELINE_VALUES)

    return AnalyticalSolution(
        num_of_samples_in_tod=NUM_OF_SAMPLES,
        num_of_baselines=NUM_OF_BASELINES,
        num_of_observed_pixels=NUM_OF_PIXELS,
        sigma=sigma,
        baseline_runs=baseline_runs,
        baseline_indexes=baseline_indexes,
        P=P,
        Cw=Cw,
        invCw=invCw,
        F=F,
        M=M,
        Z=Z,
        D=D,
        sky_signal=sky_signal,
        baseline_signal=baseline_signal,
        noise_signal=NOISE_SAMPLES,
        y=y,
        input_a=BASELINE_VALUES,
        input_maps=INPUT_MAPS,
        expected_binned_map=invM @ np.transpose(P) @ invCw @ y,
        estimated_a=estimated_a,
        estimated_maps=estimated_maps,
    )


def get_baseline_lengths_list(
    expected_solution: AnalyticalSolution,
) -> List[npt.ArrayLike]:
    if MPI_COMM_WORLD.size == 2:
        return [
            np.array([expected_solution.baseline_runs[MPI_COMM_WORLD.rank]], dtype=int)
        ]
    elif MPI_COMM_WORLD.size == 1:
        return [np.array(expected_solution.baseline_runs)]
    else:
        assert False, "This should not happen! Only up to 2 MPI ranks are allowed here"


def setup_simulation(
    sigma: float = 0.1, add_baselines: bool = True
) -> Tuple[lbs.Simulation, AnalyticalSolution]:
    """Create a Simulation object and a AnalyticSolution object that match

    The Simulation object contains the same data as in AnalyticSolution. It
    is meant to be used with the destriper to check that the result matches
    the analytic solution."""

    # The TOD is too short (7 samples) to be meaningful with more than 2 processes
    assert lbs.MPI_COMM_WORLD.size <= 2

    sim = lbs.Simulation(
        start_time=0.0,
        duration_s=NUM_OF_SAMPLES,
        random_seed=12345,
    )

    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                sampling_rate_hz=1.0, name="Mock detector", net_ukrts=sigma
            ),
        ],
        tods=[
            lbs.TodDescription(
                name="sky_signal",
                dtype=np.float32,
                description="The projected signal P·m (Eq. 2 in KurkiSuonio2009)",
            ),
            lbs.TodDescription(
                name="baseline",
                dtype=np.float32,
                description="The projected baselines F·a (Eq. 3 in KurkiSuonio2009)",
            ),
            lbs.TodDescription(
                name="white_noise",
                dtype=np.float32,
                description="The white-noise term `w` (Eq. 3 in KurkiSuonio2009)",
            ),
        ],
        num_of_obs_per_detector=sim.mpi_comm.size,
    )

    assert len(sim.observations) > 0, f"no observations for process {sim.mpi_comm.rank}"

    descr = sim.describe_mpi_distribution()
    num_of_samples = 0
    for cur_mpi_process in descr.mpi_processes:
        assert len(cur_mpi_process.observations) == 1
        num_of_samples += cur_mpi_process.observations[0].num_of_samples

    assert num_of_samples == NUM_OF_SAMPLES

    if lbs.MPI_COMM_WORLD.size == 2:
        # If there are 2 MPI processes, we assign the first baseline
        # to #0 and the second baseline to #1
        baseline_runs = np.array(
            [
                cur_proc.observations[0].num_of_samples
                for cur_proc in descr.mpi_processes
            ]
        )
    else:
        # We're running serially, so the current process will handle
        # *both* baselines, and we're free to set their length
        baseline_runs = np.array([4, 3])
    assert len(baseline_runs) == 2

    solution = create_analytical_solution(
        baseline_runs=baseline_runs,
        sigma=sigma,
        add_baselines=add_baselines,
    )

    # Since we have a sampling frequency of 1 Hz and start counting time from 0,
    # the time of each sample is an integer that is identical to the sample index
    indexes = np.array([int(x) for x in sim.observations[0].get_times()])
    sim.observations[0].sky_signal = solution.sky_signal[indexes].reshape((1, -1))
    sim.observations[0].baseline = solution.baseline_signal[indexes].reshape((1, -1))
    sim.observations[0].white_noise = solution.noise_signal[indexes].reshape((1, -1))

    hpx = Healpix_Base(1, "RING")
    sim.observations[0].pointings = hpx.pix2ang(PIXEL_INDEXES[indexes]).reshape(
        (1, -1, 2)
    )
    sim.observations[0].psi = PSI_ANGLES[indexes].reshape((1, -1))

    return (sim, solution)


def test_map_maker_parts():
    from litebird_sim.mapmaking.common import _normalize_observations_and_pointings
    from litebird_sim.mapmaking.destriper import (
        _store_pixel_idx_and_pol_angle_in_obs,
        _build_mask_detector_split,
        _build_mask_time_split,
        _build_nobs_matrix,
        _sum_components_into_obs,
        _compute_binned_map,
        _compute_baseline_sums,
    )

    if lbs.MPI_COMM_WORLD.size > 2:
        # This test can work only with 1 or 2 MPI processes, no more
        return

    nside = 1
    sim, expected_solution = setup_simulation(sigma=0.1)

    baseline_lengths_list = get_baseline_lengths_list(expected_solution)

    obs_list, ptg_list, psi_list = _normalize_observations_and_pointings(
        obs=sim.observations,
        pointings=None,
    )

    hpx = Healpix_Base(nside=nside, scheme="RING")
    _store_pixel_idx_and_pol_angle_in_obs(
        hpx=hpx,
        obs_list=obs_list,
        ptg_list=ptg_list,
        psi_list=psi_list,
        output_coordinate_system=CoordinateSystem.Ecliptic,
    )

    detector_mask_list = _build_mask_detector_split("full", obs_list)

    time_mask_list = _build_mask_time_split("full", obs_list)

    #################################################
    # Step 1: check that the N_obs matrix is correct

    nobs_matrix_cholesky = _build_nobs_matrix(
        hpx=hpx,
        obs_list=obs_list,
        ptg_list=ptg_list,
        psi_list=psi_list,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
    )

    assert (
        expected_solution.M.shape[0] % 3 == 0
    ), "Matrix M must have size (3n_pix, 3n_pix)"

    number_of_pixels = expected_solution.M.shape[0] // 3

    invpp = nobs_matrix_cholesky.get_invnpp()
    for i in range(number_of_pixels):
        # Cut out the 3×3 submatrix corresponding to the i-th pixel
        cur_M_expected = expected_solution.M[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
        cur_cholesky_expected = np.linalg.cholesky(cur_M_expected)
        # Here it is important that there are no gaps in the array
        # `PIXEL_INDEXES` (see its definition above)
        cur_cholesky_calculated = lower_triangular_to_matrix(
            nobs_matrix_cholesky.nobs_matrix[i]
        )

        np.testing.assert_allclose(
            actual=cur_cholesky_calculated, desired=cur_cholesky_expected
        )

        np.testing.assert_allclose(
            actual=invpp[i],
            desired=np.linalg.inv(cur_M_expected),
        )

    #################################################
    # Step 2: check that binning is correct, i.e., the result of
    # P^t C_w⁻¹ y, see Equations (18), (19), and (20)

    # Make sure that sky_signal += +1.0 * (baseline + white_noise)
    _sum_components_into_obs(
        obs_list=obs_list,
        target="sky_signal",
        other_components=["baseline", "white_noise"],
        factor=+1.0,
    )

    sky_map = np.empty((3, number_of_pixels))
    hit_map = np.empty(number_of_pixels)

    _compute_binned_map(
        obs_list=obs_list,
        output_sky_map=sky_map,
        output_hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        baselines_list=None,
        baseline_lengths_list=baseline_lengths_list,
        component="sky_signal",
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
    )

    # This is going to be a 3N_p vector
    expected_map = (
        np.linalg.inv(expected_solution.M)
        @ np.transpose(expected_solution.P)
        @ expected_solution.invCw
        @ expected_solution.y
    )
    # As sky_map is a (3, 12NSIDE²) vector, we cannot blindly compare it with
    # `expected_map`. Rather, we must transpose it (so that the 3×2 matrix
    # becomes a 2×3 matrix) and flatten it into a linear array
    np.testing.assert_allclose(
        actual=sky_map.transpose().flatten(), desired=expected_map
    )

    #################################################
    # Step 3: check that the F^t C_w⁻¹ Z operator (Eq. 14) works
    # correctly both on baselines (Fa) and on TODs (y)

    # Here we pick some random values for the 1/f baselines
    # The item `full_baselines` contains the baselines from
    # *all* the MPI processes, and it is used to compute the
    # analytical solution, as the analytical matrices are not
    # aware of MPI (their elements are not spread among the
    # MPI processes)
    if MPI_COMM_WORLD.size == 1:
        full_baselines = np.array([[0.0, 1.0]])
        baselines_list = [full_baselines]
    else:
        full_baselines = np.array([[0.0, 10.0]])
        # With 2 MPI processes:
        #     MPI#1: baselines_list=[array([0.])]
        #     MPI#2: baselines_list=[array([10.])]
        baselines_list = [np.array([full_baselines[MPI_COMM_WORLD.rank]])]

    # This array will hold the result
    output_baselines_list = [np.empty_like(x) for x in baselines_list]

    # Recompute the binned map using Fa as the TOD
    # (in step 2, the TOD was just y because `baselines_list`
    # only contained zeroes)
    _compute_binned_map(
        obs_list=obs_list,
        output_sky_map=sky_map,
        output_hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component=None,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
    )

    _compute_baseline_sums(
        obs_list=obs_list,
        sky_map=sky_map,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component="sky_signal",
        output_sums_list=output_baselines_list,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
    )

    expected = (
        np.transpose(expected_solution.F)
        @ expected_solution.invCw
        @ expected_solution.Z
        @ expected_solution.F
        @ full_baselines[0]
    )

    if MPI_COMM_WORLD.size == 2:
        # Just check the baseline that belongs to this MPI process
        np.testing.assert_almost_equal(
            actual=output_baselines_list[0][0], desired=expected[MPI_COMM_WORLD.rank]
        )
    else:
        np.testing.assert_allclose(
            actual=output_baselines_list[0][0],
            desired=expected,
        )

    # Now do the same using `y` instead of `Fa`
    _compute_binned_map(
        obs_list=obs_list,
        output_sky_map=sky_map,
        output_hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        baselines_list=None,
        baseline_lengths_list=baseline_lengths_list,
        component="sky_signal",
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
    )

    _compute_baseline_sums(
        obs_list=obs_list,
        sky_map=sky_map,
        baselines_list=None,
        baseline_lengths_list=baseline_lengths_list,
        component="sky_signal",
        output_sums_list=output_baselines_list,
        dm_list=detector_mask_list,
        tm_list=time_mask_list,
    )

    expected = (
        np.transpose(expected_solution.F)
        @ expected_solution.invCw
        @ expected_solution.Z
        @ expected_solution.y
    )

    if MPI_COMM_WORLD.size == 2:
        # Just check the baseline that belongs to this MPI process
        np.testing.assert_almost_equal(
            actual=output_baselines_list[0][0], desired=expected[MPI_COMM_WORLD.rank]
        )
    else:
        np.testing.assert_allclose(
            actual=output_baselines_list[0][0],
            desired=expected,
        )


def _compare_analytical_vs_estimated_map(
    actual: npt.ArrayLike, desired: npt.ArrayLike, nobs_matrix_cholesky: NobsMatrix
):
    for cur_pix in range(len(actual[0])):
        if nobs_matrix_cholesky.valid_pixel[cur_pix]:
            np.testing.assert_allclose(
                actual=actual[:, cur_pix],
                desired=desired[(3 * cur_pix) : (3 * cur_pix + 3)],
                rtol=1e-5,
            )
        else:
            assert np.isnan(actual[0, cur_pix])
            assert np.isnan(actual[1, cur_pix])
            assert np.isnan(actual[2, cur_pix])


def _make_zero_mean(x: npt.ArrayLike) -> npt.NDArray:
    return x - np.mean(x)


def _test_map_maker(use_destriper: bool, use_preconditioner: bool):
    if not use_destriper:
        assert (
            not use_preconditioner
        ), "Impossible to use a preconditioner with the binner!"

    if lbs.MPI_COMM_WORLD.size > 2:
        # This test can work only with 1 or 2 MPI processes, no more
        return

    # Create some test data *without* 1/f noise
    sim, expected_solution = setup_simulation(sigma=0.1, add_baselines=use_destriper)

    if use_destriper:
        baseline_lengths_list = get_baseline_lengths_list(expected_solution)
    else:
        baseline_lengths_list = None

    original_tod = np.copy(sim.observations[0].sky_signal)
    result = lbs.make_destriped_map(
        nside=1,
        obs=sim.observations,
        pointings=None,
        params=lbs.DestriperParameters(
            output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
            samples_per_baseline=baseline_lengths_list,
            use_preconditioner=use_preconditioner,
        ),
        components=["sky_signal", "baseline", "white_noise"],
    )

    # As the destriper messes up with TODs when there are multiple components,
    # let's check that everything was put in place again
    np.testing.assert_allclose(
        actual=sim.observations[0].sky_signal, desired=original_tod
    )

    if use_destriper:
        for cur_baseline_errors in result.baseline_errors:
            assert np.alltrue(cur_baseline_errors > 0)

        # Check that remove_destriper_baselines_from_tod works. We use a trick here:
        # we create a new null TOD and ask to remove the baselines from it, so that
        # at the end it will contain the baselines unrolled on a TOD with flipped sign.
        for cur_obs in sim.observations:
            cur_obs.unrolled_baselines = np.zeros_like(cur_obs.sky_signal)
        remove_destriper_baselines_from_tod(
            obs_list=sim.observations,
            destriper_result=result,
            component="unrolled_baselines",
        )
        cur_obs.unrolled_baselines *= -1.0

        # Check that the baselines are ok
        expected_baselines = _make_zero_mean(BASELINE_VALUES)
        if MPI_COMM_WORLD.size > 1:
            # We derive the offset to apply to the baselines from F·a,
            # which is *not* divided among the MPI processes
            baseline_offset = np.mean(expected_solution.F @ expected_baselines)
            # We use `result.baselines[0][0][0, :]` because each MPI process has
            # just *one* Observation (first index), *one* baseline (second index),
            # and *one* detector (third index)
            np.testing.assert_almost_equal(
                actual=result.baselines[0][0],
                desired=expected_baselines[MPI_COMM_WORLD.rank] - baseline_offset,
            )
            np.testing.assert_allclose(
                actual=sim.observations[0].unrolled_baselines[0],
                desired=(expected_baselines[MPI_COMM_WORLD.rank] - baseline_offset),
            )
        else:
            np.testing.assert_allclose(
                actual=_make_zero_mean(result.baselines[0][0, :]),
                desired=_make_zero_mean(expected_baselines),
            )
            np.testing.assert_allclose(
                actual=_make_zero_mean(sim.observations[0].unrolled_baselines[0]),
                desired=_make_zero_mean(expected_solution.F @ expected_baselines),
            )

        expected_destriped_map = np.copy(expected_solution.input_maps)
        expected_destriped_map[0::3] -= np.mean(expected_destriped_map[0::3])

        # Check that the binned and destriped maps are ok
        _compare_analytical_vs_estimated_map(
            actual=result.binned_map,
            desired=expected_solution.expected_binned_map,
            nobs_matrix_cholesky=result.nobs_matrix_cholesky,
        )
        # Remember that the F matrix unrolls the baselines into the TOD space
        _compare_analytical_vs_estimated_map(
            actual=result.destriped_map,
            desired=expected_destriped_map,
            nobs_matrix_cholesky=result.nobs_matrix_cholesky,
        )
    else:
        # Check the binned map
        _compare_analytical_vs_estimated_map(
            actual=result.binned_map,
            desired=expected_solution.input_maps,
            nobs_matrix_cholesky=result.nobs_matrix_cholesky,
        )


def test_map_maker_without_destriping():
    _test_map_maker(use_destriper=False, use_preconditioner=False)


def test_map_maker_with_destriping():
    _test_map_maker(use_destriper=True, use_preconditioner=False)


def test_map_maker_with_destriping_and_preconditioner():
    _test_map_maker(use_destriper=True, use_preconditioner=True)


def test_full_destriper(tmp_path):
    # The previous tests were tailored for the case of a TOD
    # with just 7 samples and one detector. Now we turn to a more
    # comprehensive test with *two* detectors

    nside = 32

    sim = lbs.Simulation(
        base_path=tmp_path,
        start_time=0,
        duration_s=astropy.time.TimeDelta(10, format="jd").to("s").value,
        random_seed=12345,
    )

    sim.set_instrument(
        lbs.InstrumentInfo(
            name="Dummy", boresight_rotangle_rad=np.deg2rad(50), hwp_rpm=46.0
        )
    )

    dets = [
        lbs.DetectorInfo(
            sampling_rate_hz=1.0,
            name="A",
            fwhm_arcmin=20.0,
            bandcenter_ghz=140.0,
            bandwidth_ghz=40.0,
            net_ukrts=50.0,
            fknee_mhz=50.0,
            quat=np.array([0.02568196, 0.00506653, 0.0, 0.99965732]),
        ),
        lbs.DetectorInfo(
            sampling_rate_hz=1.0,
            name="B",
            fwhm_arcmin=20.0,
            bandcenter_ghz=140.0,
            bandwidth_ghz=40.0,
            net_ukrts=50.0,
            fknee_mhz=50.0,
            quat=np.array([0.0145773, 0.02174247, -0.70686447, 0.70686447]),
        ),
    ]

    sim.set_scanning_strategy(
        scanning_strategy=lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            precession_rate_hz=1 / 10_020.0,
            spin_rate_hz=1 / 60.0,
        ),
    )

    sim.create_observations(
        detectors=dets,
        num_of_obs_per_detector=sim.mpi_comm.size,
    )

    assert len(sim.observations) == 1

    sim.set_hwp(
        lbs.IdealHWP(
            sim.instrument.hwp_rpm * 2 * np.pi / 60,
        ),  # applies hwp rotation angle to the polarization angle
    )
    sim.compute_pointings()

    mbs_params = lbs.MbsParameters(
        make_cmb=True,
        make_fg=True,
        seed_cmb=1,
        fg_models=[
            "pysm_synch_0",
            "pysm_freefree_1",
            "pysm_dust_0",
        ],  # set the FG models you want
        gaussian_smooth=True,
        bandpass_int=False,
        nside=nside,
        units="K_CMB",
        maps_in_ecliptic=False,
    )

    mbs = lbs.Mbs(
        simulation=sim,
        parameters=mbs_params,
        detector_list=dets,
    )
    maps = mbs.run_all()[0]  # generates the map as a dictionary

    lbs.scan_map_in_observations(
        sim.observations,
        maps=maps,
        input_map_in_galactic=True,
    )

    lbs.add_noise_to_observations(
        obs=sim.observations,
        noise_type="one_over_f",
        scale=1,
        random=sim.random,
    )

    destriper_params_noise = lbs.DestriperParameters(
        output_coordinate_system=lbs.coordinates.CoordinateSystem.Galactic,
        samples_per_baseline=100,  # ν_samp = 1 Hz ⇒ the baseline is 100 s
        iter_max=10,
        threshold=1e-6,
    )

    destriper_result = lbs.make_destriped_map(
        nside=nside, obs=sim.observations, pointings=None, params=destriper_params_noise
    )

    # Check that the destriper converged to some solution
    assert destriper_result.converged

    # Check that all the errors on the baseline values are non-negative
    for cur_baseline_errors in destriper_result.baseline_errors:
        assert np.alltrue(cur_baseline_errors >= 0.0)


def _assert_dataclasses_equal(actual, desired, params_to_check: List[str]) -> None:
    for param in params_to_check:
        actual_value = getattr(actual, param)
        desired_value = getattr(desired, param)
        assert (
            actual_value == desired_value
        ), f"Parameter {param} is different: {actual_value=} ≠ {desired_value=}"


def _test_destriper_results_io(tmp_path, use_destriper: bool):
    ############################################################
    # Create a fake DestriperResults object

    nside = 4
    npix = 12 * (nside**2)

    hit_map = np.arange(npix) + 10000.0
    binned_map = np.random.random((3, npix)) + 20000.0

    # Make the number of baselines and detectors different depending
    # on the MPI rank
    num_of_baselines = 5 + MPI_COMM_WORLD.rank
    num_of_detectors = 1 + MPI_COMM_WORLD.rank

    if use_destriper:
        baselines = [np.random.random((num_of_detectors, num_of_baselines))]
        baselines[0] -= np.mean(baselines[0])  # Make their mean zero
        baseline_errors = [1.5 + np.random.random((num_of_detectors, num_of_baselines))]
        baseline_lengths = [150 + np.arange(num_of_baselines, dtype=int) * 4]
        iter_max = 123
        threshold = 1.2345e-6
        stopping_factor = 9.163e-8
        history_of_stopping_factors = np.array([1.3e-5, 2.4e-6, 3.5e-7, 4.6e-8])
        destriped_map = np.random.random((3, npix)) + 30000.0
        samples_per_baseline = (
            np.arange(num_of_baselines, dtype=int) + MPI_COMM_WORLD.rank * 100
        )
    else:
        baselines = None
        baseline_errors = None
        baseline_lengths = None
        iter_max = None
        threshold = None
        stopping_factor = None
        history_of_stopping_factors = None
        destriped_map = None
        samples_per_baseline = None

    params = lbs.DestriperParameters(
        output_coordinate_system=lbs.CoordinateSystem.Galactic,
        samples_per_baseline=samples_per_baseline,
        iter_max=iter_max,
        threshold=threshold,
        use_preconditioner=True,
    )

    nobs_matrix = np.random.random((npix, 6))
    valid_pixel = np.random.random(npix) > 0.5
    nobs_matrix_cholesky = NobsMatrix(
        nobs_matrix=nobs_matrix,
        valid_pixel=valid_pixel,
        is_cholesky=True,
    )
    desired_results = lbs.DestriperResult(
        nside=nside,
        params=params,
        hit_map=hit_map,
        binned_map=binned_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        coordinate_system=lbs.CoordinateSystem.Galactic,
        baselines=baselines,
        baseline_errors=baseline_errors,
        baseline_lengths=baseline_lengths,
        stopping_factor=stopping_factor,
        history_of_stopping_factors=history_of_stopping_factors,
        destriped_map=destriped_map,
        converged=True,
        components=["a", "bb", "ccc", "dddd", "eeeee"],
        elapsed_time_s=12345.0,
        bytes_in_temporary_buffers=54267,
    )

    ############################################################
    # Save the results
    output_folder = Path(tmp_path) / "destriper"
    save_destriper_results(results=desired_results, output_folder=output_folder)

    ############################################################
    # Load the results
    actual_results = load_destriper_results(output_folder)

    ############################################################
    # Check that what has been loaded matches the input

    np.testing.assert_allclose(
        actual=actual_results.nobs_matrix_cholesky.nobs_matrix,
        desired=desired_results.nobs_matrix_cholesky.nobs_matrix,
    )
    np.testing.assert_allclose(
        actual=actual_results.nobs_matrix_cholesky.valid_pixel,
        desired=desired_results.nobs_matrix_cholesky.valid_pixel,
    )
    assert (
        actual_results.nobs_matrix_cholesky.is_cholesky
        == desired_results.nobs_matrix_cholesky.is_cholesky
    )

    params_to_check = ["output_coordinate_system"]
    if use_destriper:
        # Skip samples_per_baseline, as this is problematic!
        params_to_check += ["iter_max", "threshold", "use_preconditioner"]

    _assert_dataclasses_equal(
        actual=actual_results.params,
        desired=desired_results.params,
        params_to_check=params_to_check,
    )
    params_to_check = [
        "nside",
        "components",
        "coordinate_system",
        "elapsed_time_s",
        "bytes_in_temporary_buffers",
    ]
    if use_destriper:
        params_to_check += ["stopping_factor", "converged"]
    _assert_dataclasses_equal(
        actual=actual_results, desired=desired_results, params_to_check=params_to_check
    )

    np.testing.assert_allclose(
        actual=actual_results.hit_map, desired=desired_results.hit_map
    )
    np.testing.assert_allclose(
        actual=actual_results.binned_map, desired=desired_results.binned_map
    )

    if use_destriper:
        np.testing.assert_allclose(
            actual=actual_results.history_of_stopping_factors,
            desired=desired_results.history_of_stopping_factors,
        )

        assert actual_results.baselines is not None
        for cur_actual, cur_desired in zip(
            actual_results.baselines, desired_results.baselines
        ):
            np.testing.assert_allclose(
                actual=cur_actual,
                desired=cur_desired,
            )

        assert actual_results.baselines is not None
        for cur_actual, cur_desired in zip(
            actual_results.baseline_errors, desired_results.baseline_errors
        ):
            np.testing.assert_allclose(
                actual=cur_actual,
                desired=cur_desired,
            )

        assert actual_results.baseline_lengths is not None
        for cur_actual, cur_desired in zip(
            actual_results.baseline_lengths, desired_results.baseline_lengths
        ):
            assert np.alltrue(cur_actual == cur_desired)


def test_destriper_io(tmp_path):
    _test_destriper_results_io(tmp_path=tmp_path, use_destriper=True)


def test_destriper_io_without_destriper(tmp_path):
    _test_destriper_results_io(tmp_path=tmp_path, use_destriper=False)
