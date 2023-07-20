# -*- encoding: utf-8 -*-

# In this file, we test our own implementation of the destriping algorithm.
# We consider a simple TOD with very few samples, so that it's possible to
# write the matrices used to express the destriping problem in full form.
# After having created the matrices in memory and having inverted them,
# we check that the analytical solution is close enough to the output
# of the toast_destriper.

from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np
from ducc0.healpix import Healpix_Base

import litebird_sim as lbs
from litebird_sim import CoordinateSystem, MPI_COMM_WORLD, MPI_ENABLED

if MPI_ENABLED:
    import mpi4py.MPI


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
# destriping equation is going to retrieve different numbers
# (with zero mean)
BASELINE_VALUES = np.array([10.0, 12.0])


# This class is used to hold information about those matrices that are
# too complex to write down explicitly and need to be calculated.
@dataclass
class AnalyticSolution:
    num_of_samples: int = NUM_OF_SAMPLES
    num_of_baselines: int = NUM_OF_BASELINES
    num_of_pixels: int = NUM_OF_PIXELS
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
) -> AnalyticSolution:
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

    return AnalyticSolution(
        num_of_samples=NUM_OF_SAMPLES,
        num_of_baselines=NUM_OF_BASELINES,
        num_of_pixels=NUM_OF_PIXELS,
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
        estimated_a=estimated_a,
        estimated_maps=estimated_maps,
    )


def setup_simulation(
    sigma: float = 0.1, add_baselines: bool = True
) -> Tuple[lbs.Simulation, AnalyticSolution]:
    """Create a Simulation object and a AnalyticSolution object that match

    The Simulation object contains the same data as in AnalyticSolution. It
    is meant to be used with the destriper to check that the result matches
    the analytic solution."""

    # The TOD is too short (7 samples) to be meaningful with more than 2 processes
    assert lbs.MPI_COMM_WORLD.size <= 2

    sim = lbs.Simulation(
        start_time=0.0,
        duration_s=NUM_OF_SAMPLES,
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

    assert len(sim.observations) > 0, f"no observations for process {sim.mpi_comm.size}"

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

    if MPI_COMM_WORLD.size == 2:
        baseline_lengths_list = [
            np.array([expected_solution.baseline_runs[MPI_COMM_WORLD.rank]], dtype=int)
        ]
    elif MPI_COMM_WORLD.size == 1:
        baseline_lengths_list = [np.array(expected_solution.baseline_runs)]
    else:
        assert False, "This should not happen! Only up to 2 MPI ranks are allowed here"

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

    #################################################
    # Step 1: check that the N_obs matrix is correct

    nobs_matrix_cholesky = _build_nobs_matrix(
        hpx=hpx,
        obs_list=obs_list,
        ptg_list=ptg_list,
        psi_list=psi_list,
    )

    assert (
        expected_solution.M.shape[0] % 3 == 0
    ), "Matrix M must have size (3n_pix, 3n_pix)"

    number_of_pixels = expected_solution.M.shape[0] // 3

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

    #################################################
    # Step 2: check that binning is correct (the result of P^t C_w⁻¹ y,
    # see Equations (18), (19), and (20)

    # Make sure that sky_signal += +1.0 * (baseline + white_noise)
    _sum_components_into_obs(
        obs_list=obs_list,
        target="sky_signal",
        other_components=["baseline", "white_noise"],
        factor=+1.0,
    )

    sky_map = np.empty((3, number_of_pixels))
    hit_map = np.empty(number_of_pixels)

    # For the moment, we just set the baselines to zero. We'll check
    # a more complex case in step 3
    baselines_list = [
        np.zeros(len(cur_baseline_length))
        for cur_baseline_length in baseline_lengths_list
    ]

    _compute_binned_map(
        obs_list=obs_list,
        sky_map=sky_map,
        hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component="sky_signal",
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
    # Step 3: check that the F^t C_w⁻¹ Z operator works
    # correctly (F^t C_w⁻¹ Z (Fa - y) = 0, which is
    # a different way to write Eq. 14)

    baselines_list = [
        np.arange(len(cur_baseline_length), dtype=np.float64) + 10 * MPI_COMM_WORLD.rank
        for cur_baseline_length in baseline_lengths_list
    ]

    # Recompute the binned map using (Fa - y) as the TOD
    # (in step 2, the TOD was just y because `baselines_list`
    # only contained zeroes)
    _compute_binned_map(
        obs_list=obs_list,
        sky_map=sky_map,
        hit_map=hit_map,
        nobs_matrix_cholesky=nobs_matrix_cholesky,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component="sky_signal",
    )

    output_baselines_list = [np.empty_like(x) for x in baselines_list]
    _compute_baseline_sums(
        obs_list=obs_list,
        sky_map=sky_map,
        baselines_list=baselines_list,
        baseline_lengths_list=baseline_lengths_list,
        component="sky_signal",
        output_sums_list=output_baselines_list,
    )

    expected = (
        np.transpose(expected_solution.F)
        @ expected_solution.invCw
        @ expected_solution.Z
        @ (expected_solution.F @ baselines_list[0] - expected_solution.y)
    )
    np.testing.assert_allclose(
        actual=output_baselines_list[0],
        desired=expected,
    )


def _test_map_maker(use_destriper: bool):
    if lbs.MPI_COMM_WORLD.size > 2:
        # This test can work only with 1 or 2 MPI processes, no more
        return

    # Create some test data *without* 1/f noise
    sim, expected_solution = setup_simulation(sigma=0.1, add_baselines=use_destriper)

    if use_destriper:
        samples_per_baseline = np.unique(
            expected_solution.baseline_indexes, return_counts=True
        )[1]
    else:
        samples_per_baseline = None

    result = lbs.make_destriped_map(
        obs=sim.observations,
        pointings=None,
        params=lbs.DestriperParameters(
            nside=1,
            output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
            samples_per_baseline=samples_per_baseline,
        ),
        components=["sky_signal", "baseline", "white_noise"],
    )

    if use_destriper:
        pass
    else:
        # Just check the binned map
        for cur_pix in range(len(result.hit_map)):
            if result.nobs_matrix_cholesky.valid_pixel[cur_pix]:
                np.testing.assert_allclose(
                    actual=result.binned_map[:, cur_pix],
                    desired=expected_solution.input_maps[
                        (3 * cur_pix) : (3 * cur_pix + 3)
                    ],
                    rtol=1e-5,
                )
            else:
                assert np.isnan(result.binned_map[0, cur_pix])
                assert np.isnan(result.binned_map[1, cur_pix])
                assert np.isnan(result.binned_map[2, cur_pix])


def test_map_maker_without_destriping():
    _test_map_maker(use_destriper=False)


def _test_map_maker_with_destriping():
    _test_map_maker(use_destriper=True)
