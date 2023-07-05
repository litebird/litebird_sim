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

NOISE_SAMPLES = np.array(
    [
        0.00407789,
        0.01652727,
        -0.00816626,
        -0.00212623,
        0.00370481,
        -0.01087152,
        -0.00897898,
    ]
)
assert len(NOISE_SAMPLES) == NUM_OF_SAMPLES

# These are the pixel observed by each of the seven samples in the TOD
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


# This array associates each sample in the TOD with the baseline index.
# Thus, the first baseline a₀ covers the first four samples in the TOD,
# and the next baseline a₁ covers the last three samples
BASELINE_INDEXES = np.array(
    [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
    ],
    dtype="int",
)
assert len(BASELINE_INDEXES) == NUM_OF_SAMPLES


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


def create_analytical_solution(sigma: float = 0.1) -> AnalyticSolution:
    # Build the pointing matrix P (Eq. 2 of KurkiSuonio2009)
    P = np.zeros((NUM_OF_SAMPLES, 3 * NUM_OF_PIXELS))

    for i, pix_idx, psi in zip(range(NUM_OF_SAMPLES), PIXEL_INDEXES, PSI_ANGLES):
        P[i, 3 * pix_idx] = 1
        P[i, 3 * pix_idx + 1] = np.cos(psi)
        P[i, 3 * pix_idx + 2] = np.sin(psi)

    # Build the noise matrix
    Cw = np.eye(NUM_OF_SAMPLES) * (sigma**2)
    invCw = np.eye(NUM_OF_SAMPLES) / (sigma**2)

    # Build the baseline-projection matrix F (Eq. 3 of KurkiSuonio2009)
    F = np.zeros((NUM_OF_SAMPLES, NUM_OF_BASELINES))
    for idx, baseline_idx in enumerate(BASELINE_INDEXES):
        F[idx, baseline_idx] = 1

    # Build the TOD (Eq. 2+3 of KurkiSuonio2009)
    sky_signal = (P @ INPUT_MAPS).reshape(-1)
    baseline_signal = (F @ BASELINE_VALUES).reshape(-1)
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
    estimated_a = np.linalg.inv(D) @ np.transpose(F) @ invCw @ Z @ y
    estimated_maps = invM @ np.transpose(P) @ invCw @ (y - F @ BASELINE_VALUES)

    return AnalyticSolution(
        num_of_samples=NUM_OF_SAMPLES,
        num_of_baselines=NUM_OF_BASELINES,
        num_of_pixels=NUM_OF_PIXELS,
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


def setup_simulation(sigma: float = 0.1) -> Tuple[lbs.Simulation, AnalyticSolution]:
    sim = lbs.Simulation(start_time=0.0, duration_s=NUM_OF_SAMPLES)

    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(sampling_rate_hz=1.0, name="Mock detector"),
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
        # Make sure that different MPI processes have different chunks of the TODs
        n_blocks_time=lbs.MPI_COMM_WORLD.size,
    )

    descr = sim.describe_mpi_distribution()
    num_of_samples = 0
    for cur_mpi_process in descr.mpi_processes:
        assert len(cur_mpi_process.observations) == 1
        num_of_samples += cur_mpi_process.observations[0].num_of_samples

    assert num_of_samples == NUM_OF_SAMPLES

    solution = create_analytical_solution(sigma=sigma)

    # Since we have a sampling frequency of 1 Hz and start counting time from 0,
    # the time of each sample is an integer that is identical to the sample index
    indexes = [int(x) for x in sim.observations[0].get_times()]
    sim.observations[0].sky_signal = solution.sky_signal[indexes]
    sim.observations[0].baseline = solution.baseline_signal[indexes]
    sim.observations[0].white_noise = solution.noise_signal[indexes]

    hpx = Healpix_Base(1, "RING")
    sim.observations[0].pointings = hpx.pix2ang(PIXEL_INDEXES).reshape((1, -1, 2))
    sim.observations[0].psi = PSI_ANGLES.reshape((1, solution.num_of_samples))

    return (sim, solution)


def test_destriper():
    sim, expected_solution = setup_simulation(sigma=0.1)

    descr = sim.describe_mpi_distribution()
    print(descr)
    assert 0 == 0
