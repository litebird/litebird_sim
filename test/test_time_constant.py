import numpy as np

import litebird_sim as lbs


SAMPLING_RATE_HZ = 19.0
NUM_OF_SAMPLES = 19 * 600


def test_time_constant_on_a_sinusoid():
    # A one-pole filter delays a sinusoid by atan(2π f τ) and reduces its
    # amplitude by 1 / sqrt(1 + (2π f τ)²)
    freq_hz = 3.07  # ~4 f_HWP for the LFT
    tau_ms = np.array([0.0, 1.0, 5.0])
    time_s = np.arange(NUM_OF_SAMPLES) / SAMPLING_RATE_HZ
    tod = np.tile(np.cos(2 * np.pi * freq_hz * time_s), (len(tau_ms), 1))

    lbs.convolve_time_constant(tod, tau_ms, SAMPLING_RATE_HZ)

    x = 2 * np.pi * freq_hz * tau_ms * 1e-3
    expected = np.cos(2 * np.pi * freq_hz * time_s - np.arctan(x)[:, None]) / np.sqrt(
        1 + x[:, None] ** 2
    )
    # Skip the edges, where the mirror image of the timeline enters
    middle = slice(1000, -1000)
    np.testing.assert_allclose(tod[:, middle], expected[:, middle], atol=1e-4)


def test_time_constant_roundtrip():
    # A smooth signal (below 4 Hz, like the sky modulated by the HWP) is
    # recovered away from the edges, where the response of the filter decays as 1/n
    time_s = np.arange(NUM_OF_SAMPLES) / SAMPLING_RATE_HZ
    original = np.array(
        [
            np.cos(2 * np.pi * 3.07 * time_s) + 0.5 * np.sin(2 * np.pi * 0.11 * time_s),
            np.sin(2 * np.pi * 1.3 * time_s),
        ],
        dtype=np.float32,
    )
    tod = original.copy()

    lbs.convolve_time_constant(tod, 5.0, SAMPLING_RATE_HZ)
    assert not np.allclose(tod, original, atol=1e-2)

    lbs.deconvolve_time_constant(tod, 5.0, SAMPLING_RATE_HZ)
    middle = slice(100, -100)
    np.testing.assert_allclose(tod[:, middle], original[:, middle], atol=1e-5)


def test_time_constant_in_observations():
    sim = lbs.Simulation(start_time=0.0, duration_s=600.0, random_seed=12345)
    detectors = [
        lbs.DetectorInfo(
            name=f"det{i}", sampling_rate_hz=SAMPLING_RATE_HZ, time_constant_ms=tau
        )
        for i, tau in enumerate([0.0, 3.0])
    ]
    (obs,) = sim.create_observations(detectors=detectors)
    rng = np.random.default_rng(12345)
    obs.tod[:] = rng.normal(size=obs.tod.shape)
    original = obs.tod.copy()

    # The time constants are taken from DetectorInfo: τ = 0 leaves the TOD unchanged
    sim.convolve_time_constant()
    np.testing.assert_array_equal(obs.tod[0], original[0])

    expected = original.copy()
    lbs.convolve_time_constant(expected, [0.0, 3.0], SAMPLING_RATE_HZ)
    np.testing.assert_allclose(obs.tod, expected, atol=1e-6)

    # Deconvolving with the right τ gives the same result as the low-level
    # function, with the wrong τ it leaves a residual
    lbs.deconvolve_time_constant(expected, [0.0, 3.0], SAMPLING_RATE_HZ)
    wrong = obs.tod.copy()
    lbs.deconvolve_time_constant(wrong, [0.0, 3.3], SAMPLING_RATE_HZ)
    middle = slice(100, -100)
    assert not np.allclose(wrong[1, middle], original[1, middle], atol=1e-3)

    sim.deconvolve_time_constant()
    np.testing.assert_allclose(obs.tod, expected, atol=1e-6)
    np.testing.assert_allclose(obs.tod[:, middle], original[:, middle], atol=1e-2)
