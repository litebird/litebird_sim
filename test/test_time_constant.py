import numpy as np
import pytest

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


def _observation(taus):
    sim = lbs.Simulation(start_time=0.0, duration_s=60.0, random_seed=12345)
    detectors = [
        lbs.DetectorInfo(
            name=f"det{i}", sampling_rate_hz=SAMPLING_RATE_HZ, time_constant_ms=tau
        )
        for i, tau in enumerate(taus)
    ]
    (obs,) = sim.create_observations(detectors=detectors)
    obs.tod[:] = np.random.default_rng(1).normal(size=obs.tod.shape)
    return obs


def test_time_constant_missing_in_detectors():
    # A τ that is NaN in some detector raises before any TOD is modified
    obs = _observation([3.0, np.nan])
    original = obs.tod.copy()
    with pytest.raises(ValueError, match="time_constant_ms"):
        lbs.convolve_time_constant_in_observations(obs)
    np.testing.assert_array_equal(obs.tod, original)

    # Passing τ explicitly does not need the field
    lbs.convolve_time_constant_in_observations(obs, time_constant_ms=3.0)
    assert np.all(np.isfinite(obs.tod))


def test_time_constant_array_is_per_global_detector():
    obs = _observation([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="3 detectors"):
        lbs.convolve_time_constant_in_observations(obs, time_constant_ms=[1.0, 2.0])

    # Each detector takes the value at its global index (det_idx), as an MPI
    # process holding only some detectors would
    expected = obs.tod.copy()
    lbs.convolve_time_constant(expected, [1.0, 2.0, 3.0], SAMPLING_RATE_HZ)
    lbs.convolve_time_constant_in_observations(
        obs, time_constant_ms=np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(obs.tod, expected, atol=1e-6)


def test_time_constant_flags_the_edges():
    # The samples where the mirror image of the timeline enters are flagged,
    # and the error of the filter is small in the others
    freq_hz = 3.07
    obs = _observation([0.0, 5.0])
    time_s = np.arange(obs.tod.shape[1]) / SAMPLING_RATE_HZ
    obs.tod[:] = np.cos(2 * np.pi * freq_hz * time_s)

    lbs.convolve_time_constant_in_observations(obs)

    flagged = (obs.local_flags & lbs.FLAG_TIME_CONSTANT) != 0
    assert not flagged[0].any()  # τ = 0 does nothing
    length = np.flatnonzero(~flagged[1])[0]
    assert 0 < length < 50
    assert flagged[1, -length:].all() and not flagged[1, length:-length].any()

    x = 2 * np.pi * freq_hz * 5e-3
    expected = np.cos(2 * np.pi * freq_hz * time_s - np.arctan(x)) / np.sqrt(1 + x**2)
    error = np.abs(obs.tod[1] - expected)
    assert error[flagged[1]].max() > 1e-2
    assert error[~flagged[1]].max() < 1e-3


def test_time_constant_flags_around_flagged_samples():
    obs = _observation([5.0])
    lbs.get_local_flags(obs)[0, 300] = lbs.FLAG_PLANET
    lbs.get_global_flags(obs)[600] = lbs.FLAG_PLANET

    lbs.convolve_time_constant_in_observations(obs)
    flags = obs.local_flags[0].copy()
    flagged = (flags & lbs.FLAG_TIME_CONSTANT) != 0
    length = np.flatnonzero(~flagged)[0]
    for center in (300, 600):
        assert flagged[center - length : center + length + 1].all()
        assert not flagged[center - length - 1] and not flagged[center + length + 1]
    assert not flagged[450]
    assert flags[300] & lbs.FLAG_PLANET  # The other bits are kept

    # The bit set by the convolution is not extended by the deconvolution
    lbs.deconvolve_time_constant_in_observations(obs)
    deconv_flagged = (obs.local_flags[0] & lbs.FLAG_TIME_CONSTANT) != 0
    deconv_length = np.flatnonzero(~deconv_flagged)[0]
    assert not deconv_flagged[300 + max(length, deconv_length) + 1]


def test_time_constant_without_flags():
    obs = _observation([5.0])
    lbs.convolve_time_constant_in_observations(obs, bit=0)
    assert obs.local_flags is None

    # τ = 0 does not allocate the flags either
    obs = _observation([0.0])
    lbs.convolve_time_constant_in_observations(obs)
    assert obs.local_flags is None
