# -*- encoding: utf-8 -*-

import numpy as np
import pytest
import scipy.signal
import litebird_sim as lbs


def test_add_noise_to_observations():
    """Basic smoke test: ensures the function runs without crashing and shapes are correct."""
    start_time = 0
    time_span_s = 10
    sampling_hz = 1

    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=None
    )

    det1 = lbs.DetectorInfo(
        name="Boresight_detector_A",
        sampling_rate_hz=sampling_hz,
        net_ukrts=1.0,
        fknee_mhz=1e3,
    )

    det2 = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        net_ukrts=10.0,
        fknee_mhz=2e3,
    )

    sim.init_rng_hierarchy(random_seed=12_345)
    sim.create_observations(detectors=[det1, det2])

    # Test default engine (FFT) with White noise
    lbs.noise.add_noise_to_observations(
        sim.observations, "white", dets_random=sim.dets_random
    )

    assert len(sim.observations) == 1
    tod = sim.observations[0].tod
    assert tod.shape == (2, 10)

    # Ensure noise was added (not zeros)
    assert np.std(tod) > 0


def test_add_noise_to_observations_in_other_field():
    """Tests writing noise to a specific component field."""
    start_time = 0
    time_span_s = 10
    sampling_hz = 1

    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=None
    )

    det1 = lbs.DetectorInfo(
        name="Boresight_detector_A",
        sampling_rate_hz=sampling_hz,
        net_ukrts=1.0,
        fknee_mhz=1e3,
    )

    det2 = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        net_ukrts=10.0,
        fknee_mhz=2e3,
    )

    sim.create_observations(detectors=[det1, det2])

    for cur_obs in sim.observations:
        cur_obs.noise_tod = np.zeros_like(cur_obs.tod)

    sim.init_rng_hierarchy(random_seed=12_345)

    # Test broadcasting bug fix: passing arrays for NET/fknee to multi-detector obs
    lbs.noise.add_noise_to_observations(
        sim.observations,
        "one_over_f",
        dets_random=sim.dets_random,
        component="noise_tod",
        engine="fft",
        model="standard",
    )

    noise_out = sim.observations[0].noise_tod
    assert noise_out.shape == (2, 10)
    assert np.any(noise_out != 0)


def test_multi_detector_scaling():
    """
    Verifies that different detectors get different noise levels based on their NET.
    Tests the broadcasting logic fixed in noise.py.
    """
    fs = 100.0
    sim = lbs.Simulation(
        start_time=0, duration_s=1000, random_seed=None
    )  # Long enough for stats

    # Det 1: Low Noise
    det1 = lbs.DetectorInfo(name="A", sampling_rate_hz=fs, net_ukrts=10.0, fknee_mhz=0)
    # Det 2: High Noise (10x higher NET -> 10x higher sigma)
    det2 = lbs.DetectorInfo(name="B", sampling_rate_hz=fs, net_ukrts=100.0, fknee_mhz=0)

    sim.create_observations([det1, det2])
    sim.init_rng_hierarchy(42)

    lbs.noise.add_noise_to_observations(
        sim.observations, "white", dets_random=sim.dets_random
    )

    tod = sim.observations[0].tod
    std_A = np.std(tod[0])
    std_B = np.std(tod[1])

    # Check ratio is approx 10
    ratio = std_B / std_A
    assert np.isclose(ratio, 10.0, rtol=0.05)


def test_ducc_engine_constraints():
    """Verifies that DUCC engine raises error for unsupported models."""
    sim = lbs.Simulation(start_time=0, duration_s=1, random_seed=None)
    det = lbs.DetectorInfo(name="A", sampling_rate_hz=10, net_ukrts=1, fknee_mhz=100)
    sim.create_observations([det])
    sim.init_rng_hierarchy(42)

    with pytest.raises(ValueError, match="DUCC engine only supports"):
        lbs.noise.add_noise_to_observations(
            sim.observations,
            "one_over_f",
            dets_random=sim.dets_random,
            engine="ducc",
            model="standard",
        )


@pytest.mark.parametrize(
    "engine, model", [("fft", "standard"), ("fft", "keshner"), ("ducc", "keshner")]
)
def test_noise_psd_correctness(engine, model):
    """
    Scientific validation: checks that generated noise matches theoretical PSD.
    Tests FFT-Standard, FFT-Keshner, and DUCC-Keshner.
    """
    fs = 20.0
    nsamp = 2**18  # Sufficient for spectral resolution
    net = 50.0
    fknee_mhz = 100.0
    fmin_hz = 0.01
    alpha = 1.0  # Test 1/f

    # Create manual TOD array to avoid simulation overhead
    tod = np.zeros((1, nsamp))
    rng = np.random.Generator(np.random.PCG64(12345))

    # Add noise
    lbs.noise.add_noise(
        tod,
        "one_over_f",
        fs,
        net_ukrts=net,
        fknee_mhz=fknee_mhz,
        fmin_hz=fmin_hz,
        alpha=alpha,
        dets_random=[rng],
        engine=engine,
        model=model,
    )

    # Compute PSD
    f, psd_sim = scipy.signal.welch(tod[0], fs=fs, nperseg=2**16)

    # Compute Theory
    white_level = 2.0 * (net / 1e6) ** 2  # One-sided correction
    fknee_hz = fknee_mhz / 1000.0

    with np.errstate(divide="ignore"):
        if model == "standard":
            # (f^a + k^a) / (f^a + m^a)
            num = f**alpha + fknee_hz**alpha
            den = f**alpha + fmin_hz**alpha
            psd_theory = white_level * (num / den)
        elif model == "keshner":
            # ((f^2 + k^2) / (f^2 + m^2))^(a/2)
            num = f**2 + fknee_hz**2
            den = f**2 + fmin_hz**2
            psd_theory = white_level * (num / den) ** (alpha / 2.0)

    # Mask DC and very low frequencies where Welch is noisy
    mask = (f > fmin_hz * 5) & (f < fs / 2)

    # Compare
    # Note: 1/f noise is stochastic, so we need a generous tolerance
    # or averaging (which we skip here for speed). 25% error on single realization is acceptable check.
    mean_ratio = np.mean(psd_sim[mask] / psd_theory[mask])

    assert 0.75 < mean_ratio < 1.25, f"PSD mismatch for {engine}/{model}"
