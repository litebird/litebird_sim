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
        model="toast",
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
            model="toast",
        )


@pytest.mark.parametrize(
    "engine, model", [("fft", "toast"), ("fft", "keshner"), ("ducc", "keshner")]
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
        if model == "toast":
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


# ---------------------------------------------------------------------------
# Correlated noise tests
# ---------------------------------------------------------------------------


def _make_sim(n_dets, duration_s=10, sampling_hz=10.0, net=50.0, seed=42):
    """Helper: build a minimal Simulation with `n_dets` identical detectors."""
    sim = lbs.Simulation(start_time=0, duration_s=duration_s, random_seed=None)
    dets = [
        lbs.DetectorInfo(
            name=f"det_{i}",
            sampling_rate_hz=sampling_hz,
            net_ukrts=net,
            fknee_mhz=100.0,
            fmin_hz=0.01,
            alpha=1.0,
        )
        for i in range(n_dets)
    ]
    sim.init_rng_hierarchy(seed)
    sim.create_observations(detectors=dets)
    return sim


# --- helpers ---


def test_normalize_rho_scalar():
    rho = lbs.noise._normalize_rho(0.5, 4)
    assert rho.shape == (4,)
    assert np.all(rho == 0.5)


def test_normalize_rho_array():
    rho = lbs.noise._normalize_rho([0.1, 0.2, 0.3], 3)
    assert rho.shape == (3,)
    np.testing.assert_array_equal(rho, [0.1, 0.2, 0.3])


def test_normalize_rho_out_of_range():
    with pytest.raises(ValueError, match="rho values must be in"):
        lbs.noise._normalize_rho(1.5, 2)

    with pytest.raises(ValueError, match="rho values must be in"):
        lbs.noise._normalize_rho(-0.1, 2)


def test_normalize_rho_wrong_length():
    with pytest.raises(ValueError):
        lbs.noise._normalize_rho([0.1, 0.2], 3)


def test_validate_grouping_ok():
    g = lbs.noise._validate_grouping(4, [0, 0, 1, 1])
    assert g.dtype == int
    np.testing.assert_array_equal(g, [0, 0, 1, 1])


def test_validate_grouping_wrong_length():
    with pytest.raises(ValueError):
        lbs.noise._validate_grouping(4, [0, 1])


def test_build_detector_groups_none():
    sim = _make_sim(3)
    obs = sim.observations[0]
    g = lbs.noise._build_detector_groups(obs, None)
    np.testing.assert_array_equal(g, [0, 0, 0])


def test_build_detector_groups_explicit_array():
    sim = _make_sim(4)
    obs = sim.observations[0]
    labels = [0, 0, 1, 1]
    g = lbs.noise._build_detector_groups(obs, labels)
    np.testing.assert_array_equal(g, labels)


# --- common-mode model ---


def test_correlated_noise_smoke():
    """add_correlated_noise runs without error, modifies TOD in-place."""
    sim = _make_sim(3)
    obs = sim.observations[0]
    tod_orig = obs.tod.copy()

    lbs.noise.add_correlated_noise(
        tod=obs.tod,
        sampling_rate_hz=obs.sampling_rate_hz,
        net_ukrts=obs.net_ukrts,
        fknee_mhz=obs.fknee_mhz,
        fmin_hz=obs.fmin_hz,
        alpha=obs.alpha,
        dets_random=sim.dets_random,
        groups=[0, 0, 0],
    )

    assert np.any(obs.tod != tod_orig), "TOD was not modified"


def test_correlated_noise_reproducible():
    """Same seed => identical TOD."""
    sim1 = _make_sim(2, seed=7)
    sim2 = _make_sim(2, seed=7)

    for sim in (sim1, sim2):
        obs = sim.observations[0]
        lbs.noise.add_correlated_noise(
            tod=obs.tod,
            sampling_rate_hz=obs.sampling_rate_hz,
            net_ukrts=obs.net_ukrts,
            fknee_mhz=obs.fknee_mhz,
            fmin_hz=obs.fmin_hz,
            alpha=obs.alpha,
            dets_random=sim.dets_random,
            groups=[0, 0],
            rho=0.4,
        )

    np.testing.assert_array_equal(sim1.observations[0].tod, sim2.observations[0].tod)


def test_correlated_noise_rho_zero_uncorrelated():
    """rho=0 => detectors share no common mode => correlation should be near zero."""
    n_det = 4
    sim = _make_sim(n_det, duration_s=10000, sampling_hz=10.0, seed=99)
    obs = sim.observations[0]

    lbs.noise.add_correlated_noise(
        tod=obs.tod,
        sampling_rate_hz=obs.sampling_rate_hz,
        net_ukrts=obs.net_ukrts,
        fknee_mhz=obs.fknee_mhz,
        fmin_hz=obs.fmin_hz,
        alpha=obs.alpha,
        dets_random=sim.dets_random,
        groups=np.zeros(n_det, dtype=int),
        rho=0.0,
    )

    # Pearson correlation between any two detectors should be small
    corr = np.corrcoef(obs.tod)
    off_diag = corr[~np.eye(n_det, dtype=bool)]
    assert np.all(np.abs(off_diag) < 0.1), (
        f"Expected near-zero cross-correlation for rho=0, got max={np.max(np.abs(off_diag)):.3f}"
    )


def test_correlated_noise_rho_one_identical():
    """rho=1 => all detectors in the same group are identical."""
    sim = _make_sim(3, duration_s=100, seed=5)
    obs = sim.observations[0]

    lbs.noise.add_correlated_noise(
        tod=obs.tod,
        sampling_rate_hz=obs.sampling_rate_hz,
        net_ukrts=obs.net_ukrts,
        fknee_mhz=obs.fknee_mhz,
        fmin_hz=obs.fmin_hz,
        alpha=obs.alpha,
        dets_random=sim.dets_random,
        groups=[0, 0, 0],
        rho=1.0,
    )

    np.testing.assert_allclose(obs.tod[0], obs.tod[1], rtol=1e-6)
    np.testing.assert_allclose(obs.tod[0], obs.tod[2], rtol=1e-6)


def test_correlated_noise_two_groups_independent():
    """Detectors in different groups share no common mode."""
    n_det = 4
    sim = _make_sim(n_det, duration_s=10000, sampling_hz=10.0, seed=13)
    obs = sim.observations[0]

    lbs.noise.add_correlated_noise(
        tod=obs.tod,
        sampling_rate_hz=obs.sampling_rate_hz,
        net_ukrts=obs.net_ukrts,
        fknee_mhz=obs.fknee_mhz,
        fmin_hz=obs.fmin_hz,
        alpha=obs.alpha,
        dets_random=sim.dets_random,
        groups=[0, 0, 1, 1],
        rho=1.0,  # within each group, dets are identical
    )

    # Within-group: identical
    np.testing.assert_allclose(obs.tod[0], obs.tod[1], rtol=1e-6)
    np.testing.assert_allclose(obs.tod[2], obs.tod[3], rtol=1e-6)

    # Cross-group: not identical (independent streams)
    assert not np.allclose(obs.tod[0], obs.tod[2])


def test_correlated_noise_white_common_mode():
    """common_mode_type='white' runs without error and produces non-zero output."""
    sim = _make_sim(2, seed=3)
    obs = sim.observations[0]
    tod_before = obs.tod.copy()

    lbs.noise.add_correlated_noise(
        tod=obs.tod,
        sampling_rate_hz=obs.sampling_rate_hz,
        net_ukrts=obs.net_ukrts,
        fknee_mhz=obs.fknee_mhz,
        fmin_hz=obs.fmin_hz,
        alpha=obs.alpha,
        dets_random=sim.dets_random,
        groups=[0, 0],
        rho=0.5,
        common_mode_type="white",
    )

    assert np.any(obs.tod != tod_before)


def test_correlated_noise_invalid_common_mode_type():
    sim = _make_sim(2)
    obs = sim.observations[0]
    with pytest.raises(ValueError, match="Unknown common_mode_type"):
        lbs.noise.add_correlated_noise(
            tod=obs.tod,
            sampling_rate_hz=obs.sampling_rate_hz,
            net_ukrts=obs.net_ukrts,
            fknee_mhz=obs.fknee_mhz,
            fmin_hz=obs.fmin_hz,
            alpha=obs.alpha,
            dets_random=sim.dets_random,
            groups=[0, 0],
            common_mode_type="bad_type",
        )


def test_correlated_noise_missing_groups_and_corr_matrix():
    sim = _make_sim(2)
    obs = sim.observations[0]
    with pytest.raises(ValueError, match="Either 'corr_matrix' or 'groups'"):
        lbs.noise.add_correlated_noise(
            tod=obs.tod,
            sampling_rate_hz=obs.sampling_rate_hz,
            net_ukrts=obs.net_ukrts,
            fknee_mhz=obs.fknee_mhz,
            fmin_hz=obs.fmin_hz,
            alpha=obs.alpha,
            dets_random=sim.dets_random,
        )


# --- Cholesky / corr_matrix model ---


def test_correlated_noise_cholesky_smoke():
    """add_correlated_noise with corr_matrix runs and modifies TOD."""
    n = 3
    R = np.full((n, n), 0.5)
    np.fill_diagonal(R, 1.0)

    sim = _make_sim(n, seed=1)
    obs = sim.observations[0]
    tod_before = obs.tod.copy()

    lbs.noise.add_correlated_noise(
        tod=obs.tod,
        sampling_rate_hz=obs.sampling_rate_hz,
        net_ukrts=obs.net_ukrts,
        fknee_mhz=obs.fknee_mhz,
        fmin_hz=obs.fmin_hz,
        alpha=obs.alpha,
        dets_random=sim.dets_random,
        corr_matrix=R,
    )

    assert np.any(obs.tod != tod_before)


def test_correlated_noise_cholesky_identity_uncorrelated():
    """corr_matrix=I => detectors are uncorrelated (same as independent 1/f)."""
    n_det = 4
    sim = _make_sim(n_det, duration_s=10000, sampling_hz=10.0, seed=77)
    obs = sim.observations[0]

    lbs.noise.add_correlated_noise(
        tod=obs.tod,
        sampling_rate_hz=obs.sampling_rate_hz,
        net_ukrts=obs.net_ukrts,
        fknee_mhz=obs.fknee_mhz,
        fmin_hz=obs.fmin_hz,
        alpha=obs.alpha,
        dets_random=sim.dets_random,
        corr_matrix=np.eye(n_det),
    )

    corr = np.corrcoef(obs.tod)
    off_diag = corr[~np.eye(n_det, dtype=bool)]
    assert np.all(np.abs(off_diag) < 0.15), (
        f"Expected near-zero cross-correlation for I matrix, "
        f"got max={np.max(np.abs(off_diag)):.3f}"
    )


def test_correlated_noise_cholesky_full_correlation():
    """corr_matrix of all-ones => all detectors identical (up to sigma scaling)."""
    n = 3
    # All-ones matrix is rank-1: L = [[1,0,0],[1,0,0],[1,0,0]]
    R = np.ones((n, n))
    # Not strictly PD; regularise slightly
    R += 1e-10 * np.eye(n)

    sim = _make_sim(n, duration_s=100, seed=55)
    obs = sim.observations[0]

    lbs.noise.add_correlated_noise(
        tod=obs.tod,
        sampling_rate_hz=obs.sampling_rate_hz,
        net_ukrts=obs.net_ukrts,
        fknee_mhz=obs.fknee_mhz,
        fmin_hz=obs.fmin_hz,
        alpha=obs.alpha,
        dets_random=sim.dets_random,
        corr_matrix=R,
    )

    # All rows should be nearly identical (same unit stream, same sigma).
    # Use atol rather than rtol: near zero crossings the relative error blows up
    # even though the absolute difference is tiny (~8e-9 = regularisation term).
    typical = np.std(obs.tod[0])
    atol = 1e-3 * typical  # 0.1% of RMS is tight enough
    np.testing.assert_allclose(obs.tod[0], obs.tod[1], atol=atol)
    np.testing.assert_allclose(obs.tod[0], obs.tod[2], atol=atol)


def test_correlated_noise_cholesky_reproducible():
    """Same seed => identical output for corr_matrix path."""
    n = 2
    R = np.array([[1.0, 0.8], [0.8, 1.0]])

    sim1 = _make_sim(n, seed=11)
    sim2 = _make_sim(n, seed=11)

    for sim in (sim1, sim2):
        obs = sim.observations[0]
        lbs.noise.add_correlated_noise(
            tod=obs.tod,
            sampling_rate_hz=obs.sampling_rate_hz,
            net_ukrts=obs.net_ukrts,
            fknee_mhz=obs.fknee_mhz,
            fmin_hz=obs.fmin_hz,
            alpha=obs.alpha,
            dets_random=sim.dets_random,
            corr_matrix=R,
        )

    np.testing.assert_array_equal(sim1.observations[0].tod, sim2.observations[0].tod)


def test_correlated_noise_cholesky_not_psd():
    """Non-PSD corr_matrix raises ValueError."""
    n = 3
    # A clearly non-PD matrix
    R = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    sim = _make_sim(n, seed=1)
    obs = sim.observations[0]

    with pytest.raises(ValueError, match="not positive-definite"):
        lbs.noise.add_correlated_noise(
            tod=obs.tod,
            sampling_rate_hz=obs.sampling_rate_hz,
            net_ukrts=obs.net_ukrts,
            fknee_mhz=obs.fknee_mhz,
            fmin_hz=obs.fmin_hz,
            alpha=obs.alpha,
            dets_random=sim.dets_random,
            corr_matrix=R,
        )


def test_correlated_noise_cholesky_wrong_shape():
    """corr_matrix with wrong shape raises ValueError."""
    sim = _make_sim(3, seed=1)
    obs = sim.observations[0]

    with pytest.raises(ValueError, match="must be a"):
        lbs.noise.add_correlated_noise(
            tod=obs.tod,
            sampling_rate_hz=obs.sampling_rate_hz,
            net_ukrts=obs.net_ukrts,
            fknee_mhz=obs.fknee_mhz,
            fmin_hz=obs.fmin_hz,
            alpha=obs.alpha,
            dets_random=sim.dets_random,
            corr_matrix=np.eye(2),  # wrong: 2x2 for 3 detectors
        )


# --- add_noise_to_observations with correlated noise ---


def test_add_noise_to_observations_correlated_group_by_none():
    """noise_type='correlated' with group_by=None (single group)."""
    sim = _make_sim(3, seed=20)
    tod_before = sim.observations[0].tod.copy()

    lbs.noise.add_noise_to_observations(
        sim.observations,
        "correlated",
        dets_random=sim.dets_random,
        correlation={"group_by": None, "rho": 0.3},
    )

    assert np.any(sim.observations[0].tod != tod_before)


def test_add_noise_to_observations_correlated_explicit_groups():
    """noise_type='correlated' with explicit groups array."""
    sim = _make_sim(4, seed=21)

    lbs.noise.add_noise_to_observations(
        sim.observations,
        "correlated",
        dets_random=sim.dets_random,
        correlation={"groups": [0, 0, 1, 1], "rho": 0.5},
    )

    assert np.any(sim.observations[0].tod != 0)


def test_add_noise_to_observations_correlated_corr_matrix():
    """noise_type='correlated' with corr_matrix key."""
    n = 3
    R = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.6], [0.3, 0.6, 1.0]])
    sim = _make_sim(n, seed=22)

    lbs.noise.add_noise_to_observations(
        sim.observations,
        "correlated",
        dets_random=sim.dets_random,
        correlation={"corr_matrix": R},
    )

    assert np.any(sim.observations[0].tod != 0)


def test_add_noise_to_observations_correlated_missing_correlation():
    """noise_type='correlated' without correlation dict raises ValueError."""
    sim = _make_sim(2, seed=1)

    with pytest.raises(ValueError, match="requires the 'correlation' argument"):
        lbs.noise.add_noise_to_observations(
            sim.observations,
            "correlated",
            dets_random=sim.dets_random,
        )


def test_add_noise_to_observations_unknown_noise_type():
    sim = _make_sim(2, seed=1)
    with pytest.raises(ValueError, match="Unknown noise type"):
        lbs.noise.add_noise_to_observations(
            sim.observations, "bad_type", dets_random=sim.dets_random
        )


# --- Simulation.add_noise integration ---


def test_simulation_add_noise_correlated_common_mode():
    """Simulation.add_noise with noise_type='correlated' and common-mode model."""
    sim = _make_sim(3, seed=30)
    sim.add_noise(
        noise_type="correlated",
        correlation={"group_by": None, "rho": 0.5},
    )
    assert np.any(sim.observations[0].tod != 0)


def test_simulation_add_noise_correlated_cholesky():
    """Simulation.add_noise with noise_type='correlated' and corr_matrix."""
    n = 3
    R = np.array([[1.0, 0.7, 0.4], [0.7, 1.0, 0.7], [0.4, 0.7, 1.0]])
    sim = _make_sim(n, seed=31)
    sim.add_noise(
        noise_type="correlated",
        correlation={"corr_matrix": R},
    )
    assert np.any(sim.observations[0].tod != 0)
