"""Tests for the shared systematic-effect driver in ``observation_utilities.py``.

The driver concentrates the bookkeeping that every effect module used to
duplicate (observation normalization, per-detector RNG resolution, TOD
component indirection), so it can be exercised directly here without going
through any particular effect.
"""

import numpy as np
import pytest
from astropy.time import Time

import litebird_sim as lbs


def _make_two_observations():
    """Build a two-detector simulation with one local observation."""
    dets = [
        lbs.DetectorInfo(name="det_A", sampling_rate_hz=1.0),
        lbs.DetectorInfo(name="det_B", sampling_rate_hz=1.0),
    ]
    sim = lbs.Simulation(
        start_time=Time("2025-02-02T00:00:00"),
        duration_s=10.0,
        random_seed=12345,
    )
    sim.create_observations(detectors=dets, split_list_over_processes=False)
    return sim, dets


def test_normalize_observations_single_is_wrapped():
    sim, _ = _make_two_observations()
    obs = sim.observations[0]

    result = lbs.normalize_observations(obs)

    assert result == [obs]
    assert result[0] is obs


def test_normalize_observations_list_is_copied():
    sim, _ = _make_two_observations()

    result = lbs.normalize_observations(sim.observations)

    # Same elements, but a fresh list the caller may mutate safely.
    assert result == sim.observations
    assert result is not sim.observations


def test_normalize_observations_rejects_other_types():
    with pytest.raises(TypeError):
        lbs.normalize_observations("not an observation")


def test_for_each_observation_without_rng_yields_tod_and_none():
    sim, _ = _make_two_observations()

    yielded = list(lbs.for_each_observation(sim.observations))

    assert len(yielded) == len(sim.observations)
    for (cur_obs, tod, dets_random), expected_obs in zip(yielded, sim.observations):
        assert cur_obs is expected_obs
        # The driver hands back the actual TOD array, not a copy.
        assert tod is expected_obs.tod
        assert dets_random is None


def test_for_each_observation_respects_component():
    sim, _ = _make_two_observations()
    for obs in sim.observations:
        obs.custom_tod = np.zeros_like(obs.tod)

    for cur_obs, tod, _ in lbs.for_each_observation(
        sim.observations, component="custom_tod"
    ):
        assert tod is cur_obs.custom_tod


def test_for_each_observation_rejects_unknown_component():
    sim, _ = _make_two_observations()
    with pytest.raises(ValueError, match="registered TOD components"):
        list(lbs.for_each_observation(sim.observations, component="toed"))


def test_for_each_observation_resolves_dets_random_once():
    sim, dets = _make_two_observations()
    hierarchy = lbs.RNGHierarchy(12345, num_detectors_per_rank=len(dets))
    dets_random = hierarchy.get_detector_level_generators_on_rank(0)

    seen = [
        rng
        for _, _, rng in lbs.for_each_observation(
            sim.observations, dets_random=dets_random, requires_rng=True
        )
    ]

    # Same generator list is handed out on every iteration.
    assert all(rng is dets_random for rng in seen)
    assert len(dets_random) >= sim.observations[0].n_detectors


def test_for_each_observation_requires_seed_or_generators_when_stochastic():
    sim, _ = _make_two_observations()

    with pytest.raises(ValueError):
        list(lbs.for_each_observation(sim.observations, requires_rng=True))


def test_for_each_observation_with_pointings_pairs_arrays():
    sim, _ = _make_two_observations()
    obs = sim.observations[0]
    n_det, n_samp = obs.tod.shape
    pointings = np.zeros((n_det, n_samp, 3))

    yielded = list(lbs.for_each_observation_with_pointings(obs, pointings))

    assert len(yielded) == 1
    cur_obs, tod, cur_ptg = yielded[0]
    assert cur_obs is obs
    assert tod is obs.tod
    assert cur_ptg is pointings


def test_for_each_observation_with_pointings_pairs_lists():
    sim, _ = _make_two_observations()
    obs_list = sim.observations
    pointings_list = [
        np.zeros((obs.tod.shape[0], obs.tod.shape[1], 3)) for obs in obs_list
    ]

    yielded = list(lbs.for_each_observation_with_pointings(obs_list, pointings_list))

    assert len(yielded) == len(obs_list)
    for (cur_obs, _, cur_ptg), exp_obs, exp_ptg in zip(
        yielded, obs_list, pointings_list
    ):
        assert cur_obs is exp_obs
        assert cur_ptg is exp_ptg
