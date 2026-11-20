"""Detector and time splits of the map-makers"""

import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim.mapmaking.splits import (
    T_SURVEY_S,
    T_YEAR_S,
    _build_mask_detector_split,
    _build_mask_time_split,
)


def _observations(duration_s, sampling_rate_hz, wafers=("L00",)):
    sim = lbs.Simulation(start_time=0.0, duration_s=duration_s, random_seed=1)
    dets = [
        lbs.DetectorInfo(name=f"d{i}", sampling_rate_hz=sampling_rate_hz, wafer=w)
        for i, w in enumerate(wafers)
    ]
    sim.create_observations(detectors=dets)
    return sim.observations


def test_unknown_splits_raise_everywhere():
    obs = _observations(100.0, 1.0)
    with pytest.raises(ValueError, match="time split"):
        _build_mask_time_split("yaer2", obs)
    with pytest.raises(ValueError, match="time split"):
        lbs.check_valid_splits(obs, time_splits="yaer2")
    with pytest.raises(ValueError, match="[Dd]etector split"):
        _build_mask_detector_split("wafr", obs)
    with pytest.raises(ValueError, match="[Dd]etector split"):
        lbs.check_valid_splits(obs, detector_splits="wafr")


def test_impossible_splits_raise():
    # 1.5 years: year1 and survey3 fit, year2 and survey4 do not
    obs = _observations(1.5 * T_YEAR_S, 1e-3)
    lbs.check_valid_splits(obs, time_splits=["year1", "survey3"])
    for split in ("year2", "survey4"):
        with pytest.raises(AssertionError, match=split):
            lbs.check_valid_splits(obs, time_splits=split)
        with pytest.raises(AssertionError, match=split):
            _build_mask_time_split(split, obs)

    with pytest.raises(AssertionError, match="L01"):
        _build_mask_detector_split("waferL01", obs)


def test_period_masks_partition_the_samples():
    obs = _observations(1.5 * T_YEAR_S, 1e-3)
    times = obs[0].get_times() - obs[0].get_times()[0]

    year1 = _build_mask_time_split("year1", obs)[0]
    np.testing.assert_array_equal(year1, times < T_YEAR_S)

    surveys = [_build_mask_time_split(f"survey{i}", obs)[0] for i in (1, 2, 3)]
    np.testing.assert_array_equal(np.sum(surveys, axis=0), 1)
    np.testing.assert_array_equal(
        surveys[1], (times >= T_SURVEY_S) & (times < 2 * T_SURVEY_S)
    )


def test_wafer_split():
    obs = _observations(100.0, 1.0, wafers=("L00", "L01", "L00"))
    np.testing.assert_array_equal(
        _build_mask_detector_split("waferL00", obs)[0], [True, False, True]
    )
    np.testing.assert_array_equal(_build_mask_detector_split("full", obs)[0], True)


def test_check_valid_splits_is_silent(capsys):
    lbs.check_valid_splits(_observations(100.0, 1.0), "full", "odd")
    assert capsys.readouterr().out == ""


def test_map_maker_validates_detector_split():
    sim = lbs.Simulation(start_time=0.0, duration_s=100.0, random_seed=1)
    sim.set_scanning_strategy(lbs.SpinningScanningStrategy(0.7, 1e-4, 1e-2))
    sim.set_instrument(lbs.InstrumentInfo(name="i", spin_boresight_angle_rad=0.8))
    sim.create_observations(
        detectors=[lbs.DetectorInfo(name="d", sampling_rate_hz=10.0, wafer="L00")]
    )
    sim.prepare_pointings()

    # Called directly, without going through Simulation
    with pytest.raises(ValueError, match="[Dd]etector split"):
        lbs.make_binned_map(
            nside=8, observations=sim.observations, detector_split="wafr"
        )
