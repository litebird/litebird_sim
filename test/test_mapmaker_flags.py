"""Contract of the map-makers: a flagged sample does not change the map"""

import numpy as np
import pytest

import litebird_sim as lbs

FLAG_BIT = 0b10


def _simulation():
    sim = lbs.Simulation(start_time=0.0, duration_s=1800.0, random_seed=1)
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.radians(30),
            spin_rate_hz=0.5 / 60,
            precession_rate_hz=1 / 3600,
        )
    )
    sim.set_instrument(
        lbs.InstrumentInfo(name="i", spin_boresight_angle_rad=np.deg2rad(65))
    )
    # One T/B pair, for the pair-differencing
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name=f"d{pol}",
                sampling_rate_hz=2.0,
                net_ukrts=1.0,
                pol_angle_rad=angle,
                wafer="W",
                pixel=0,
                pol=pol,
            )
            for pol, angle in (("T", 0.0), ("B", np.pi / 2))
        ]
    )
    sim.prepare_pointings()
    sim.precompute_pointings()
    (obs,) = sim.observations
    rng = np.random.default_rng(1)
    obs.tod[:] = rng.normal(size=obs.tod.shape)

    # Flag one sample out of seven in each detector (different ones), with a
    # bit that matters, and some samples with a bit that does not
    obs.local_flags = np.zeros(obs.tod.shape, dtype=np.uint8)
    obs.local_flags[0, ::7] = FLAG_BIT
    obs.local_flags[1, 3::7] = FLAG_BIT
    obs.local_flags[:, 5::11] |= 0b01
    return sim


def _binner(obs, nside):
    return lbs.make_binned_map(
        nside, obs, flags="local_flags", flag_mask=FLAG_BIT
    ).binned_map


def _destriper(obs, nside):
    params = lbs.DestriperParameters(samples_per_baseline=60, iter_max=20)
    return lbs.make_destriped_map(
        nside, obs, params=params, flags="local_flags", flag_mask=FLAG_BIT
    ).destriped_map


def _pair_differencing(obs, nside):
    return lbs.make_pair_differenced_map(
        nside, obs, flags="local_flags", flag_mask=FLAG_BIT
    ).binned_map


@pytest.mark.parametrize("map_maker", [_binner, _destriper, _pair_differencing])
def test_flagged_samples_do_not_change_the_map(map_maker):
    nside = 8
    sim = _simulation()
    (obs,) = sim.observations
    reference = map_maker(obs, nside)

    flagged = (obs.local_flags & FLAG_BIT) != 0
    obs.tod[flagged] = 1e6
    np.testing.assert_allclose(map_maker(obs, nside), reference, rtol=1e-8, atol=1e-8)

    # The samples with the other bits are used
    obs.tod[((obs.local_flags & 0b01) != 0) & ~flagged] += 1.0
    assert not np.allclose(map_maker(obs, nside), reference)


def test_simulation_pair_differencing_passes_the_flags():
    nside = 8
    sim = _simulation()
    (obs,) = sim.observations
    expected = _pair_differencing(obs, nside)
    result = sim.make_pair_differenced_map(
        nside, flags="local_flags", flag_mask=FLAG_BIT, append_to_report=False
    )
    np.testing.assert_array_equal(result.binned_map, expected)
    (split,) = sim.make_pair_differenced_map_splits(
        nside,
        write_to_disk=False,
        flags="local_flags",
        flag_mask=FLAG_BIT,
        append_to_report=False,
    ).values()
    np.testing.assert_array_equal(split.binned_map, expected)
