import numpy as np

import litebird_sim as lbs

NSIDE = 16
N_M_COUPLES = np.array([[0, 0], [2, 0], [4, 0]])


def _observations(det_names, duration_s, num_of_obs_per_detector=1):
    sim = lbs.Simulation(start_time=0.0, duration_s=duration_s, random_seed=1)
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            precession_rate_hz=1 / 3600,
            spin_rate_hz=1 / 60,
        ),
        delta_time_s=1.0,
    )
    sim.set_instrument(
        lbs.InstrumentInfo(name="instr", spin_boresight_angle_rad=np.deg2rad(50.0))
    )
    dets = [
        lbs.DetectorInfo(
            name=name,
            sampling_rate_hz=5.0,
            quat=lbs.RotQuaternion(
                np.array(lbs.quat_rotation_y(np.deg2rad(1.0 + 2.0 * i)))
            ),
        )
        for i, name in enumerate(det_names)
    ]
    sim.create_observations(
        detectors=dets,
        num_of_obs_per_detector=num_of_obs_per_detector,
        allocate_tod=False,
    )
    sim.prepare_pointings()
    return sim.observations


def _make(obs_list):
    return lbs.make_h_maps(
        observations=obs_list,
        nside=NSIDE,
        n_m_couples=N_M_COUPLES,
        save_to_file=False,
    )


def test_h_maps_detector_names_across_observations():
    # Each observation holds a different detector: the maps must be stored under
    # the detector that produced them
    obs_a = _observations(["a"], duration_s=600.0)
    obs_b_alt = _observations(["x", "b"], duration_s=600.0)

    ref_a = _make(obs_a).h_maps["a"]
    ref_b = _make(obs_b_alt).h_maps["b"]

    combined = _make(obs_a + obs_b_alt).h_maps
    assert set(combined.keys()) == {"a", "x", "b"}

    for nm in ref_a:
        np.testing.assert_array_equal(combined["a"][nm].real, ref_a[nm].real)
        np.testing.assert_array_equal(combined["b"][nm].real, ref_b[nm].real)
        np.testing.assert_array_equal(combined["b"][nm].imag, ref_b[nm].imag)


def test_h_maps_accumulate_over_observations():
    # Splitting the same time stream into several observations must not change
    # the maps: the hits of all the observations must be combined
    single = _make(_observations(["a"], duration_s=1200.0))
    split = _make(_observations(["a"], duration_s=1200.0, num_of_obs_per_detector=3))

    for nm in single.h_maps["a"]:
        np.testing.assert_allclose(
            split.h_maps["a"][nm].real, single.h_maps["a"][nm].real, atol=1e-12
        )
        np.testing.assert_allclose(
            split.h_maps["a"][nm].imag, single.h_maps["a"][nm].imag, atol=1e-12
        )
    np.testing.assert_allclose(split.duration_s, single.duration_s)


def test_h_maps_duration_is_union_of_time_intervals():
    # Observations with the same start but different lengths must give the
    # length of their union, whatever their order
    short = _observations(["a"], duration_s=600.0)
    long = _observations(["b"], duration_s=1200.0)

    assert _make(short + long).duration_s == _make(long + short).duration_s
    np.testing.assert_allclose(_make(short + long).duration_s, 1200.0)


def test_union_length_s():
    from litebird_sim.mapmaking.h_maps import _union_length_s

    assert _union_length_s([]) == 0.0
    assert _union_length_s([(0, 10), (0, 5)]) == 10.0
    assert _union_length_s([(20, 30), (0, 10), (5, 12)]) == 22.0
