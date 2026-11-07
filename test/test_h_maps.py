import healpy as hp
import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim.mapmaking.h_maps import load_h_maps_from_file


SPIN_PAIRS = np.array([[0, 0], [2, 0], [4, 0], [0, 2], [2, -4]])


def _make_observation(detector_samples, start_time=0.0, hwp_angle=None):
    names = list(detector_samples)
    n_samples = len(detector_samples[names[0]][0])
    obs = lbs.Observation(
        detectors=[{"name": name, "pol_angle_rad": 0.3} for name in names],
        n_samples_global=n_samples,
        sampling_rate_hz=1.0,
        start_time_global=start_time,
        allocate_tod=False,
    )
    obs.pointing_matrix = np.empty((len(names), n_samples, 3))
    for idet, (pixels, psi) in enumerate(detector_samples.values()):
        theta, phi = hp.pix2ang(1, pixels)
        obs.pointing_matrix[idet] = np.column_stack((theta, phi, psi))
    if hwp_angle is not None:
        obs.has_hwp = True
        obs.hwp_angle = np.asarray(hwp_angle)
    return obs


def _assert_map_matches_samples(h_map, pixels, psi, hwp_angle):
    counts = np.bincount(pixels, minlength=12)
    if (h_map.n, h_map.m) == (0, 0):
        expected_real = counts
        expected_imag = np.zeros(12)
    else:
        phase = h_map.n * psi + h_map.m * hwp_angle
        expected_real = np.divide(
            np.bincount(pixels, weights=np.cos(phase), minlength=12),
            counts,
            out=np.zeros(12),
            where=counts > 0,
        )
        expected_imag = np.divide(
            np.bincount(pixels, weights=np.sin(phase), minlength=12),
            counts,
            out=np.zeros(12),
            where=counts > 0,
        )
    np.testing.assert_allclose(h_map.real, expected_real, atol=1e-14)
    np.testing.assert_allclose(h_map.imag, expected_imag, atol=1e-14)


def test_h_maps_keep_detector_names_across_observations():
    chunks = [
        {"A": ([0, 0, 1], [0.0, 0.2, 0.4]), "B": ([2, 3, 3], [0.1, 0.3, 0.5])},
        {"C": ([4, 4], [0.6, 0.8])},
        {"E": ([5, 6], [0.7, 0.9]), "D": ([7, 7], [0.8, 1.0])},
    ]
    result = lbs.make_h_maps(
        observations=[_make_observation(chunk) for chunk in chunks],
        nside=1,
        n_m_couples=SPIN_PAIRS,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        save_to_file=False,
        nthreads=1,
    )

    assert set(result.h_maps) == {"A", "B", "C", "D", "E"}
    for chunk in chunks:
        for name, (pixels, psi) in chunk.items():
            assert set(result.h_maps[name]) == set(map(tuple, SPIN_PAIRS))
            for h_map in result.h_maps[name].values():
                assert h_map.det_info == name
                _assert_map_matches_samples(
                    h_map, np.array(pixels), np.array(psi), np.zeros(len(pixels))
                )


@pytest.mark.parametrize("with_hwp", [False, True])
@pytest.mark.parametrize("time_split", ["full", "even"])
def test_h_maps_accumulate_before_normalizing(tmp_path, with_hwp, time_split):
    # Unequal hit counts distinguish sample-weighted accumulation from averaging
    # the already normalized maps. The detector order changes between chunks.
    chunks = [
        {"A": ([0, 0, 1], [0.0, 0.2, 0.4]), "B": ([2, 3, 3], [0.1, 0.3, 0.5])},
        {
            "B": ([2, 2, 2, 4, 4], [0.7, 0.9, 1.1, 1.3, 1.5]),
            "A": ([0, 0, 0, 0, 1], [0.6, 0.8, 1.0, 1.2, 1.4]),
        },
        {"A": ([5], [1.6]), "B": ([6], [1.7])},
    ]
    observations = []
    masks = []
    start_time = 0.0
    for chunk in chunks:
        n_samples = len(next(iter(chunk.values()))[0])
        hwp_angle = np.arange(n_samples) * 0.13 + start_time * 0.07
        observations.append(
            _make_observation(chunk, start_time, hwp_angle if with_hwp else None)
        )
        mask = np.ones(n_samples, dtype=bool)
        if time_split == "even":
            mask[0::2] = False
        masks.append(mask)
        start_time += n_samples

    output_directory = tmp_path / "h_maps"
    result = lbs.make_h_maps(
        observations=observations,
        nside=1,
        n_m_couples=SPIN_PAIRS,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        time_split=time_split,
        output_directory=str(output_directory),
        nthreads=1,
    )

    combined_samples = {}
    combined_hwp = np.concatenate(
        [
            (obs.hwp_angle if with_hwp else np.zeros(obs.n_samples))[mask]
            for obs, mask in zip(observations, masks)
        ]
    )
    for name in ["A", "B"]:
        pixels = np.concatenate(
            [np.array(chunk[name][0])[mask] for chunk, mask in zip(chunks, masks)]
        )
        psi = np.concatenate(
            [np.array(chunk[name][1])[mask] for chunk, mask in zip(chunks, masks)]
        )
        combined_samples[name] = pixels, psi
        assert set(result.h_maps[name]) == set(map(tuple, SPIN_PAIRS))
        loaded = load_h_maps_from_file(output_directory / f"h_maps_det_{name}.h5")
        for pair, h_map in result.h_maps[name].items():
            _assert_map_matches_samples(h_map, pixels, psi, combined_hwp)
            np.testing.assert_array_equal(loaded.h_maps[name][pair].real, h_map.real)
            np.testing.assert_array_equal(loaded.h_maps[name][pair].imag, h_map.imag)

    combined = lbs.make_h_maps(
        observations=_make_observation(
            combined_samples, hwp_angle=combined_hwp if with_hwp else None
        ),
        nside=1,
        n_m_couples=SPIN_PAIRS,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        save_to_file=False,
        nthreads=1,
    )
    for name in ["A", "B"]:
        for pair, h_map in result.h_maps[name].items():
            np.testing.assert_allclose(h_map.real, combined.h_maps[name][pair].real)
            np.testing.assert_allclose(h_map.imag, combined.h_maps[name][pair].imag)


def test_h_maps_repeated_spin_pairs_do_not_count_samples_twice():
    obs = _make_observation({"A": ([0, 0, 1], [0.0, 0.2, 0.4])})
    result = lbs.make_h_maps(
        observations=[obs, obs],
        nside=1,
        n_m_couples=np.array([[0, 0], [2, 0], [0, 0], [2, 0]]),
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        save_to_file=False,
        nthreads=1,
    )
    assert set(result.h_maps["A"]) == {(0, 0), (2, 0)}
    for h_map in result.h_maps["A"].values():
        _assert_map_matches_samples(
            h_map,
            np.array([0, 0, 1, 0, 0, 1]),
            np.tile([0.0, 0.2, 0.4], 2),
            np.zeros(6),
        )


def test_h_maps_stream_to_disk(tmp_path):
    chunks = [
        {"A": ([0, 0, 1], [0.0, 0.2, 0.4]), "B": ([2, 3, 3], [0.1, 0.3, 0.5])},
        {"B": ([2, 2], [0.7, 0.9]), "A": ([0, 1], [0.6, 0.8])},
        {"C": ([4], [1.0])},
    ]
    observations = [
        _make_observation(
            chunk, hwp_angle=np.arange(len(next(iter(chunk.values()))[0])) * 0.13
        )
        for chunk in chunks
    ]
    reference = lbs.make_h_maps(
        observations,
        nside=1,
        n_m_couples=SPIN_PAIRS,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        save_to_file=False,
    )
    result = lbs.make_h_maps(
        observations,
        nside=1,
        n_m_couples=SPIN_PAIRS,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        output_directory=str(tmp_path),
        keep_maps=False,
    )
    assert result.h_maps == {}
    assert set(result.filenames) == set(reference.h_maps)
    for name, filename in result.filenames.items():
        loaded = load_h_maps_from_file(filename)
        assert loaded.coordinate_system == reference.coordinate_system
        assert loaded.duration_s == reference.duration_s
        for pair, expected in reference.h_maps[name].items():
            np.testing.assert_array_equal(loaded.h_maps[name][pair].real, expected.real)
            np.testing.assert_array_equal(loaded.h_maps[name][pair].imag, expected.imag)


def test_h_maps_stream_bounds_buffers_and_writes_before_next_detector(
    tmp_path, monkeypatch
):
    import weakref
    import importlib

    h_maps_module = importlib.import_module("litebird_sim.mapmaking.h_maps")

    samples = {name: ([0, 1], [0.2, 0.4]) for name in ("A", "B", "C")}
    obs = _make_observation(samples)
    original_zeros = np.zeros
    refs = []
    allocations = 0
    pairs = len(SPIN_PAIRS)

    def tracked_zeros(shape, *args, **kwargs):
        nonlocal allocations
        if isinstance(shape, tuple) and shape == (12, 3):
            array = original_zeros(shape, *args, **kwargs)
            refs.append(weakref.ref(array))
            allocations += 1
            assert sum(ref() is not None for ref in refs) <= pairs
            return array
        return original_zeros(shape, *args, **kwargs)

    original_compute = h_maps_module._compute_pixel_indices_single_detector

    def compute_after_previous_write(*args, **kwargs):
        idet = kwargs["detector_index"]
        if idet:
            previous = list(samples)[idet - 1]
            assert (tmp_path / f"h_maps_det_{previous}.h5").exists()
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(np, "zeros", tracked_zeros)
    monkeypatch.setattr(
        h_maps_module,
        "_compute_pixel_indices_single_detector",
        compute_after_previous_write,
    )
    result = lbs.make_h_maps(
        obs,
        nside=1,
        n_m_couples=SPIN_PAIRS,
        output_coordinate_system=lbs.CoordinateSystem.Ecliptic,
        output_directory=str(tmp_path),
        keep_maps=False,
    )
    assert result.h_maps == {}
    assert allocations == pairs
    assert all(ref() is None for ref in refs)


def test_h_maps_require_an_output():
    obs = _make_observation({"A": ([0], [0.2])})
    with pytest.raises(ValueError, match="save_to_file"):
        lbs.make_h_maps(obs, nside=1, save_to_file=False, keep_maps=False)


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
