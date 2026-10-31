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
