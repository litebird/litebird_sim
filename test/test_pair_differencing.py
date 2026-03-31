import healpy as hp
import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim import CoordinateSystem
from litebird_sim.mapmaking.pair_differencing import make_pair_differenced_map


def _build_observation(
    detectors: list[dict],
) -> tuple[lbs.Observation, np.ndarray, int]:
    n_samples = 4
    theta, phi = hp.pix2ang(1, 0)
    psi = np.array([0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0])

    obs = lbs.Observation(
        detectors=detectors,
        n_samples_global=n_samples,
        start_time_global=0.0,
        sampling_rate_hz=1.0,
    )

    pointings = np.empty((len(detectors), n_samples, 3), dtype=np.float64)
    pointings[:, :, 0] = theta
    pointings[:, :, 1] = phi
    pointings[:, :, 2] = psi

    return obs, pointings, hp.ang2pix(1, theta, phi)


def test_make_pair_differenced_map_recovers_qu():
    detectors = [
        {
            "name": "pair_T",
            "wafer": "L00",
            "pixel": 12,
            "pol": "T",
            "pol_angle_rad": 0.0,
            "net_ukrts": 2.0,
        },
        {
            "name": "pair_B",
            "wafer": "L00",
            "pixel": 12,
            "pol": "B",
            "pol_angle_rad": np.pi / 2.0,
            "net_ukrts": 4.0,
        },
    ]
    obs, pointings, pix_idx = _build_observation(detectors)

    q_true = 3.0
    u_true = -1.5
    psi_base = pointings[0, :, 2]
    psi_t = psi_base + obs.pol_angle_rad[0]
    psi_b = psi_base + obs.pol_angle_rad[1]

    obs.tod[0] = q_true * np.cos(2 * psi_t) + u_true * np.sin(2 * psi_t)
    obs.tod[1] = q_true * np.cos(2 * psi_b) + u_true * np.sin(2 * psi_b)

    result = make_pair_differenced_map(
        nside=1,
        observations=[obs],
        pointings=[pointings],
        output_coordinate_system=CoordinateSystem.Ecliptic,
    )

    assert result.binned_map.shape == (2, 12)
    assert np.allclose(result.binned_map[:, pix_idx], np.array([q_true, u_true]))

    unseen_mask = np.ones(result.binned_map.shape[1], dtype=bool)
    unseen_mask[pix_idx] = False
    assert np.all(result.binned_map[:, unseen_mask] == hp.UNSEEN)

    pair_weight = 0.5 * (
        detectors[0]["net_ukrts"] ** 2 + detectors[1]["net_ukrts"] ** 2
    )
    design_matrix = np.column_stack(
        (
            np.cos(2 * psi_t) - np.cos(2 * psi_b),
            np.sin(2 * psi_t) - np.sin(2 * psi_b),
        )
    )
    expected_npp = design_matrix.T @ design_matrix / pair_weight

    assert np.allclose(result.invnpp[pix_idx], np.linalg.inv(expected_npp))


def test_make_pair_differenced_map_rejects_unpaired_detectors():
    detectors = [
        {
            "name": "pair_T0",
            "wafer": "L00",
            "pixel": 12,
            "pol": "T",
            "pol_angle_rad": 0.0,
        },
        {
            "name": "pair_T1",
            "wafer": "L00",
            "pixel": 12,
            "pol": "T",
            "pol_angle_rad": 0.0,
        },
        {
            "name": "pair_B",
            "wafer": "L00",
            "pixel": 12,
            "pol": "B",
            "pol_angle_rad": np.pi / 2.0,
        },
    ]
    obs, pointings, _ = _build_observation(detectors)

    with pytest.raises(ValueError, match="more than one 'T' detector"):
        make_pair_differenced_map(
            nside=1,
            observations=[obs],
            pointings=[pointings],
            output_coordinate_system=CoordinateSystem.Ecliptic,
        )


def test_make_pair_differenced_map_is_split_aware_for_pairs():
    detectors = [
        {
            "name": "pair_T",
            "wafer": "L00",
            "pixel": 12,
            "pol": "T",
            "pol_angle_rad": 0.0,
            "net_ukrts": 2.0,
        },
        {
            "name": "pair_B",
            "wafer": "L00",
            "pixel": 12,
            "pol": "B",
            "pol_angle_rad": np.pi / 2.0,
            "net_ukrts": 4.0,
        },
        {
            "name": "extra_T",
            "wafer": "L01",
            "pixel": 99,
            "pol": "T",
            "pol_angle_rad": 0.0,
            "net_ukrts": 3.0,
        },
    ]
    obs, pointings, pix_idx = _build_observation(detectors)

    q_true = 1.5
    u_true = -0.25
    psi_base = pointings[0, :, 2]
    psi_t = psi_base + obs.pol_angle_rad[0]
    psi_b = psi_base + obs.pol_angle_rad[1]

    obs.tod[0] = q_true * np.cos(2 * psi_t) + u_true * np.sin(2 * psi_t)
    obs.tod[1] = q_true * np.cos(2 * psi_b) + u_true * np.sin(2 * psi_b)
    obs.tod[2] = 0.0

    # This should succeed because the selected split waferL00 is correctly paired,
    # even though the full detector list is not.
    result = make_pair_differenced_map(
        nside=1,
        observations=[obs],
        pointings=[pointings],
        output_coordinate_system=CoordinateSystem.Ecliptic,
        detector_split="waferL00",
    )

    assert np.allclose(result.binned_map[:, pix_idx], np.array([q_true, u_true]))


def test_simulation_make_pair_differenced_map_splits(tmp_path):
    detectors = [
        {
            "name": "pair_T",
            "wafer": "L00",
            "pixel": 12,
            "pol": "T",
            "pol_angle_rad": 0.0,
            "net_ukrts": 2.0,
        },
        {
            "name": "pair_B",
            "wafer": "L00",
            "pixel": 12,
            "pol": "B",
            "pol_angle_rad": np.pi / 2.0,
            "net_ukrts": 4.0,
        },
    ]
    obs, pointings, pix_idx = _build_observation(detectors)

    q_true = 0.75
    u_true = 0.5
    psi_base = pointings[0, :, 2]
    psi_t = psi_base + obs.pol_angle_rad[0]
    psi_b = psi_base + obs.pol_angle_rad[1]
    obs.tod[0] = q_true * np.cos(2 * psi_t) + u_true * np.sin(2 * psi_t)
    obs.tod[1] = q_true * np.cos(2 * psi_b) + u_true * np.sin(2 * psi_b)

    # The Simulation wrapper does not accept external pointings, so we preload them.
    obs.pointing_matrix = pointings

    sim = lbs.Simulation(
        base_path=tmp_path / "pairdiff_splits",
        start_time=0.0,
        duration_s=1.0,
        random_seed=1,
    )
    sim.observations = [obs]

    results = sim.make_pair_differenced_map_splits(
        nside=1,
        output_coordinate_system=CoordinateSystem.Ecliptic,
        detector_splits=["full"],
        time_splits=["full"],
        write_to_disk=False,
    )

    assert "full_full" in results
    assert np.allclose(
        results["full_full"].binned_map[:, pix_idx], np.array([q_true, u_true])
    )
