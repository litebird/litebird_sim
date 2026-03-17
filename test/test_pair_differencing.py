import healpy as hp
import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim import CoordinateSystem
from litebird_sim.mapmaking.pair_differencing import make_pair_differenced_map


def _build_observation(detectors: list[dict]) -> tuple[lbs.Observation, np.ndarray, int]:
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

    pair_weight = 0.5 * (detectors[0]["net_ukrts"] ** 2 + detectors[1]["net_ukrts"] ** 2)
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