import healpy as hp
import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim import CoordinateSystem
from litebird_sim.mapmaking.pair_differencing import make_pair_differenced_map


def _build_observation(
    detectors: list[dict],
    sampling_rate_hz: float = 1.0,
) -> tuple[lbs.Observation, np.ndarray, int]:
    n_samples = 4
    theta, phi = hp.pix2ang(1, 0)
    psi = np.array([0.0, np.pi / 8.0, np.pi / 4.0, 3.0 * np.pi / 8.0])

    obs = lbs.Observation(
        detectors=detectors,
        n_samples_global=n_samples,
        start_time_global=0.0,
        sampling_rate_hz=sampling_rate_hz,
    )

    pointings = np.empty((len(detectors), n_samples, 3), dtype=np.float64)
    pointings[:, :, 0] = theta
    pointings[:, :, 1] = phi
    pointings[:, :, 2] = psi

    return obs, pointings, hp.ang2pix(1, theta, phi)


@pytest.mark.parametrize("sampling_rate_hz", [1.0, 8.0])
def test_make_pair_differenced_map_recovers_qu(sampling_rate_hz):
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
    obs, pointings, pix_idx = _build_observation(detectors, sampling_rate_hz)

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

    pair_variance = sampling_rate_hz * (
        detectors[0]["net_ukrts"] ** 2 + detectors[1]["net_ukrts"] ** 2
    )
    design_matrix = np.column_stack(
        (
            np.cos(2 * psi_t) - np.cos(2 * psi_b),
            np.sin(2 * psi_t) - np.sin(2 * psi_b),
        )
    )
    expected_npp = design_matrix.T @ design_matrix / pair_variance

    assert np.allclose(result.invnpp[pix_idx], np.linalg.inv(expected_npp))


def test_make_pair_differenced_map_covariance_matches_binner():
    # Equal-NET orthogonal detectors with uniform angle coverage decouple I
    # from Q/U, so pair differencing and the IQU binner must agree on Q/U.
    detectors = [
        {
            "name": f"pair_{pol}",
            "wafer": "L00",
            "pixel": 12,
            "pol": pol,
            "pol_angle_rad": pol_angle_rad,
            "net_ukrts": 2.0,
        }
        for pol, pol_angle_rad in [("T", 0.0), ("B", np.pi / 2.0)]
    ]
    obs, pointings, pix_idx = _build_observation(detectors, sampling_rate_hz=3.0)
    q_true, u_true = 3.0, -1.5
    psi = pointings[:, :, 2] + obs.pol_angle_rad[:, None]
    obs.tod = 100.0 + q_true * np.cos(2 * psi) + u_true * np.sin(2 * psi)
    kwargs = dict(
        nside=1,
        observations=obs,
        pointings=pointings,
        output_coordinate_system=CoordinateSystem.Ecliptic,
    )

    result = make_pair_differenced_map(**kwargs)
    binned = lbs.make_binned_map(**kwargs)

    np.testing.assert_allclose(result.binned_map[:, pix_idx], [q_true, u_true])
    np.testing.assert_allclose(
        result.binned_map[:, pix_idx], binned.binned_map[1:, pix_idx]
    )
    # For N uniformly spaced orientations, each Q/U variance is sigma^2 / N.
    expected_covariance = (
        np.eye(2) * obs.sampling_rate_hz * obs.net_ukrts[0] ** 2 / obs.n_samples
    )
    np.testing.assert_allclose(result.invnpp[pix_idx], expected_covariance, atol=1e-14)
    np.testing.assert_allclose(
        result.invnpp[pix_idx], binned.invnpp[pix_idx, 1:, 1:], atol=1e-14
    )


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
