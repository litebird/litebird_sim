"""Check that the map-makers account for the polarization efficiency (issue #566)"""

import numpy as np
import healpy as hp
import pytest
import litebird_sim as lbs
from litebird_sim import CoordinateSystem, HealpixMap

NSIDE = 16
# Different γ for each detector, so that we also check that every detector
# uses its own value
POL_EFFICIENCIES = [0.8, 0.9, 0.85, 0.95]


def _simulate_noiseless_obs(tmp_path):
    sim = lbs.Simulation(
        base_path=tmp_path, start_time=0.0, duration_s=86400.0, random_seed=1
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.radians(45),
            spin_rate_hz=1 / 1200,
            precession_rate_hz=1 / 11520,
        )
    )
    instr = lbs.InstrumentInfo(
        spin_boresight_angle_rad=np.radians(50), boresight_rotangle_rad=0.0
    )

    # Two T/B pairs, so that the same observation works with pair differencing too
    dets = [
        lbs.DetectorInfo(
            name=f"det{idx}",
            sampling_rate_hz=1.0,
            net_ukrts=1.0,  # Only sets the map-making weights; no noise is added
            quat=[0.0, 0.0, 0.0, 1.0],
            pol_angle_rad=pol_angle,
            pol_efficiency=gamma,
            wafer="W",
            pixel=pixel,
            pol=pol,
        )
        for idx, (pol_angle, gamma, pixel, pol) in enumerate(
            zip(
                [0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4],
                POL_EFFICIENCIES,
                [0, 0, 1, 1],
                ["T", "B", "T", "B"],
            )
        )
    ]
    (obs,) = sim.create_observations(detectors=dets)
    lbs.prepare_pointings(obs, instr, sim.spin2ecliptic_quats)

    # Constant T, Q, U sky
    sky = np.ones((3, hp.nside2npix(NSIDE))) * np.array([[10.0], [2.0], [-3.0]])
    sky_map = HealpixMap(sky, coordinates=CoordinateSystem.Ecliptic)
    lbs.scan_map_in_observations(obs, maps={d.name: sky_map for d in dets})

    return obs, sky


@pytest.mark.parametrize("mapmaker", ["binner", "destriper", "pair_differencing"])
def test_mapmakers_account_for_pol_efficiency(tmp_path, mapmaker):
    obs, sky = _simulate_noiseless_obs(tmp_path)

    if mapmaker == "binner":
        out = lbs.make_binned_map(
            nside=NSIDE,
            observations=[obs],
            output_coordinate_system=CoordinateSystem.Ecliptic,
        ).binned_map
    elif mapmaker == "destriper":
        out = lbs.make_destriped_map(
            nside=NSIDE,
            observations=[obs],
            params=lbs.DestriperParameters(
                output_coordinate_system=CoordinateSystem.Ecliptic
            ),
        ).binned_map
    else:
        out = lbs.make_pair_differenced_map(
            nside=NSIDE,
            observations=[obs],
            output_coordinate_system=CoordinateSystem.Ecliptic,
        ).binned_map

    # Pair differencing only returns Q and U
    out_qu = out[-2:]
    seen = np.isfinite(out_qu[0]) & (out_qu[0] != hp.UNSEEN)
    assert np.sum(seen) > 0

    if mapmaker != "pair_differencing":
        np.testing.assert_allclose(out[0, seen], sky[0, seen], rtol=1e-6)
    np.testing.assert_allclose(out_qu[:, seen], sky[1:, seen], rtol=1e-6)
