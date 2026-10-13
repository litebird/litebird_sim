import healpy as hp
import numpy as np

import litebird_sim as lbs


def test_observation_get_hitmap():
    nside = 16
    sim = lbs.Simulation(start_time=0.0, duration_s=3600.0, random_seed=1)
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45),
            precession_rate_hz=1 / 3600 / 3.2,
            spin_rate_hz=0.05 / 60,
        )
    )
    sim.set_instrument(lbs.InstrumentInfo(spin_boresight_angle_rad=np.deg2rad(50)))
    dets = [
        lbs.DetectorInfo(name="a", sampling_rate_hz=1.0),
        lbs.DetectorInfo(
            name="b",
            sampling_rate_hz=1.0,
            quat=np.array(lbs.quat_rotation_y(np.deg2rad(5))),
        ),
    ]
    (obs,) = sim.create_observations(detectors=dets)
    sim.prepare_pointings()

    pointings, _ = obs.get_pointings("all")
    for coords, ptg in [
        (lbs.CoordinateSystem.Ecliptic, pointings),
        (
            lbs.CoordinateSystem.Galactic,
            lbs.rotate_coordinates_e2g(pointings.reshape(-1, 3)),
        ),
    ]:
        expected = np.bincount(
            hp.ang2pix(nside, ptg[..., 0].ravel(), ptg[..., 1].ravel()),
            minlength=hp.nside2npix(nside),
        )
        hitmap = obs.get_hitmap(nside, output_coordinate_system=coords)
        np.testing.assert_array_equal(hitmap, expected)
        assert hitmap.sum() == obs.n_detectors * obs.n_samples

    # A single detector, by index
    assert obs.get_hitmap(nside, detector_idx=1).sum() == obs.n_samples
