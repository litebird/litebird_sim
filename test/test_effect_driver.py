"""The TOD effects resolve their inputs and check the fields of the detectors
before modifying any TOD"""

import numpy as np
import pytest
from astropy.time import Time

import litebird_sim as lbs
from litebird_sim.hwp_diff_emiss import add_2f_to_observations
from litebird_sim.observation_utilities import _detector_keys

N_SAMPLES = 20

FULL_DETECTOR = {
    "name": "d0",
    "channel": "ch",
    "sampling_rate_hz": 1.0,
    "net_ukrts": 10.0,
    "fknee_mhz": 10.0,
    "fmin_hz": 1e-5,
    "alpha": 1.0,
    "bandcenter_ghz": 140.0,
    "bandwidth_ghz": 40.0,
    "pol_angle_rad": 0.0,
    "pol_efficiency": 1.0,
    "mueller_hwp": None,
    "amplitude_2f_k": 1.0,
    "wafer": "W",
}


def _observation(missing: str | None = None):
    det = {k: v for k, v in FULL_DETECTOR.items() if k != missing}
    obs = lbs.Observation(
        detectors=[det],
        n_samples_observation=N_SAMPLES,
        start_time_observation=0.0,
        sampling_rate_hz=1.0,
    )
    return obs


def test_detector_keys():
    obs = _observation()
    assert _detector_keys(obs, {"d0": 1, "other": 2}, "sky") == ["d0"]
    assert _detector_keys(obs, {"ch": 1}, "sky") == ["ch"]
    assert _detector_keys(obs, object(), "sky") is None
    with pytest.raises(ValueError, match="sky.*neither the names nor the channels"):
        _detector_keys(obs, {"x": 1}, "sky")


def _noise(observations):
    lbs.add_noise_to_observations(observations, noise_type="one_over_f", user_seed=1)


def _gain_drift(observations):
    lbs.apply_gaindrift_to_observations(observations, user_seed=1)


def _non_linearity(observations):
    lbs.apply_quadratic_nonlin_to_observations(observations, user_seed=1)


def _two_f(observations):
    add_2f_to_observations(observations)


def _scan_map(observations):
    nside = 2
    sky = lbs.HealpixMap(
        values=np.ones((3, 12 * nside**2)),
        nside=nside,
        coordinates=lbs.CoordinateSystem.Ecliptic,
    )
    pointings = [np.full((1, N_SAMPLES, 3), 0.5) for _ in observations]
    lbs.scan_map_in_observations(observations, maps=sky, pointings=pointings)


@pytest.mark.parametrize(
    "effect, field",
    [
        (_noise, "fknee_mhz"),
        (_noise, "net_ukrts"),
        (_gain_drift, "wafer"),
        (_non_linearity, "bandcenter_ghz"),
        (_two_f, "pol_angle_rad"),
        (_scan_map, "pol_angle_rad"),
    ],
)
def test_missing_field_leaves_the_tods_untouched(effect, field):
    # The field is missing in the second observation: the error names it, and
    # the first observation is not modified
    observations = [_observation(), _observation(missing=field)]
    with pytest.raises(ValueError, match=f"observation 1 have no '{field}'"):
        effect(observations)
    for obs in observations:
        assert np.all(obs.tod == 0.0)


def _simulation():
    sim = lbs.Simulation(
        start_time=Time("2025-01-01T00:00:00"), duration_s=60.0, random_seed=1
    )
    sim.set_scanning_strategy(
        lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=np.deg2rad(45.0),
            precession_rate_hz=1 / 3600,
            spin_rate_hz=1 / 60,
            start_time=sim.start_time,
        )
    )
    sim.set_instrument(
        lbs.InstrumentInfo(name="i", spin_boresight_angle_rad=np.deg2rad(50.0))
    )
    sim.create_observations(
        detectors=[
            lbs.DetectorInfo(
                name=f"d{i}",
                channel="ch",
                sampling_rate_hz=2.0,
                net_ukrts=10.0,
                fknee_mhz=10.0,
                bandcenter_ghz=140.0,
                bandwidth_ghz=40.0,
                wafer="W",
            )
            for i in range(2)
        ],
        tods=[
            lbs.TodDescription(
                name="tod", units=lbs.Units.K_CMB, dtype=np.float64, description="tod"
            ),
            lbs.TodDescription(
                name="extra",
                units=lbs.Units.K_CMB,
                dtype=np.float64,
                description="extra",
            ),
        ],
    )
    sim.set_hwp(lbs.IdealHWP(ang_speed_radpsec=2.0))
    sim.prepare_pointings()
    return sim


def _sky(lmax=8):
    rng = np.random.default_rng(1)
    nalm = lbs.SphericalHarmonics.num_of_alm_from_lmax(lmax)
    sky = lbs.SphericalHarmonics(
        values=rng.normal(size=(3, nalm)) + 0j,
        lmax=lmax,
        coordinates=lbs.CoordinateSystem.Galactic,
    )
    return sky


@pytest.mark.parametrize("call", ["scan_map", "sky", "beam", "dipole"])
def test_dictionary_without_the_detectors(call):
    sim = _simulation()
    obs_list = sim.observations
    wrong = {"x": _sky()}
    with pytest.raises(ValueError, match="neither the names nor the channels"):
        if call == "scan_map":
            lbs.scan_map_in_observations(obs_list, maps=wrong)
        elif call in ("sky", "beam"):
            beam = lbs.gauss_beam_to_alm(lmax=8, mmax=2, fwhm_rad=0.1)
            lbs.add_convolved_sky_to_observations(
                obs_list,
                sky_alms=wrong if call == "sky" else _sky(),
                beam_alms=wrong if call == "beam" else beam,
            )
        else:
            pos_vel = lbs.spacecraft_pos_and_vel(
                lbs.SpacecraftOrbit(sim.start_time),
                observations=obs_list,
                delta_time_s=10.0,
            )
            lbs.add_dipole_to_observations(
                obs_list,
                pos_and_vel=pos_vel,
                dipole_type=lbs.DipoleType.LINEAR,
                apply_convolution=True,
                beam_alms=wrong,
            )
    for obs in obs_list:
        assert np.all(obs.tod == 0.0)


def test_chain_of_effects_on_another_component():
    # Signal, dipole, 2f, noise and gain drift all go into "extra"
    sim = _simulation()
    obs_list = sim.observations
    lbs.scan_map_in_observations(
        obs_list, maps={"ch": _sky()}, hwp=sim.hwp, component="extra"
    )
    pos_vel = lbs.spacecraft_pos_and_vel(
        lbs.SpacecraftOrbit(sim.start_time), observations=obs_list, delta_time_s=10.0
    )
    lbs.add_dipole_to_observations(obs_list, pos_and_vel=pos_vel, component="extra")
    add_2f_to_observations(obs_list, amplitude_2f_k=1.0, component="extra")
    lbs.add_noise_to_observations(
        obs_list, noise_type="one_over_f", user_seed=1, component="extra"
    )
    lbs.apply_gaindrift_to_observations(obs_list, user_seed=1, component="extra")

    for obs in obs_list:
        assert np.all(obs.tod == 0.0)
        assert np.all(np.isfinite(obs.extra)) and np.any(obs.extra != 0.0)
