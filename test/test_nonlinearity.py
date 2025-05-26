import numpy as np
import litebird_sim as lbs
from astropy.time import Time


def test_add_quadratic_nonlinearity():
    # Test function to check consistency of wrappers and low level functions
    start_time = Time("2025-02-02T00:00:00")
    mission_time_days = 1
    sampling_hz = 1

    dets = [
        lbs.DetectorInfo(name="det_A", sampling_rate_hz=sampling_hz),
        lbs.DetectorInfo(name="det_B", sampling_rate_hz=sampling_hz),
    ]

    nl_params = lbs.NonLinParams(sampling_gaussian_loc=0.0, sampling_gaussian_scale=0.1)

    random_seed = 12345
    sim = lbs.Simulation(
        base_path="nonlin_example",
        start_time=start_time,
        duration_s=mission_time_days * 24 * 3600.0,
        random_seed=random_seed,
    )

    sim.create_observations(
        detectors=dets,
        split_list_over_processes=False,
    )

    # Creating fiducial TODs
    sim.observations[0].nl_2_self = np.ones_like(sim.observations[0].tod)
    sim.observations[0].nl_2_obs = np.ones_like(sim.observations[0].tod)
    sim.observations[0].nl_2_det = np.ones_like(sim.observations[0].tod)

    # Applying non-linearity using the `Simulation` class method
    sim.apply_quadratic_nonlin(
        nl_params=nl_params,
        component="nl_2_self",
    )

    # Applying non-linearity on the given TOD component of an `Observation` object
    sim.init_random(random_seed)
    lbs.apply_quadratic_nonlin_to_observations(
        observations=sim.observations,
        nl_params=nl_params,
        component="nl_2_obs",
        random=sim.random,
    )

    # Applying non-linearity on the TOD arrays of the individual detectors.
    sim.init_random(random_seed)
    for idx, tod in enumerate(sim.observations[0].nl_2_det):
        lbs.apply_quadratic_nonlin_for_one_detector(
            tod_det=tod,
            det_name=sim.observations[0].name[idx],
            nl_params=nl_params,
            user_seed=12345,
            random=sim.random,
        )

    # Check if the three non-linear tods are equal

    np.testing.assert_array_equal(
        sim.observations[0].nl_2_self, sim.observations[0].nl_2_obs
    )
    np.testing.assert_array_equal(
        sim.observations[0].nl_2_self, sim.observations[0].nl_2_det
    )

    # Check if non-linearity is applied correctly
    sim.init_random(random_seed)
    sim.observations[0].tod_origin = np.ones_like(sim.observations[0].tod)
    for idx, tod in enumerate(sim.observations[0].nl_2_det):
        g_one_over_k = sim.random.normal(
            loc=nl_params.sampling_gaussian_loc,
            scale=nl_params.sampling_gaussian_scale,
        )

        np.testing.assert_array_equal(
            sim.observations[0].nl_2_self[idx],
            sim.observations[0].tod_origin[idx]
            * (1 + g_one_over_k * sim.observations[0].tod_origin[idx]),
        )
