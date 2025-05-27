from copy import deepcopy
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
    RNG_hierarchy = lbs.RNGHierarchy(
        random_seed, num_ranks=1, num_detectors_per_rank=len(dets)
    )
    dets_random = RNG_hierarchy.get_detector_level_generators_on_rank(0)
    lbs.apply_quadratic_nonlin_to_observations(
        observations=sim.observations,
        nl_params=nl_params,
        component="nl_2_obs",
        dets_random=dets_random,
    )

    # Applying non-linearity on the TOD arrays of the individual detectors.
    RNG_hierarchy = lbs.RNGHierarchy(
        random_seed, num_ranks=1, num_detectors_per_rank=len(dets)
    )
    dets_random = RNG_hierarchy.get_detector_level_generators_on_rank(0)
    for idx, tod in enumerate(sim.observations[0].nl_2_det):
        lbs.apply_quadratic_nonlin_for_one_detector(
            tod_det=tod,
            nl_params=nl_params,
            random=dets_random[idx],
        )

    # Check if the three non-linear tods are equal

    np.testing.assert_array_equal(
        sim.observations[0].nl_2_self, sim.observations[0].nl_2_obs
    )
    np.testing.assert_array_equal(
        sim.observations[0].nl_2_self, sim.observations[0].nl_2_det
    )

    # Check if non-linearity is applied correctly
    RNG_hierarchy = lbs.RNGHierarchy(
        random_seed, num_ranks=1, num_detectors_per_rank=len(dets)
    )
    dets_random = RNG_hierarchy.get_detector_level_generators_on_rank(0)
    sim.observations[0].tod_origin = np.ones_like(sim.observations[0].tod)
    for idx, tod in enumerate(sim.observations[0].nl_2_det):
        g_one_over_k = dets_random[idx].normal(
            loc=nl_params.sampling_gaussian_loc,
            scale=nl_params.sampling_gaussian_scale,
        )

        _tod = deepcopy(sim.observations[0].tod_origin[idx])
        for i in range(len(_tod)):
            _tod[i] += g_one_over_k * _tod[i] ** 2

        np.testing.assert_array_equal(
            sim.observations[0].nl_2_self[idx],
            _tod,
        )
