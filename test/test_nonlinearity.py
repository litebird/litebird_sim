from copy import deepcopy

import litebird_sim as lbs
import numpy as np
from astropy.time import Time
from numpy.random import PCG64, Generator, SeedSequence


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

    lbs.apply_quadratic_nonlin_to_observations(
        observations=sim.observations,
        nl_params=nl_params,
        component="nl_2_obs",
    )

    # Applying non-linearity on the TOD arrays of the individual detectors.
    seeds = [sum(ord(c) for c in dn) for dn in sim.observations[0].name]
    sg = SeedSequence(seeds)
    dets_random = [
        Generator(PCG64(s)) for s in sg.spawn(sim.observations[0].n_detectors)
    ]
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
    seeds = [sum(ord(c) for c in dn) for dn in sim.observations[0].name]
    sg = SeedSequence(seeds)
    dets_random = [
        Generator(PCG64(s)) for s in sg.spawn(sim.observations[0].n_detectors)
    ]
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
