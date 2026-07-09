from copy import deepcopy

import numpy as np
import pytest
from astropy.time import Time

import litebird_sim as lbs
from litebird_sim.observations import TodDescription
from litebird_sim.units import Units, UnitUtils


@pytest.mark.parametrize(
    "automatic_unit_conversion",
    [
        False,
        True,
    ],
)
def test_add_quadratic_nonlinearity(automatic_unit_conversion):
    # Test function to check consistency of wrappers and low level functions
    start_time = Time("2025-02-02T00:00:00")
    mission_time_days = 1
    sampling_hz = 1

    dets = [
        lbs.DetectorInfo(
            name="det_A", sampling_rate_hz=sampling_hz, bandcenter_ghz=100
        ),
        lbs.DetectorInfo(
            name="det_B", sampling_rate_hz=sampling_hz, bandcenter_ghz=100
        ),
    ]

    random_seed = 12345

    nl_params = lbs.NonLinParams(
        sampling_gaussian_loc=0.0,
        sampling_gaussian_scale=0.1,
        user_seed=random_seed,
        units=Units.K_CMB,
    )

    sim = lbs.Simulation(
        base_path="nonlin_example",
        start_time=start_time,
        duration_s=mission_time_days * 24 * 3600.0,
        random_seed=random_seed,
    )

    if automatic_unit_conversion:
        custom_tod = TodDescription(
            name="tod",
            units=Units.MJy_over_sr,
            dtype=np.float32,
            description="custom tod",
        )
        sim.create_observations(
            detectors=dets,
            split_list_over_processes=False,
            tods=[custom_tod],
        )

    else:
        sim.create_observations(
            detectors=dets,
            split_list_over_processes=False,
        )

    if automatic_unit_conversion:
        assert nl_params.units != sim.tod_list[0].units, "error"

    # Creating fiducial TODs
    sim.observations[0].nl_2_self = np.ones_like(sim.observations[0].tod)
    sim.observations[0].nl_2_obs = np.ones_like(sim.observations[0].tod)
    sim.observations[0].nl_2_det = np.ones_like(sim.observations[0].tod)

    # Applying non-linearity using the `Simulation` class method
    sim.apply_quadratic_nonlin(
        nl_params=nl_params,
        component="nl_2_self",
        user_seed=random_seed,
    )

    # Applying non-linearity on the given TOD component of an `Observation` object
    lbs.apply_quadratic_nonlin_to_observations(
        observations=sim.observations,
        nl_params=nl_params,
        component="nl_2_obs",
    )

    # Applying non-linearity on the TOD arrays of the individual detectors.
    RNG_hierarchy = lbs.RNGHierarchy(
        random_seed, comm=lbs.MPI_COMM_WORLD, num_detectors_per_rank=len(dets)
    )
    dets_random = RNG_hierarchy.get_detector_level_generators_on_rank(0)
    for idx, tod in enumerate(sim.observations[0].nl_2_det):
        lbs.apply_quadratic_nonlin_for_one_detector(
            tod_det=tod,
            nl_params=nl_params,
            random=dets_random[idx],
            tod_units=sim.observations[0].tod_list[0].units,
            bandcenter_ghz_det=dets[0].bandcenter_ghz,
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
        random_seed, comm=lbs.MPI_COMM_WORLD, num_detectors_per_rank=len(dets)
    )
    dets_random = RNG_hierarchy.get_detector_level_generators_on_rank(0)
    sim.observations[0].tod_origin = np.ones_like(sim.observations[0].tod)
    for idx, tod in enumerate(sim.observations[0].nl_2_det):
        g_nonlin = dets_random[idx].normal(
            loc=nl_params.sampling_gaussian_loc,
            scale=nl_params.sampling_gaussian_scale,
        )

        _tod = deepcopy(sim.observations[0].tod_origin[idx])

        if nl_params.units != sim.observations[0].tod_list[0].units:
            conv_factor_nl = UnitUtils.get_conversion_factor(
                nl_params.units,
                sim.observations[0].tod_list[0].units,
                dets[idx].bandcenter_ghz,
            )
        else:
            conv_factor_nl = 1

        g_nonlin = (1 / conv_factor_nl) * g_nonlin

        for i in range(len(_tod)):
            _tod[i] += g_nonlin * _tod[i] ** 2

        np.testing.assert_array_equal(
            sim.observations[0].nl_2_self[idx],
            _tod,
        )
