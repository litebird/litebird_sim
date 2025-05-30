# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs


def test_add_noise_to_observations():
    start_time = 0
    time_span_s = 10
    sampling_hz = 1

    # by setting random_seed=None here and calling sim.init_random below, we
    # are testing the change of random number generator and seed setting, too
    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=None
    )

    det1 = lbs.DetectorInfo(
        name="Boresight_detector_A",
        sampling_rate_hz=sampling_hz,
        net_ukrts=1.0,
        fknee_mhz=1e3,
    )

    det2 = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        net_ukrts=10.0,
        fknee_mhz=2e3,
    )

    sim.init_rng_hierarchy(random_seed=12_345)
    sim.create_observations(detectors=[det1, det2])

    lbs.noise.add_noise_to_observations(
        sim.observations, "white", dets_random=sim.dets_random
    )

    assert len(sim.observations) == 1

    tod = sim.observations[0].tod
    assert tod.shape == (2, 10)

    # fmt: off
    reference = np.array([
        [-4.3633665e-07,  1.1562618e-06,  7.8354390e-07,  5.6534816e-07,        -1.5919208e-07,  1.3325960e-06,  1.8658482e-06,  1.6043715e-06,        -3.1574638e-07, -1.6436400e-07],       [ 2.5071698e-05, -6.3309367e-06,  1.1325796e-05, -7.9862139e-07,         4.4909466e-06, -6.1964606e-06, -2.4171566e-06,  6.3236403e-06,        -1.1995277e-05,  4.1830776e-06]])
    # fmt: on

    assert np.allclose(tod, reference)


def test_add_noise_to_observations_in_other_field():
    start_time = 0
    time_span_s = 10
    sampling_hz = 1

    # by setting random_seed=None here and calling sim.init_random below, we
    # are testing the change of random number generator and seed setting, too
    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=None
    )

    det1 = lbs.DetectorInfo(
        name="Boresight_detector_A",
        sampling_rate_hz=sampling_hz,
        net_ukrts=1.0,
        fknee_mhz=1e3,
    )

    det2 = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        net_ukrts=10.0,
        fknee_mhz=2e3,
    )

    sim.create_observations(detectors=[det1, det2])

    for cur_obs in sim.observations:
        cur_obs.noise_tod = np.zeros_like(cur_obs.tod)

    sim.init_rng_hierarchy(random_seed=12_345)

    lbs.noise.add_noise_to_observations(
        sim.observations,
        "one_over_f",
        dets_random=sim.dets_random,
        component="noise_tod",
    )

    assert len(sim.observations) == 1

    # Check that the "tod" field has been left unchanged
    assert np.allclose(sim.observations[0].tod, 0.0)

    # Check that "noise_tod" has some non-zero data in it
    assert np.sum(np.abs(sim.observations[0].noise_tod)) > 0.0
