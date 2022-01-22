# -*- encoding: utf-8 -*-

import numpy as np
import litebird_sim as lbs


def test_add_noise_to_observations():
    start_time = 0
    time_span_s = 10
    sampling_hz = 1

    sim = lbs.Simulation(start_time=start_time, duration_s=time_span_s)

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

    sim.init_random(seed=12_345)
    lbs.noise.add_noise_to_observations(sim.observations, "white", random=sim.random)

    assert len(sim.observations) == 1

    tod = sim.observations[0].tod
    assert tod.shape == (2, 10)

    print(repr(tod))
    # fmt: off
    reference = np.array([
        [+1.7875197e-06, -4.8864092e-07, +1.0823729e-06, -4.4991100e-07,
         -5.4109887e-07, +2.7580990e-07, -3.9022507e-07, -2.2114153e-07,
         +2.0820102e-07, +1.7433838e-06],
        [+1.0560724e-05, -1.1145157e-05, +8.1773060e-06, +4.4965477e-06,
         +4.2855218e-06, +9.2935315e-06, -7.5876560e-06, +3.3198667e-06,
         +5.1126663e-06, -9.2129788e-08],
    ])
    # fmt: on

    assert np.allclose(tod, reference)
