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

    np.random.seed(seed=123_456_789)
    sim.create_observations(detectors=[det1, det2])

    random = np.random.default_rng(1234567890)
    lbs.noise.add_noise_to_observations(sim.observations, "white", random=random)

    assert len(sim.observations) == 1

    tod = sim.observations[0].tod
    assert tod.shape == (2, 10)

    # fmt: off
    reference = np.array(
        [
            [-2.0393272e-06, +1.8250503e-08, -7.3260622e-07, +8.0924798e-07,
             +5.9536052e-07, +2.7355711e-07, -5.1849185e-07, +5.6577466e-07,
             -2.4610816e-07, +3.4414407e-09],
            [-1.4176262e-05, -1.0636459e-06, -2.9512157e-06, +1.6421784e-06,
             -7.8052644e-06, -6.4513415e-06, +2.3054397e-05, +2.9360060e-06,
             +1.1997460e-06, +1.5504318e-05],
        ]
    )
    # fmt: on

    assert np.allclose(tod, reference)
