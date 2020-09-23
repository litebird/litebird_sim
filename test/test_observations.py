# -*- encoding: utf-8 -*-

import numpy as np
import astropy.time as astrotime
import litebird_sim as lbs


def test_observation_time():
    ref_time = astrotime.Time("2020-02-20", format="iso")

    obs_no_mjd = lbs.Observation(
        detectors=1,
        start_time=0.0,
        sampling_rate_hz=5.0,
        n_samples=5,
    )
    obs_mjd_astropy = lbs.Observation(
        detectors=1,
        start_time=ref_time,
        sampling_rate_hz=5.0,
        n_samples=5,
    )

    plain_times = obs_no_mjd.get_times()
    assert np.allclose(plain_times, np.array([0.0, 0.2, 0.4, 0.6, 0.8]))

    assert isinstance(obs_mjd_astropy.get_times(astropy_times=True), astrotime.Time)
    assert np.allclose(
        (obs_mjd_astropy.get_times(astropy_times=True) - ref_time).jd,
        np.array([0.0, 2.31481681e-06, 4.62962635e-06, 6.94444316e-06, 9.25925997e-06]),
    )
    assert np.allclose(
        obs_mjd_astropy.get_times(normalize=False, astropy_times=False),
        np.array(
            [6.98544069e8, 6.98544069e8, 6.98544070e8, 6.98544070e8, 6.98544070e8]
        ),
    )
    assert np.allclose(
        obs_mjd_astropy.get_times(normalize=True), np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    )

def test_observation_tod_array():
    obs = lbs.Observation(
        detectors=3,
        n_samples=10,
        start_time=0.0,
        sampling_rate_hz=1.0,
    )

    assert obs.tod.shape == (3, 10)
    assert obs.tod.dtype == np.float32
