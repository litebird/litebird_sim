# -*- encoding: utf-8 -*-

import astropy.time as astrotime
import numpy as np
import numpy.testing as nptest

import litebird_sim as lbs


def test_observation_time():
    ref_time = astrotime.Time("2020-02-20", format="iso")

    obs_no_mjd = lbs.Observation(
        detectors=1, start_time_global=0.0, sampling_rate_hz=5.0, n_samples_global=5
    )
    obs_mjd_astropy = lbs.Observation(
        detectors=1,
        start_time_global=ref_time,
        sampling_rate_hz=5.0,
        n_samples_global=5,
    )

    assert isinstance(obs_no_mjd.get_delta_time(), float)
    nptest.assert_allclose(obs_no_mjd.get_delta_time(), 0.2)
    assert isinstance(obs_no_mjd.get_time_span(), float)
    nptest.assert_allclose(obs_no_mjd.get_time_span(), 1.0)

    assert isinstance(obs_mjd_astropy.get_delta_time(), astrotime.TimeDelta)
    nptest.assert_allclose(obs_mjd_astropy.get_delta_time().to("ms").value, 200.0)
    assert isinstance(obs_mjd_astropy.get_time_span(), astrotime.TimeDelta)
    nptest.assert_allclose(obs_mjd_astropy.get_time_span().to("ms").value, 1000.0)

    plain_times = obs_no_mjd.get_times()
    nptest.assert_allclose(plain_times, np.array([0.0, 0.2, 0.4, 0.6, 0.8]))

    assert isinstance(obs_mjd_astropy.get_times(astropy_times=True), astrotime.Time)
    nptest.assert_allclose(
        (obs_mjd_astropy.get_times(astropy_times=True) - ref_time).jd,
        np.array([0.0, 2.314815e-06, 4.629630e-06, 6.944444e-06, 9.259259e-06]),
    )
    nptest.assert_allclose(
        obs_mjd_astropy.get_times(normalize=False, astropy_times=False),
        np.array(
            [6.98544069e8, 6.98544069e8, 6.98544070e8, 6.98544070e8, 6.98544070e8]
        ),
    )
    nptest.assert_allclose(
        obs_mjd_astropy.get_times(normalize=True), np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    )


def test_observation_tod_array():
    obs = lbs.Observation(
        detectors=3, n_samples_global=10, start_time_global=0.0, sampling_rate_hz=1.0
    )

    assert obs.tod.shape == (3, 10)
    assert obs.tod.dtype == np.float32
