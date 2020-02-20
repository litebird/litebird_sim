# -*- encoding: utf-8 -*-

import numpy as np
import astropy.time as astrotime
import litebird_sim as lbs


def test_observation():
    ref_time = astrotime.Time("2020-02-20", format="iso")

    obs_no_mjd = lbs.Observation(
        detector="A", start_time=0.0, sampfreq_hz=5.0, nsamples=5, use_mjd=False,
    )
    obs_mjd = lbs.Observation(
        detector="B",
        start_time=float(ref_time.mjd),
        sampfreq_hz=5.0,
        nsamples=5,
        use_mjd=True,
    )
    obs_mjd_astropy = lbs.Observation(
        detector="B", start_time=ref_time, sampfreq_hz=5.0, nsamples=5, use_mjd=True,
    )

    assert np.allclose(obs_no_mjd.get_times(), np.array([0.0, 0.2, 0.4, 0.6, 0.8]))
    assert np.allclose(
        obs_mjd.get_times() - ref_time.mjd,
        np.array([0.0, 2.31481681e-06, 4.62962635e-06, 6.94444316e-06, 9.25925997e-06]),
    )
    assert np.allclose(
        obs_mjd_astropy.get_times() - ref_time.mjd,
        np.array([0.0, 2.31481681e-06, 4.62962635e-06, 6.94444316e-06, 9.25925997e-06]),
    )
