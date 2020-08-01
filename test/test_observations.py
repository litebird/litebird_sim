# -*- encoding: utf-8 -*-

import numpy as np
import astropy.time as astrotime
import litebird_sim as lbs


def test_observation():
    ref_time = astrotime.Time("2020-02-20")

    obs_no_mjd = lbs.Observation(
        detector="A", start_time=0.0, sampling_rate_hz=5.0, nsamples=5, use_mjd=False,
    )
    obs_mjd = lbs.Observation(
        detector="B",
        start_time=float(ref_time.mjd),
        sampling_rate_hz=5.0,
        nsamples=5,
        use_mjd=True,
    )
    obs_mjd_astropy = lbs.Observation(
        detector="B",
        start_time=ref_time,
        sampling_rate_hz=5.0,
        nsamples=5,
        use_mjd=True,
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


def test_pointing_generation():
    obs = lbs.Observation(
        detector="A", start_time=0.0, sampling_rate_hz=1.0, nsamples=100, use_mjd=False
    )

    sstr = lbs.ScanningStrategy(
        spin_sun_angle_deg=45.0,
        spin_boresight_angle_deg=50.0,
        precession_period_min=1.0,
        spin_rate_rpm=1.0,
    )
    # We're simulating 100 seconds, but we're recalculating the
    # quaternions once every 60 s
    obs.generate_pointing_information(sstr, delta_time_s=60.0)

    assert obs.bore2ecliptic_quat.shape == (2, 4)
    assert len(obs.pointing_time_s) == 2
    assert obs.pointing_time_s[0] == 0.0
    assert obs.pointing_time_s[1] == 60.0


def test_pointing_generation_mjd():
    ref_time = astrotime.Time("2020-02-20", scale="tdb")
    obs = lbs.Observation(
        detector="A",
        start_time=ref_time,
        sampling_rate_hz=1.0,
        nsamples=100,
        use_mjd=True,
    )

    sstr = lbs.ScanningStrategy(
        spin_sun_angle_deg=45.0,
        spin_boresight_angle_deg=50.0,
        precession_period_min=1.0,
        spin_rate_rpm=1.0,
    )
    # We're simulating 100 seconds, but we're recalculating the
    # quaternions once every 60 s
    obs.generate_pointing_information(sstr, delta_time_s=60.0)

    assert obs.bore2ecliptic_quat.shape == (2, 4)
    assert len(obs.pointing_time_s) == 2
    assert np.isclose(obs.pointing_time_s[1] - obs.pointing_time_s[0], 60.0)
