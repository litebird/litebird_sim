# -*- encoding: utf-8 -*-

import numpy as np
import pytest

import litebird_sim as lbs
from litebird_sim.maps_and_harmonics import HealpixMap, SphericalHarmonics

import healpy as hp

STRICT_TYPES_TEST_FIELDS = (
    "tod_dtype, pointings_dtype, single_precision, strict_typing, expect_error"
)

STRICT_TYPES_TEST_PARAMETERS = [
    # Single-precision calculations, strict typing
    (np.float32, np.float32, True, True, False),
    (np.float64, np.float32, True, True, False),
    (np.float32, np.float64, True, True, True),
    (np.float64, np.float64, True, True, True),
    # Double-precision calculations, strict typing
    (np.float32, np.float32, False, True, True),
    (np.float64, np.float32, False, True, True),
    (np.float32, np.float64, False, True, False),
    (np.float64, np.float64, False, True, False),
    # Single-precision calculations, no strict typing
    (np.float32, np.float32, True, False, False),
    (np.float64, np.float32, True, False, False),
    (np.float32, np.float64, True, False, False),
    (np.float64, np.float64, True, False, False),
    # Double-precision calculations, no strict typing
    (np.float32, np.float32, False, False, False),
    (np.float64, np.float32, False, False, False),
    (np.float32, np.float64, False, False, False),
    (np.float64, np.float64, False, False, False),
]


@pytest.mark.parametrize(STRICT_TYPES_TEST_FIELDS, STRICT_TYPES_TEST_PARAMETERS)
def test_beam_convolution_strict_types(
    tod_dtype,
    pointings_dtype,
    single_precision: bool,
    strict_typing: bool,
    expect_error: bool,
):
    rng = np.random.default_rng()

    num_of_samples = 100
    num_of_detectors = 2
    tod = rng.random((num_of_detectors, num_of_samples), dtype=tod_dtype)
    pointings = rng.uniform(
        low=0.0, high=np.pi, size=(num_of_detectors, num_of_samples, 3)
    ).astype(pointings_dtype)
    hwp_angle = rng.uniform(low=0.0, high=2.0 * np.pi, size=num_of_samples).astype(
        pointings_dtype
    )
    mueller_hwp = np.zeros((num_of_detectors, 4, 4), dtype=tod_dtype)

    # Assume an ideal HWP for both detectors
    for det_idx in range(num_of_detectors):
        mueller_hwp[det_idx, :, :] = np.identity(4)
    mueller_hwp[:, 2:4, 2:4] *= -1

    lmax = 10
    mmax_sky = 10
    sky_alms = lbs.SphericalHarmonics(
        values=rng.random((3, SphericalHarmonics.num_of_alm_from_lmax(lmax, mmax_sky))),
        lmax=lmax,
        mmax=mmax_sky,
    )
    mmax_beam = mmax_sky - 4
    beam_alms = lbs.SphericalHarmonics(
        values=rng.random(
            (3, SphericalHarmonics.num_of_alm_from_lmax(lmax, mmax_beam))
        ),
        lmax=lmax,
        mmax=mmax_beam,
    )

    convolution_params = lbs.BeamConvolutionParameters(
        lmax=lmax,
        mmax=mmax_beam,
        single_precision=single_precision,
        epsilon=1e-5,
        strict_typing=strict_typing,
    )

    arguments = {
        "tod": tod,
        "pointings": pointings,
        "sky_alms": sky_alms,
        "beam_alms": beam_alms,
        "hwp_angle": hwp_angle,
        "mueller_hwp": mueller_hwp,
        "convolution_params": convolution_params,
    }

    if expect_error:
        with pytest.raises(TypeError):
            lbs.add_convolved_sky(**arguments)
    else:
        lbs.add_convolved_sky(**arguments)


def test_beam_convolution():
    start_time = 0
    time_span_s = 3600
    nside_in = 256
    sampling_hz = 1

    net = 50.0

    tolerance = 1e-5

    lmax = 128
    mmax = lmax - 4

    fwhm_arcmin = 4.0 * 60

    npix = HealpixMap.nside_to_npix(nside_in)

    sim = lbs.Simulation(
        start_time=start_time, duration_s=time_span_s, random_seed=12345
    )

    scanning = lbs.SpinningScanningStrategy(
        spin_sun_angle_rad=0.785_398_163_397_448_3,
        precession_rate_hz=8.664_850_513_998_931e-05,
        spin_rate_hz=0.000_833_333_333_333_333_4,
        start_time=start_time,
    )

    spin2ecliptic_quats = scanning.generate_spin2ecl_quaternions(
        start_time, time_span_s, delta_time_s=60
    )

    instr = lbs.InstrumentInfo(
        boresight_rotangle_rad=0.0,
        spin_boresight_angle_rad=0.872_664_625_997_164_8,
        spin_rotangle_rad=3.141_592_653_589_793,
    )

    detT = lbs.DetectorInfo(
        name="Boresight_detector_T",
        sampling_rate_hz=sampling_hz,
        fwhm_arcmin=fwhm_arcmin,
        net_ukrts=net,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
        pol_angle_rad=0.0,
    )

    detB = lbs.DetectorInfo(
        name="Boresight_detector_B",
        sampling_rate_hz=sampling_hz,
        fwhm_arcmin=fwhm_arcmin,
        net_ukrts=net,
        bandcenter_ghz=100.0,
        quat=[0.0, 0.0, 0.0, 1.0],
        pol_angle_rad=np.pi / 2.0,
    )

    np.random.seed(seed=123_456_789)
    inmaps = np.random.normal(0, 1, (3, npix))

    maps = lbs.HealpixMap(
        hp.smoothing(inmaps, fwhm=np.deg2rad(fwhm_arcmin / 60.0), pol=True)
    )
    alms = lbs.SphericalHarmonics(values=hp.map2alm(inmaps, lmax=lmax), lmax=lmax)

    (obs1,) = sim.create_observations(detectors=[detT, detB], tod_dtype=np.float64)
    (obs2,) = sim.create_observations(detectors=[detT, detB], tod_dtype=np.float64)

    lbs.prepare_pointings(
        obs1,
        instr,
        spin2ecliptic_quats,
    )

    lbs.prepare_pointings(
        obs2,
        instr,
        spin2ecliptic_quats,
    )

    lbs.scan_map_in_observations(
        observations=obs1,
        maps=maps,
    )

    blms = lbs.generate_gauss_beam_alms(
        observation=obs2,
        lmax=lmax,
        mmax=mmax,
    )

    Convparams = lbs.BeamConvolutionParameters(
        lmax=lmax,
        mmax=mmax,
        single_precision=False,
        epsilon=1e-5,
    )

    lbs.add_convolved_sky_to_observations(
        observations=obs2,
        sky_alms=alms,
        beam_alms=blms,
        convolution_params=Convparams,
        pointings_dtype=np.float64,
    )

    np.testing.assert_allclose(
        obs1.tod,
        obs2.tod,
        rtol=tolerance,
        atol=0.1,
    )


test_beam_convolution()
