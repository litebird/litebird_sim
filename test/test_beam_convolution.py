# -*- encoding: utf-8 -*-

import numpy as np
import pytest

import litebird_sim as lbs

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


def num_of_alms(lmax: int, mmax: int) -> int:
    return mmax * (2 * lmax + 1 - mmax) // 2 + lmax + 1


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
        values=rng.random((3, num_of_alms(lmax, mmax_sky))),
        lmax=lmax,
        mmax=mmax_sky,
    )
    mmax_beam = mmax_sky - 4
    beam_alms = lbs.SphericalHarmonics(
        values=rng.random((3, num_of_alms(lmax, mmax_beam))),
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
