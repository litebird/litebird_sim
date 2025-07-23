# -*- encoding: utf-8 -*-

import pytest
import litebird_sim as lbs
import numpy as np
import numpy.testing as npt


def test_nside_to_npix():
    assert lbs.nside_to_npix(1) == 12
    assert lbs.nside_to_npix(32) == 12288
    assert lbs.nside_to_npix(2048) == 50331648

    with pytest.raises(AssertionError):
        assert lbs.nside_to_npix(123) == 1


def test_npix_to_nside():
    assert lbs.npix_to_nside(12) == 1
    assert lbs.npix_to_nside(12288) == 32
    assert lbs.npix_to_nside(50331648) == 2048

    with pytest.raises(AssertionError):
        assert lbs.npix_to_nside(123) == 1


def test_nside_to_pixel_solid_angle_sterad():
    actual = np.array([lbs.nside_to_pixel_solid_angle_sterad(2**i) for i in range(12)])
    npt.assert_almost_equal(
        actual,
        [
            1.0471975511965976,
            0.2617993877991494,
            0.06544984694978735,
            0.016362461737446838,
            0.0040906154343617095,
            0.0010226538585904274,
            0.00025566346464760684,
            6.391586616190171e-05,
            1.5978966540475428e-05,
            3.994741635118857e-06,
            9.986854087797142e-07,
            2.4967135219492856e-07,
        ],
    )


def test_nside_to_resolution_rad():
    actual = np.array([lbs.nside_to_resolution_rad(2**i) for i in range(12)])
    npt.assert_almost_equal(
        actual,
        [
            1.0233267079464885,
            0.5116633539732443,
            0.2558316769866221,
            0.12791583849331106,
            0.06395791924665553,
            0.031978959623327766,
            0.015989479811663883,
            0.007994739905831941,
            0.003997369952915971,
            0.0019986849764579854,
            0.0009993424882289927,
            0.0004996712441144963,
        ],
    )


def test_num_of_alms():
    assert lbs.num_of_alms(lmax=5, mmax=4) == 20
    assert lbs.num_of_alms(lmax=5) == 21
    assert lbs.num_of_alms(lmax=8, mmax=2) == 24

    # Be sure that values too large for mmax are clipped
    assert lbs.num_of_alms(lmax=4, mmax=4) == 15
    assert lbs.num_of_alms(lmax=4, mmax=7) == 15
    assert lbs.num_of_alms(lmax=4, mmax=-1) == 15
    assert lbs.num_of_alms(lmax=4) == 15
