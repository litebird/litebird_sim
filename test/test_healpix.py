# -*- encoding: utf-8 -*-

import pytest
import litebird_sim as lbs


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


def test_num_of_alms():
    assert lbs.num_of_alms(lmax=5, mmax=4) == 20
    assert lbs.num_of_alms(lmax=5) == 21
    assert lbs.num_of_alms(lmax=8, mmax=2) == 24

    # Be sure that values too large for mmax are clipped
    assert lbs.num_of_alms(lmax=4, mmax=4) == 15
    assert lbs.num_of_alms(lmax=4, mmax=7) == 15
    assert lbs.num_of_alms(lmax=4, mmax=-1) == 15
    assert lbs.num_of_alms(lmax=4) == 15
