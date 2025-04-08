# -*- encoding: utf-8 -*-

import healpy
import numpy as np
import pytest

from litebird_sim import SphericalHarmonics


def test_constructor():
    nside = 16
    lmax = 10

    # Only temperature
    harmonics = SphericalHarmonics(
        values=healpy.map2alm(
            maps=np.random.rand(healpy.nside2npix(nside)), lmax=lmax, mmax=lmax
        ),
        lmax=lmax,
    )
    assert harmonics.nstokes == 1
    assert harmonics.values.shape[1] == SphericalHarmonics.num_of_alm_from_lmax(
        lmax=lmax
    )

    harmonics = SphericalHarmonics(
        values=healpy.map2alm(
            maps=np.random.rand(3, healpy.nside2npix(nside)), lmax=lmax, mmax=lmax
        ),
        lmax=lmax,
    )
    assert harmonics.nstokes == 3
    assert len(harmonics.values[0, :]) == SphericalHarmonics.num_of_alm_from_lmax(
        lmax=lmax
    )
    assert len(harmonics.values[1, :]) == SphericalHarmonics.num_of_alm_from_lmax(
        lmax=lmax
    )
    assert len(harmonics.values[2, :]) == SphericalHarmonics.num_of_alm_from_lmax(
        lmax=lmax
    )


def test_num_of_alm_coefficients():
    assert SphericalHarmonics.num_of_alm_from_lmax(lmax=4) == 15
    assert SphericalHarmonics.num_of_alm_from_lmax(lmax=4, mmax=3) == 14


def test_alm_array_size():
    assert SphericalHarmonics.alm_array_size(lmax=4) == (3, 15)
    assert SphericalHarmonics.alm_array_size(lmax=4, nstokes=1) == (1, 15)
    assert SphericalHarmonics.alm_array_size(lmax=4, mmax=3) == (3, 14)
    assert SphericalHarmonics.alm_array_size(lmax=4, mmax=3, nstokes=1) == (1, 14)


def test_spherical_harmonics():
    values = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        ]
    )

    harmonics = SphericalHarmonics(values, lmax=2)
    np.testing.assert_array_equal(harmonics.values, values)
    assert harmonics.lmax == 2
    assert harmonics.mmax == 2
    assert harmonics.nstokes == 1

    values = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            [-5.0, -3.0, 0.0, 3.0, 5.0, 7.0],
            [-6.0, -4.0, 2.0, 4.0, 6.0, 8.0],
        ]
    )

    harmonics = SphericalHarmonics(values, lmax=2)
    np.testing.assert_array_equal(harmonics.values, values)
    assert harmonics.lmax == 2
    assert harmonics.mmax == 2
    assert harmonics.nstokes == 3

    np.testing.assert_array_equal(harmonics.values, values)
    assert harmonics.lmax == 2
    assert harmonics.mmax == 2
    assert harmonics.nstokes == 3

    with pytest.raises(ValueError):
        _ = SphericalHarmonics(values[0:2, :], lmax=2)

    with pytest.raises(ValueError):
        _ = SphericalHarmonics(values, lmax=500)
