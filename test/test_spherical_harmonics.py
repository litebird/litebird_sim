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


def test_arithmetic_operations():
    lmax = 2
    v1 = np.ones((3, SphericalHarmonics.num_of_alm_from_lmax(lmax)))
    v2 = 2 * np.ones_like(v1)

    sh1 = SphericalHarmonics(v1, lmax)
    sh2 = SphericalHarmonics(v2, lmax)

    sh_sum = sh1 + sh2
    np.testing.assert_array_equal(sh_sum.values, 3 * np.ones_like(v1))

    sh_scaled = sh1 * 2.0
    np.testing.assert_array_equal(sh_scaled.values, 2.0 * np.ones_like(v1))

    sh_scaled_vec = sh1 * np.array([1.0, 0.5, 0.0])
    expected = v1 * np.array([[1.0], [0.5], [0.0]])
    np.testing.assert_array_equal(sh_scaled_vec.values, expected)


def test_convolution():
    lmax = 3
    nalm = SphericalHarmonics.num_of_alm_from_lmax(lmax)
    values = np.ones((3, nalm))
    sh = SphericalHarmonics(values, lmax)

    # Scalar filter
    f_ell = np.arange(lmax + 1)
    sh_conv = sh.convolve(f_ell)

    l_arr = SphericalHarmonics.alm_l_array(lmax)
    kernel = f_ell[l_arr]

    expected = values * kernel
    np.testing.assert_array_equal(sh_conv.values, expected)

    # Vector filter (per Stokes)
    f_ell_vec = np.stack([f_ell, f_ell**2, np.ones_like(f_ell)])
    sh_conv_vec = sh.convolve(f_ell_vec)

    kernel_vec = np.stack([f[l_arr] for f in f_ell_vec])
    expected_vec = values * kernel_vec

    np.testing.assert_array_equal(sh_conv_vec.values, expected_vec)


def test_healpy_io(tmp_path):
    nside = 16

    lmax = 10
    mmax = 4

    nstokes = 3

    sh = SphericalHarmonics(
        values=healpy.map2alm(
            maps=np.random.rand(nstokes, healpy.nside2npix(nside)),
            lmax=lmax,
            mmax=mmax,
        ),
        lmax=lmax,
        mmax=mmax,
    )
    file = tmp_path / "test_alm.fits.gz"
    sh.write_fits(str(file))

    sh_loaded = SphericalHarmonics.read_fits(str(file))
    np.testing.assert_allclose(sh_loaded.values, sh.values)
    assert sh_loaded.lmax == sh.lmax
    assert sh_loaded.mmax == sh.mmax
