# -*- encoding: utf-8 -*-

import numpy as np
import healpy as hp

# Assuming gauss_beam_to_alm is exposed in the top-level package or imported correctly
from litebird_sim import gauss_beam_to_alm


def test_gaussbeam_circular():
    # All the values used in this test have been computed using the Planck LevelS
    # beam functions, authored by Mark Ashdown
    lmax = 5
    mmax = 4

    alm = gauss_beam_to_alm(
        lmax=lmax,
        mmax=mmax,
        fwhm_rad=0.5,
        ellipticity=1.0,
        psi_ell_rad=0.1,
        psi_pol_rad=0.3,
        cross_polar_leakage=1e-9,
    )

    # Check shape against (nstokes, nalm)
    # nalm for lmax=5, mmax=4 is 20
    assert alm.values.shape == (3, 20)
    assert alm.lmax == lmax
    assert alm.mmax == mmax

    expected_alm = np.zeros((3, 20), dtype=np.complex128)

    # Use hp.Alm.getidx(lmax, l, m) instead of custom alm_index
    expected_alm[0, hp.Alm.getidx(lmax, 0, 0)] = 0.28209479205597293
    expected_alm[0, hp.Alm.getidx(lmax, 1, 0)] = 0.46706343371176917
    expected_alm[0, hp.Alm.getidx(lmax, 2, 0)] = 0.5509860287336683
    expected_alm[0, hp.Alm.getidx(lmax, 3, 0)] = 0.5694624871151546
    expected_alm[0, hp.Alm.getidx(lmax, 4, 0)] = 0.5391604701572851
    expected_alm[0, hp.Alm.getidx(lmax, 5, 0)] = 0.4757666450251654
    expected_alm[1, hp.Alm.getidx(lmax, 2, 2)] = (
        -0.2488289072239636 + 0.17023301441135358j
    )
    expected_alm[1, hp.Alm.getidx(lmax, 3, 2)] = (
        -0.2571729971076956 + 0.17594151343292616j
    )
    expected_alm[1, hp.Alm.getidx(lmax, 4, 2)] = (
        -0.24348840734842728 + 0.1665793818715549j
    )
    expected_alm[1, hp.Alm.getidx(lmax, 5, 2)] = (
        -0.2148593398045074 + 0.14699318297625882j
    )
    expected_alm[2, hp.Alm.getidx(lmax, 2, 2)] = (
        -0.17023301441135358 - 0.2488289072239636j
    )
    expected_alm[2, hp.Alm.getidx(lmax, 3, 2)] = (
        -0.17594151343292616 - 0.2571729971076956j
    )
    expected_alm[2, hp.Alm.getidx(lmax, 4, 2)] = (
        -0.1665793818715549 - 0.24348840734842728j
    )
    expected_alm[2, hp.Alm.getidx(lmax, 5, 2)] = (
        -0.14699318297625882 - 0.2148593398045074j
    )

    # We avoid implementing a `for` loop here so if anything fails we see
    # immediately which is the culprit
    np.testing.assert_allclose(alm.values[0, :], expected_alm[0, :])
    np.testing.assert_allclose(alm.values[1, :], expected_alm[1, :])
    np.testing.assert_allclose(alm.values[2, :], expected_alm[2, :])


def test_gaussbeam_elliptical():
    # All the values used in this test have been computed using the Planck LevelS
    # beam functions, authored by Mark Ashdown
    lmax = 5
    mmax = 4

    alm = gauss_beam_to_alm(
        lmax=lmax,
        mmax=mmax,
        fwhm_rad=np.sqrt(0.8 * 0.5),
        ellipticity=1.6,
        psi_ell_rad=0.1,
        psi_pol_rad=0.3,
        cross_polar_leakage=1e-9,
    )

    assert alm.values.shape == (3, 20)
    assert alm.lmax == lmax
    assert alm.mmax == mmax

    expected_alm = np.zeros((3, 20), dtype=np.complex128)
    expected_alm[0, hp.Alm.getidx(lmax, 0, 0)] = 0.28209479205597293
    expected_alm[0, hp.Alm.getidx(lmax, 1, 0)] = 0.4510636776920152
    expected_alm[0, hp.Alm.getidx(lmax, 2, 0)] = 0.4972002452549269
    expected_alm[0, hp.Alm.getidx(lmax, 2, 2)] = (
        0.0256681071151606 - 0.005203182904754614j
    )
    expected_alm[0, hp.Alm.getidx(lmax, 3, 0)] = 0.4662852536091051
    expected_alm[0, hp.Alm.getidx(lmax, 3, 2)] = (
        0.04794485997107865 - 0.009718904267195683j
    )
    expected_alm[0, hp.Alm.getidx(lmax, 4, 0)] = 0.39112854390813984
    expected_alm[0, hp.Alm.getidx(lmax, 4, 2)] = (
        0.06637983560826029 - 0.013455858833210285j
    )
    expected_alm[0, hp.Alm.getidx(lmax, 4, 4)] = (
        0.005456345526127067 - 0.002306905887538831j
    )
    expected_alm[0, hp.Alm.getidx(lmax, 5, 0)] = 0.30061451809145445
    expected_alm[0, hp.Alm.getidx(lmax, 5, 2)] = (
        0.07512150149738082 - 0.015227882235998861j
    )
    expected_alm[0, hp.Alm.getidx(lmax, 5, 4)] = (
        0.009203862389390863 - 0.0038913306044336717j
    )
    expected_alm[1, hp.Alm.getidx(lmax, 2, 0)] = -0.027866479337096971
    expected_alm[1, hp.Alm.getidx(lmax, 2, 2)] = (
        -0.237412138169877 + 0.16207576771636262j
    )
    expected_alm[1, hp.Alm.getidx(lmax, 3, 0)] = -0.05205114828724336
    expected_alm[1, hp.Alm.getidx(lmax, 3, 2)] = (
        -0.22374180747751102 + 0.15177693480974516j
    )
    expected_alm[1, hp.Alm.getidx(lmax, 4, 0)] = -0.072065006939484449
    expected_alm[1, hp.Alm.getidx(lmax, 4, 2)] = (
        -0.18980912167737168 + 0.12688139186280961j
    )
    expected_alm[1, hp.Alm.getidx(lmax, 4, 4)] = (
        -0.027455646494545925 + 0.028063435540835433j
    )
    expected_alm[1, hp.Alm.getidx(lmax, 5, 0)] = -0.081555362062987274
    expected_alm[1, hp.Alm.getidx(lmax, 5, 2)] = (
        -0.14896322820154498 + 0.096894605449718577j
    )
    expected_alm[1, hp.Alm.getidx(lmax, 5, 4)] = (
        -0.031349458197051351 + 0.031759153900950313j
    )

    expected_alm[2, hp.Alm.getidx(lmax, 2, 0)] = -0.011781758493831703
    expected_alm[2, hp.Alm.getidx(lmax, 2, 2)] = (
        -0.16223422188311257 - 0.23663045922461737j
    )
    expected_alm[2, hp.Alm.getidx(lmax, 3, 0)] = -0.022006872523380971
    expected_alm[2, hp.Alm.getidx(lmax, 3, 2)] = (
        -0.15236805946881316 - 0.22082569798355023j
    )
    expected_alm[2, hp.Alm.getidx(lmax, 4, 0)] = -0.030468596242332596
    expected_alm[2, hp.Alm.getidx(lmax, 4, 2)] = (
        -0.12824095615990538 - 0.18310218043647988j
    )
    expected_alm[2, hp.Alm.getidx(lmax, 4, 4)] = (
        -0.028063435540835433 - 0.027055590188799702j
    )
    expected_alm[2, hp.Alm.getidx(lmax, 5, 0)] = -0.034481054031966558
    expected_alm[2, hp.Alm.getidx(lmax, 5, 2)] = (
        -0.099187943113193178 - 0.13764983832560829j
    )
    expected_alm[2, hp.Alm.getidx(lmax, 5, 4)] = (
        -0.031759153900950306 - 0.030340449748765472j
    )

    # We avoid implementing a `for` loop here so if anything fails we see
    # immediately which is the culprit
    np.testing.assert_allclose(alm.values[0, :], expected_alm[0, :])
    np.testing.assert_allclose(alm.values[1, :], expected_alm[1, :])
    np.testing.assert_allclose(alm.values[2, :], expected_alm[2, :])
