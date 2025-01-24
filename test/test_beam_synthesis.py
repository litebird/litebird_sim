# -*- encoding: utf-8 -*-

import numpy as np

from litebird_sim import gauss_beam_to_alm, alm_index


def test_gaussbeam_circular():
    # All the values used in this test have been computed using the Planck LevelS
    # beam functions, authored by Mark Ashdown
    lmax = 5
    mmax = 4

    alm = gauss_beam_to_alm(
        lmax=lmax,
        mmax=mmax,
        fwhm_min_rad=0.5,
        fwhm_max_rad=None,
        psi_ell_rad=0.1,
        psi_pol_rad=0.3,
        cross_polar_leakage=1e-9,
    )

    assert alm.values.shape == (3, 20)
    assert alm.lmax == lmax
    assert alm.mmax == mmax

    expected_alm = np.zeros((3, 20), dtype=np.complex128)
    expected_alm[0, alm_index(lmax, 0, 0)] = 0.28209479205597293
    expected_alm[0, alm_index(lmax, 1, 0)] = 0.46706343371176917
    expected_alm[0, alm_index(lmax, 2, 0)] = 0.5509860287336683
    expected_alm[0, alm_index(lmax, 3, 0)] = 0.5694624871151546
    expected_alm[0, alm_index(lmax, 4, 0)] = 0.5391604701572851
    expected_alm[0, alm_index(lmax, 5, 0)] = 0.4757666450251654
    expected_alm[1, alm_index(lmax, 2, 2)] = -0.2273741959610235 + 0.15555505672403316j
    expected_alm[1, alm_index(lmax, 3, 2)] = -0.23499883551559272 + 0.16077135329365197j
    expected_alm[1, alm_index(lmax, 4, 2)] = -0.22249416864113872 + 0.1522164504087869j
    expected_alm[1, alm_index(lmax, 5, 2)] = -0.1963335778700126 + 0.13431902733429554j
    expected_alm[2, alm_index(lmax, 2, 2)] = -0.15555505672403316 - 0.2273741959610235j
    expected_alm[2, alm_index(lmax, 3, 2)] = -0.16077135329365197 - 0.23499883551559272j
    expected_alm[2, alm_index(lmax, 4, 2)] = -0.1522164504087869 - 0.22249416864113872j
    expected_alm[2, alm_index(lmax, 5, 2)] = -0.13431902733429554 - 0.1963335778700126j

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
        fwhm_min_rad=0.5,
        fwhm_max_rad=0.8,
        psi_ell_rad=0.1,
        psi_pol_rad=0.3,
        cross_polar_leakage=1e-9,
    )

    assert alm.values.shape == (3, 20)
    assert alm.lmax == lmax
    assert alm.mmax == mmax

    expected_alm = np.zeros((3, 20), dtype=np.complex128)
    expected_alm[0, alm_index(lmax, 0, 0)] = 0.28209479205597293
    expected_alm[0, alm_index(lmax, 1, 0)] = 0.4510636776920152
    expected_alm[0, alm_index(lmax, 2, 0)] = 0.4972002452549269
    expected_alm[0, alm_index(lmax, 2, 2)] = 0.0256681071151606 - 0.005203182904754614j
    expected_alm[0, alm_index(lmax, 3, 0)] = 0.4662852536091051
    expected_alm[0, alm_index(lmax, 3, 2)] = 0.04794485997107865 - 0.009718904267195683j
    expected_alm[0, alm_index(lmax, 4, 0)] = 0.39112854390813984
    expected_alm[0, alm_index(lmax, 4, 2)] = 0.06637983560826029 - 0.013455858833210285j
    expected_alm[0, alm_index(lmax, 4, 4)] = (
        0.005456345526127067 - 0.002306905887538831j
    )
    expected_alm[0, alm_index(lmax, 5, 0)] = 0.30061451809145445
    expected_alm[0, alm_index(lmax, 5, 2)] = 0.07512150149738082 - 0.015227882235998861j
    expected_alm[0, alm_index(lmax, 5, 4)] = (
        0.009203862389390863 - 0.0038913306044336717j
    )
    expected_alm[1, alm_index(lmax, 2, 0)] = -0.024122740986080488
    expected_alm[1, alm_index(lmax, 2, 2)] = -0.205516866581687 + 0.14030160457109472j
    expected_alm[1, alm_index(lmax, 3, 0)] = -0.045058306539990926
    expected_alm[1, alm_index(lmax, 3, 2)] = -0.19368308440573004 + 0.13138637435273962j
    expected_alm[1, alm_index(lmax, 4, 0)] = -0.06238339172781836
    expected_alm[1, alm_index(lmax, 4, 2)] = -0.16430910498705525 + 0.10983543758200465j
    expected_alm[1, alm_index(lmax, 4, 4)] = (
        -0.023767101720368118 + 0.02429323699423821j
    )
    expected_alm[1, alm_index(lmax, 5, 0)] = -0.07059875958038433
    expected_alm[1, alm_index(lmax, 5, 2)] = -0.1289506767929808 + 0.08387724340550018j
    expected_alm[1, alm_index(lmax, 5, 4)] = (
        -0.027137797028227294 + 0.027492451924840226j
    )
    expected_alm[2, alm_index(lmax, 2, 0)] = -0.010198931306291946
    expected_alm[2, alm_index(lmax, 2, 2)] = -0.14043877112078462 - 0.20484020274831682j
    expected_alm[2, alm_index(lmax, 3, 0)] = -0.019050346452933527
    expected_alm[2, alm_index(lmax, 3, 2)] = -0.1318980840261679 - 0.19115874133537236j
    expected_alm[2, alm_index(lmax, 4, 0)] = -0.026375274984407938
    expected_alm[2, alm_index(lmax, 4, 2)] = -0.11101235042399053 - 0.1585032116624718j
    expected_alm[2, alm_index(lmax, 4, 4)] = -0.02429323699423821 - 0.02342079121136466j
    expected_alm[2, alm_index(lmax, 5, 0)] = -0.029848676801912326
    expected_alm[2, alm_index(lmax, 5, 2)] = -0.0858624812886358 - 0.11915719085059061j
    expected_alm[2, alm_index(lmax, 5, 4)] = (
        -0.027492451924840223 - 0.02626434440594484j
    )

    # We avoid implementing a `for` loop here so if anything fails we see
    # immediately which is the culprit
    np.testing.assert_allclose(alm.values[0, :], expected_alm[0, :])
    np.testing.assert_allclose(alm.values[1, :], expected_alm[1, :])
    np.testing.assert_allclose(alm.values[2, :], expected_alm[2, :])
