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
    expected_alm[0, alm_index(lmax, 0, 0)] = 0.14104738845624820
    expected_alm[0, alm_index(lmax, 1, 0)] = 0.23885579431100701
    expected_alm[0, alm_index(lmax, 2, 0)] = 0.28819763689250677
    expected_alm[0, alm_index(lmax, 3, 0)] = 0.30465257198281420
    expected_alm[0, alm_index(lmax, 4, 0)] = 0.29501745134291479
    expected_alm[0, alm_index(lmax, 5, 0)] = 0.26626467731399417
    expected_alm[1, alm_index(lmax, 2, 2)] = (
        -0.11892988803804101 + 8.1364315578558929e-2j
    )
    expected_alm[1, alm_index(lmax, 3, 2)] = (
        -0.12572030804212322 + 8.6009891936508431e-2j
    )
    expected_alm[1, alm_index(lmax, 4, 2)] = (
        -0.12174420100967033 + 8.3289690709554068e-2j
    )
    expected_alm[1, alm_index(lmax, 5, 2)] = (
        -0.10987885868853610 + 7.5172173128476391e-2j
    )
    expected_alm[2, alm_index(lmax, 2, 2)] = (
        -8.1364315578558929e-2 - 0.11892988803804101j
    )
    expected_alm[2, alm_index(lmax, 3, 2)] = (
        -8.6009891936508431e-2 - 0.12572030804212322j
    )
    expected_alm[2, alm_index(lmax, 4, 2)] = (
        -8.3289690709554068e-2 - 0.12174420100967033j
    )
    expected_alm[2, alm_index(lmax, 5, 2)] = (
        -7.5172173128476391e-2 - 0.10987885868853610j
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
    expected_alm[0, alm_index(lmax, 0, 0)] = 0.14104738845624820
    expected_alm[0, alm_index(lmax, 1, 0)] = 0.23471087240802022
    expected_alm[0, alm_index(lmax, 2, 0)] = 0.26895695951654608
    expected_alm[0, alm_index(lmax, 2, 2)] = (
        9.2637996780089316e-3 - 1.8778651098129214e-3j
    )
    expected_alm[0, alm_index(lmax, 3, 0)] = 0.26169431235733642
    expected_alm[0, alm_index(lmax, 3, 2)] = (
        2.0229990053945386e-2 - 4.1008218889218764e-3j
    )
    expected_alm[0, alm_index(lmax, 4, 0)] = 0.22710169874490110
    expected_alm[0, alm_index(lmax, 4, 2)] = (
        3.1002286867586749e-2 - 6.2844744982185636e-3j
    )
    expected_alm[0, alm_index(lmax, 4, 4)] = (
        2.0424351433163258e-3 - 8.6352772309211310e-4j
    )
    expected_alm[0, alm_index(lmax, 5, 0)] = 0.17994552868717820
    expected_alm[0, alm_index(lmax, 5, 2)] = (
        3.7854070200179411e-2 - 7.6733997025078375e-3j
    )
    expected_alm[0, alm_index(lmax, 5, 4)] = (
        3.8783408938473643e-3 - 1.6397362199717859e-3j
    )
    expected_alm[1, alm_index(lmax, 2, 0)] = -8.7060657986679571e-3
    expected_alm[1, alm_index(lmax, 2, 2)] = (
        -0.11107129850744472 + 7.5915755217188616e-2j
    )
    expected_alm[1, alm_index(lmax, 3, 0)] = -1.9012028924149264e-2
    expected_alm[1, alm_index(lmax, 3, 2)] = (
        -0.10839256203544269 + 7.3800830635644049e-2j
    )
    expected_alm[1, alm_index(lmax, 4, 0)] = -2.9135772408683529e-2
    expected_alm[1, alm_index(lmax, 4, 2)] = (
        -9.4804197751842290e-2 + 6.3895361228901132e-2j
    )
    expected_alm[1, alm_index(lmax, 4, 4)] = (
        -1.1071305198455381e-2 + 1.1346004296964120e-2j
    )
    expected_alm[1, alm_index(lmax, 5, 0)] = -3.5575037517864477e-2
    expected_alm[1, alm_index(lmax, 5, 2)] = (
        -7.6321126778235424e-2 + 5.0384175545691287e-2j
    )
    expected_alm[1, alm_index(lmax, 5, 4)] = (
        -1.3608417120453785e-2 + 1.3853572261906777e-2j
    )
    expected_alm[2, alm_index(lmax, 2, 0)] = -3.6808657956535393e-3
    expected_alm[2, alm_index(lmax, 2, 2)] = (
        -7.5948766633373369e-2 - 0.11090844810438716j
    )
    expected_alm[2, alm_index(lmax, 3, 0)] = -8.0381573711037126e-3
    expected_alm[2, alm_index(lmax, 3, 2)] = (
        -7.3962895926818123e-2 - 0.10759306898002143j
    )
    expected_alm[2, alm_index(lmax, 4, 0)] = -1.2318407713559661e-2
    expected_alm[2, alm_index(lmax, 4, 2)] = (
        -6.4335906730237452e-2 - 9.2630918906236925e-2j
    )
    expected_alm[2, alm_index(lmax, 4, 4)] = (
        -1.1346004299813440e-2 - 1.0967503849404401e-2j
    )
    expected_alm[2, alm_index(lmax, 5, 0)] = -1.5040885493724827e-2
    expected_alm[2, alm_index(lmax, 5, 2)] = (
        -5.1220718932651017e-2 - 7.2194329340068503e-2j
    )
    expected_alm[2, alm_index(lmax, 5, 4)] = (
        -1.3853572270340774e-2 - 1.3301164946881163e-2j
    )

    # We avoid implementing a `for` loop here so if anything fails we see
    # immediately which is the culprit
    np.testing.assert_allclose(alm.values[0, :], expected_alm[0, :])
    np.testing.assert_allclose(alm.values[1, :], expected_alm[1, :])
    np.testing.assert_allclose(alm.values[2, :], expected_alm[2, :])
