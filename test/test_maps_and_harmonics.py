# -*- encoding: utf-8 -*-

import healpy
import numpy as np
import numpy.testing as npt
import pytest

from litebird_sim import (
    SphericalHarmonics,
    HealpixMap,
    synthesize_alm,
    compute_cl,
    rotate_alm,
)
from litebird_sim import Units, CoordinateSystem


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
    """Test the convolution of SphericalHarmonics with a function f(ell)."""
    lmax = 3
    nalm = SphericalHarmonics.num_of_alm_from_lmax(lmax)

    # FIX 1: Usa complex128 come standard per SH.
    values = np.ones((3, nalm), dtype=np.complex128)

    sh = SphericalHarmonics(values, lmax=lmax)

    # -------------------------------------------------------------------------
    # 1. Scalar Filter (Apply same f_ell to all Stokes)
    # -------------------------------------------------------------------------
    f_ell = np.arange(lmax + 1, dtype=np.float64)
    sh_conv = sh.convolve(f_ell)

    l_arr = SphericalHarmonics.alm_l_array(sh.lmax, mmax=sh.mmax)
    kernel = f_ell[l_arr]  # Broadcast da l a (nstokes, alm)

    expected = np.ones((3, nalm), dtype=np.complex128) * kernel

    # FIX 4: assert_allclose è preferibile per operazioni float/complex
    np.testing.assert_allclose(
        sh_conv.values, expected, err_msg="Scalar convolution mismatch"
    )

    # -------------------------------------------------------------------------
    # 2. Vector Filter (Apply different f_ell per Stokes parameter)
    # -------------------------------------------------------------------------
    values = np.ones((3, nalm), dtype=np.complex128)
    sh = SphericalHarmonics(values, lmax=lmax)

    f_ell_vec = np.stack([f_ell, f_ell**2, np.ones_like(f_ell)])
    sh_conv_vec = sh.convolve(f_ell_vec)

    # kernel_vec shape: (nstokes, nalm)
    kernel_vec = np.stack([f[l_arr] for f in f_ell_vec])
    expected_vec = np.ones((3, nalm), dtype=np.complex128) * kernel_vec

    np.testing.assert_allclose(
        sh_conv_vec.values, expected_vec, err_msg="Vector convolution mismatch"
    )


def test_io_roundtrip_internal(tmp_path):
    """
    Test writing and reading back using ONLY litebird_sim (no healpy involved).
    """
    filename = tmp_path / "test_io_internal.fits"
    lmax = 10
    mmax = 4

    nstokes = 3

    # Create random complex data
    # (Using fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    nalm = SphericalHarmonics.num_of_alm_from_lmax(lmax, mmax)
    # T, E, B components
    values = rng.standard_normal((nstokes, nalm)) + 1j * rng.standard_normal(
        (nstokes, nalm)
    )

    sh_orig = SphericalHarmonics(values=values, lmax=lmax, mmax=mmax)

    # Write using new Astropy implementation
    sh_orig.write_fits(str(filename))

    # Read back using new Astropy implementation
    sh_loaded = SphericalHarmonics.read_fits(str(filename))

    assert sh_loaded.lmax == lmax
    assert sh_loaded.mmax == mmax
    assert sh_loaded.nstokes == nstokes
    # Use strict equality or strict close check
    np.testing.assert_allclose(sh_loaded.values, sh_orig.values, rtol=1e-10)


def test_nside_to_npix():
    assert HealpixMap.nside_to_npix(1) == 12
    assert HealpixMap.nside_to_npix(32) == 12288
    assert HealpixMap.nside_to_npix(2048) == 50331648

    with pytest.raises(AssertionError):
        assert HealpixMap.nside_to_npix(123) == 1


def test_npix_to_nside():
    assert HealpixMap.npix_to_nside(12) == 1
    assert HealpixMap.npix_to_nside(12288) == 32
    assert HealpixMap.npix_to_nside(50331648) == 2048

    with pytest.raises(AssertionError):
        assert HealpixMap.npix_to_nside(123) == 1


def test_nside_to_pixel_solid_angle_sterad():
    actual = np.array(
        [HealpixMap.nside_to_pixel_solid_angle_sterad(2**i) for i in range(12)]
    )
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
    actual = np.array([HealpixMap.nside_to_resolution_rad(2**i) for i in range(12)])
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


def test_healpixmap_init_validation():
    """Test initialization logic, reshaping, and error handling."""
    nside = 16
    npix = 12 * nside**2

    # 1. Test scalar input (1D array promotion to 2D)
    values_1d = np.zeros(npix)
    m = HealpixMap(values=values_1d, nside=nside)
    assert m.nstokes == 1
    assert m.values.shape == (1, npix)

    # 2. Test tuple input conversion
    values_tuple = (np.zeros(npix), np.zeros(npix), np.zeros(npix))
    m = HealpixMap(values=values_tuple, nside=nside)
    assert m.nstokes == 3
    assert isinstance(m.values, np.ndarray)
    assert m.values.shape == (3, npix)

    # 3. Test invalid NSIDE (not power of 2)
    with pytest.raises(AssertionError):
        HealpixMap(values=np.zeros(12 * 3**2), nside=3)

    # 4. Test mismatch between NSIDE and pixel count
    with pytest.raises(ValueError, match="Wrong number of pixels"):
        HealpixMap(values=np.zeros(100), nside=nside)

    # 5. Test invalid Stokes dimension (e.g., 2 components)
    with pytest.raises(ValueError, match="Stokes parameters .* should be 1 or 3"):
        HealpixMap(values=np.zeros((2, npix)), nside=nside)


def test_healpixmap_algebra_scalar():
    """Test basic arithmetic operations with scalars."""
    nside = 8
    npix = 12 * nside**2
    val1 = np.ones((1, npix))
    val2 = np.ones((1, npix)) * 2.0

    m1 = HealpixMap(values=val1, nside=nside)
    m2 = HealpixMap(values=val2, nside=nside)

    # Addition
    m_sum = m1 + m2
    np.testing.assert_array_equal(m_sum.values, 3.0)

    # Subtraction
    m_diff = m2 - m1
    np.testing.assert_array_equal(m_diff.values, 1.0)

    # In-place addition
    m1 += m2
    np.testing.assert_array_equal(m1.values, 3.0)

    # Multiplication by scalar
    m_mult = m2 * 2.0
    np.testing.assert_array_equal(m_mult.values, 4.0)


def test_healpixmap_algebra_stokes():
    """Test multiplication by Stokes vectors."""
    nside = 8
    npix = 12 * nside**2
    # Create I, Q, U map where I=1, Q=2, U=3 everywhere
    values = np.array([np.ones(npix), np.ones(npix) * 2, np.ones(npix) * 3])
    m = HealpixMap(values=values, nside=nside)

    # Multiply by Stokes vector [1, 0.5, 0]
    # Expected: I=1, Q=1, U=0
    stokes_vec = np.array([1.0, 0.5, 0.0])
    m_new = m * stokes_vec

    expected = np.array([np.ones(npix), np.ones(npix), np.zeros(npix)])
    np.testing.assert_array_equal(m_new.values, expected)

    # Test invalid vector shape
    with pytest.raises(ValueError):
        _ = m * np.array([1.0, 1.0])


def test_healpixmap_consistency_checks():
    """Test consistency guards (nside, nest, stokes compatibility)."""
    nside1 = 8
    nside2 = 16
    npix1 = 12 * nside1**2
    npix2 = 12 * nside2**2

    m1 = HealpixMap(values=np.zeros((1, npix1)), nside=nside1, nest=False)
    m2 = HealpixMap(values=np.zeros((1, npix2)), nside=nside2, nest=False)  # Diff NSIDE
    m3 = HealpixMap(values=np.zeros((1, npix1)), nside=nside1, nest=True)  # Diff NEST
    m4 = HealpixMap(
        values=np.zeros((3, npix1)), nside=nside1, nest=False
    )  # Diff Stokes

    with pytest.raises(ValueError, match="matching nside, nest, and nstokes"):
        _ = m1 + m2

    with pytest.raises(ValueError, match="matching nside, nest, and nstokes"):
        _ = m1 + m3

    with pytest.raises(ValueError, match="matching nside, nest, and nstokes"):
        _ = m1 + m4


def test_healpixmap_units_propagation():
    """Test units compatibility and propagation rules."""
    nside = 8
    npix = 12 * nside**2
    vals = np.zeros((1, npix))

    # Assume Units.K_CMB and Units.K_RJ exist in constants.py,
    # otherwise use mocked Enum members or Units.None logic depending on implementation.
    # Here we test logic based on the code provided:

    m_none = HealpixMap(values=vals, nside=nside, units=None)
    m_kcmb = HealpixMap(values=vals, nside=nside, units=Units.K_CMB)
    m_krj = HealpixMap(values=vals, nside=nside, units=Units.K_RJ)

    # 1. None + K_CMB -> K_CMB (Inheritance)
    res = m_none + m_kcmb
    assert res.units == Units.K_CMB

    # 2. K_CMB + K_CMB -> K_CMB (Match)
    res = m_kcmb + m_kcmb
    assert res.units == Units.K_CMB

    # 3. K_CMB + K_RJ -> Error (Mismatch)
    with pytest.raises(ValueError, match="Incompatible units"):
        _ = m_kcmb + m_krj

    # 4. Multiplication override
    # K_CMB * scalar -> K_CMB
    res = m_kcmb * 2.0
    assert res.units == Units.K_CMB

    # K_CMB * scalar (with override) -> K_RJ
    res = m_kcmb.__mul__(2.0, units=Units.K_RJ)
    assert res.units == Units.K_RJ


def test_healpixmap_copy_and_equality():
    """Test deep copy and equality checks."""
    nside = 4
    npix = 12 * nside**2
    m1 = HealpixMap(values=np.random.rand(1, npix), nside=nside, units=Units.K_CMB)

    # Test Copy
    m2 = m1.copy()
    assert m1 == m2
    assert m1.values is not m2.values  # Ensure deep copy of array

    # Modify copy
    m2.values[0, 0] += 10.0
    assert m1 != m2
    assert not m1.allclose(m2)

    # Test allclose with tolerance
    m3 = m1.copy()
    m3.values += 1e-9
    assert m1.allclose(m3, atol=1e-8)


def test_rotate_alm():
    """Test spherical harmonic rotations (Euler angles, coordinate transforms and resizing)."""
    # Setup parameters
    lmax = 4
    nstokes = 3
    n_coeffs = SphericalHarmonics.num_of_alm_from_lmax(lmax)

    # Create random coefficients (T, E, B)
    rng = np.random.default_rng(42)
    values = rng.standard_normal((nstokes, n_coeffs)) + 1j * rng.standard_normal(
        (nstokes, n_coeffs)
    )

    # --- FIX: Ensure m=0 coefficients are real ---
    # For real-valued maps, a_l0 must be Real.
    # In standard packing, the first (lmax + 1) elements correspond to m=0.
    values[:, : lmax + 1] = values[:, : lmax + 1].real
    # ---------------------------------------------

    # -------------------------------------------------------------------------
    # 1. Identity Rotation (Explicit 0,0,0)
    # -------------------------------------------------------------------------
    sh = SphericalHarmonics(values, lmax=lmax, coordinates=CoordinateSystem.Ecliptic)

    # Explicit angles (0,0,0) -> Should preserve values but reset coordinates to None
    sh_ident = rotate_alm(sh, psi=0.0, theta=0.0, phi=0.0)

    np.testing.assert_allclose(
        sh.values,
        sh_ident.values,
        err_msg="Identity rotation (0,0,0) should not change values.",
    )
    assert sh_ident.coordinates is None, (
        "Explicit angles should reset coordinates to None (generic rotation)."
    )

    # -------------------------------------------------------------------------
    # 2. Predefined Rotation: Ecliptic -> Galactic (e2g)
    # -------------------------------------------------------------------------
    sh_ecl = SphericalHarmonics(
        values, lmax=lmax, coordinates=CoordinateSystem.Ecliptic
    )
    sh_gal = rotate_alm(sh_ecl, kind="e2g")

    assert sh_gal.coordinates == CoordinateSystem.Galactic
    assert sh_gal is not sh_ecl  # Should be a new object (default inplace=False)

    # -------------------------------------------------------------------------
    # 3. Round Trip: Ecliptic -> Galactic -> Ecliptic
    # -------------------------------------------------------------------------
    sh_back = rotate_alm(sh_gal, kind="g2e")

    assert sh_back.coordinates == CoordinateSystem.Ecliptic

    # Check that we recover original values
    np.testing.assert_allclose(
        sh_ecl.values,
        sh_back.values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="E2G -> G2E round trip failed.",
    )

    # -------------------------------------------------------------------------
    # 4. In-place Operation
    # -------------------------------------------------------------------------
    sh_inplace = sh_ecl.copy()
    # Inplace allowed only if mmax doesn't change (default mmax_out=None -> keeps input mmax)
    rotate_alm(sh_inplace, kind="e2g", inplace=True)

    assert sh_inplace.coordinates == CoordinateSystem.Galactic
    assert not np.allclose(sh_ecl.values, sh_inplace.values), (
        "In-place rotation did not modify values."
    )

    # -------------------------------------------------------------------------
    # 5. Resizing mmax (New Feature)
    # -------------------------------------------------------------------------
    # Reduce mmax (truncate high m modes)
    reduced_mmax = lmax - 2
    sh_reduced = rotate_alm(sh_ecl, psi=0.1, mmax_out=reduced_mmax)

    assert sh_reduced.mmax == reduced_mmax
    assert sh_reduced.lmax == lmax  # lmax remains unchanged

    expected_size = SphericalHarmonics.alm_array_size(lmax, reduced_mmax, nstokes)
    assert sh_reduced.values.shape == expected_size
    assert sh_reduced.values.shape[1] < sh_ecl.values.shape[1]

    # -------------------------------------------------------------------------
    # 6. Error Handling & Validation
    # -------------------------------------------------------------------------

    # A. Coordinate Mismatch: sh_gal is Galactic, asking for E2G
    with pytest.raises(ValueError, match="requires input in Ecliptic"):
        rotate_alm(sh_gal, kind="e2g")

    # B. Coordinate Mismatch: sh_ecl is Ecliptic, asking for G2E
    with pytest.raises(ValueError, match="requires input in Galactic"):
        rotate_alm(sh_ecl, kind="g2e")

    # C. Ambiguity: Both kind and explicit angles provided
    with pytest.raises(ValueError, match="Cannot specify both"):
        rotate_alm(sh_ecl, kind="e2g", psi=0.1)

    # D. Invalid mmax_out: Provided mmax_out > object lmax
    # (Replaces the old lmax check)
    with pytest.raises(ValueError, match="Provided mmax_out"):
        rotate_alm(sh_ecl, psi=0.1, mmax_out=lmax + 1)

    # E. Invalid Inplace Resizing: Trying to change mmax in-place
    with pytest.raises(ValueError, match="Cannot perform inplace"):
        rotate_alm(sh_ecl, psi=0.1, mmax_out=lmax - 1, inplace=True)

    # F. No-Op Warning: No rotation specified and mmax unchanged
    with pytest.warns(UserWarning, match="No rotation specified"):
        res = rotate_alm(sh_ecl)  # Defaults: kind=None, mmax_out=None (-> input mmax)
        np.testing.assert_array_equal(res.values, sh_ecl.values)


def test_synthesize_alm_scalar():
    """Test synthesis of intensity-only (TT) maps."""
    lmax = 10
    cl_tt = np.ones(lmax + 1)

    # 1. Automatic lmax detection from input array length
    sh = synthesize_alm({"TT": cl_tt})
    assert sh.nstokes == 1
    assert sh.lmax == lmax
    # Verify the array size matches the lmax
    expected_size = SphericalHarmonics.num_of_alm_from_lmax(lmax)
    assert sh.values.shape == (1, expected_size)

    # 2. Explicit lmax (should override or match)
    sh2 = synthesize_alm({"TT": cl_tt}, lmax=lmax)
    assert sh2.lmax == lmax


def test_synthesize_alm_polarization():
    """Test synthesis of polarized maps (T, E, B) and implicit zero-filling."""
    lmax = 5
    cl_tt = np.ones(lmax + 1)
    cl_ee = np.ones(lmax + 1)
    cl_bb = np.ones(lmax + 1)
    cl_te = np.zeros(lmax + 1)

    # Case A: Full Standard CMB (TT, EE, BB, TE)
    sh = synthesize_alm({"TT": cl_tt, "EE": cl_ee, "BB": cl_bb, "TE": cl_te}, lmax=lmax)
    assert sh.nstokes == 3
    assert sh.values.shape == (3, SphericalHarmonics.num_of_alm_from_lmax(lmax))

    # Case B: Implicit zero B-modes (only TE provided)
    # The function should automatically set nstokes=3 and fill B with zeros
    sh_nob = synthesize_alm({"TT": cl_tt, "EE": cl_ee, "TE": cl_te}, lmax=lmax)
    assert sh_nob.nstokes == 3
    # Verify B-mode values are exactly zero
    npt.assert_allclose(sh_nob.values[2], 0.0)


def test_synthesize_alm_units():
    """Test that units are correctly passed to the SphericalHarmonics object."""
    lmax = 2
    cl_tt = np.ones(lmax + 1)

    sh = synthesize_alm({"TT": cl_tt}, lmax=lmax, units=Units.uK_CMB)
    assert sh.units == Units.uK_CMB


def test_synthesize_alm_mmax_cut():
    """Test that specifying mmax reduces the output array size correctly."""
    lmax = 10
    mmax = 2
    cl_tt = np.ones(lmax + 1)

    sh = synthesize_alm({"TT": cl_tt}, lmax=lmax, mmax=mmax)
    assert sh.lmax == lmax
    assert sh.mmax == mmax

    # Check that the number of coefficients corresponds to the mmax cut
    expected_size = SphericalHarmonics.num_of_alm_from_lmax(lmax, mmax)
    assert sh.values.shape[1] == expected_size


def test_compute_cl_basic_auto():
    """Test basic auto-spectrum computation for scalar and polarized inputs."""
    lmax = 4
    n_coeffs = SphericalHarmonics.num_of_alm_from_lmax(lmax)

    # 1. Scalar (TT)
    val = np.random.standard_normal((1, n_coeffs)) + 1j * np.random.standard_normal(
        (1, n_coeffs)
    )
    sh = SphericalHarmonics(val, lmax=lmax)

    cls = compute_cl(sh)
    assert "TT" in cls
    assert len(cls["TT"]) == lmax + 1
    # Sanity check: Auto-spectrum must be real and non-negative
    # (Using complex inputs might give small imaginary residuals due to precision, but real part should be > 0)
    assert np.all(cls["TT"] >= 0)

    # 2. Polarization
    val_pol = np.random.standard_normal((3, n_coeffs)) + 1j * np.random.standard_normal(
        (3, n_coeffs)
    )
    sh_pol = SphericalHarmonics(val_pol, lmax=lmax)
    cls_pol = compute_cl(sh_pol)
    # Should contain all auto keys + cross keys (TE, TB, EB)
    expected_keys = {"TT", "EE", "BB", "TE", "TB", "EB"}
    assert expected_keys.issubset(cls_pol.keys())


def test_compute_cl_cross_symmetrization():
    """Test cross-spectrum logic: symmetrized vs full output."""
    lmax = 4
    n_coeffs = SphericalHarmonics.num_of_alm_from_lmax(lmax)

    # Create two random polarized maps
    val1 = np.random.randn(3, n_coeffs) + 1j * np.random.randn(3, n_coeffs)
    val2 = np.random.randn(3, n_coeffs) + 1j * np.random.randn(3, n_coeffs)

    sh1 = SphericalHarmonics(val1, lmax=lmax)
    sh2 = SphericalHarmonics(val2, lmax=lmax)

    # 1. Symmetrized (Default) -> Should produce averages like TE = (T1E2 + E1T2)/2
    cls_sym = compute_cl(sh1, sh2, symmetrize=True)
    assert {"TT", "EE", "BB", "TE", "TB", "EB"} == set(cls_sym.keys())
    assert "ET" not in cls_sym

    # 2. Non-symmetrized -> Should produce directional cross spectra (TE != ET)
    cls_raw = compute_cl(sh1, sh2, symmetrize=False)
    expected_full_keys = {"TT", "EE", "BB", "TE", "TB", "EB", "ET", "BT", "BE"}
    assert expected_full_keys == set(cls_raw.keys())


def test_compute_cl_input_clamping_and_mismatch():
    """
    Test that compute_cl robustly handles:
    1. Inputs with different lmax (should use intersection).
    2. Requested lmax larger than input (should warn and clamp).
    """
    lmax_large = 10
    lmax_small = 5

    # Create zero-filled objects for shape testing
    sh_large = SphericalHarmonics(
        np.zeros((1, SphericalHarmonics.num_of_alm_from_lmax(lmax_large))),
        lmax=lmax_large,
    )
    sh_small = SphericalHarmonics(
        np.zeros((1, SphericalHarmonics.num_of_alm_from_lmax(lmax_small))),
        lmax=lmax_small,
    )

    # 1. Intersection Logic: Cross spectrum between lmax=10 and lmax=5
    # The result should have length corresponding to the smaller lmax (5)
    cls = compute_cl(sh_large, sh_small)
    assert len(cls["TT"]) == lmax_small + 1

    # 2. Warning Logic: Request lmax=20 from lmax=5 input
    with pytest.warns(UserWarning, match="Requested lmax"):
        cls_clamp = compute_cl(sh_small, lmax=20)

    # It should have clamped effectively to lmax_small
    assert len(cls_clamp["TT"]) == lmax_small + 1
