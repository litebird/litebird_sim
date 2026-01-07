# -*- encoding: utf-8 -*-
from io import StringIO
from pathlib import Path
import unittest
from typing import Type

import numpy as np
import numpy.testing as npt
import pytest

from litebird_sim.beam_convolution import SphericalHarmonics
from litebird_sim.grasp2alm import (
    BeamGrid,
    BeamCut,
    ticra_cut_to_alm,
    ticra_grid_to_alm,
)

REFERENCE_BEAM_FILES_PATH = Path(__file__).parent / "grasp2alm_reference"

# The GRASP grid file has been built by hand using the following Mathematica functions:
#
# toThetaPhi[az_, el_] := {Sqrt[az^2 + el^2], If[az != 0 \[Or] el != 0, ArcTan[-az, el], 0]};
# rotStokes[\[Theta]_, \[CurlyPhi]_, {i_, q_, u_, v_}] :=
#   If[\[Theta] != 0,
#    {i, q Cos[2 \[CurlyPhi]] - u Sin[2 \[CurlyPhi]],
#     q Sin[2 \[CurlyPhi]] + u Cos[2 \[CurlyPhi]], v},
#    (* No transformation is needed at the North Pole*)
#    {i, q, u, v}];
# toStokes[{E\[Theta]_, E\[CurlyPhi]_}] := {
#    Abs[E\[Theta]]^2 + Abs[E\[CurlyPhi]]^2,
#    Abs[E\[Theta]]^2 - Abs[E\[CurlyPhi]]^2,
#    2 Re[E\[Theta] * Conjugate[E\[CurlyPhi]]],
#    -2 Im[E\[CurlyPhi] * Conjugate[E\[CurlyPhi]]]
#    };
# testField[az_, el_] := {az/(20 °) + el/(40 °) \[ImaginaryJ],
#    az/(15 °) + el/(30 °) \[ImaginaryJ]};
#
# A few notes:
#
# - `toThetaPhi` converts an azimuth/elevation couple into a θ/φ pair;
# - `rotStokes` rotates the Stokes parameters referring to the (θ, φ) point
# - `toStokes` converts an electric vector into a Stokes vector (V is always zero)
# - `testField` generates an electric field (E_co, E_cx) at a specific location on the
#   far field sphere, using the azimuth/elevation coordinates
#
# The test field is computed on the following grid of values:
#
# | Azimuth | Elevation |
# |---------|-----------|
# |    −90° |      −90° |
# |    −45° |      −90° |
# |      0° |      −90° |
# |     45° |      −90° |
# |     90° |      −90° |
# |    −90° |        0° |
# |    −45° |        0° |
# |      0° |        0° |
# |     45° |        0° |
# |     90° |        0° |
# |    −90° |       90° |
# |    −45° |       90° |
# |      0° |       90° |
# |     45° |       90° |
# |     90° |       90° |

AZ_EL_GRID_GRASP_FILE_CONTENTS = """
Test header
VERSION: TICRA-EM-FIELD-V0.1
FREQUENCY_NAME: freq
FREQUENCIES [GHz]:
123.0
++++
1
1 3 2 5
0 0
-90.0 -90.0 90.0 90.0
5 3 0
-4.5   -2.25   -6 -3
-2.25  -2.25   -3 -3
 0     -2.25    0 -3
 2.25  -2.25    3 -3
 4.5   -2.25    6 -3
-4.5      0    -6  0
-2.25     0    -3  0
 0        0     0  0
 2.25     0     3  0
 4.5      0     6  0
-4.5      2.25 -6  3
-2.25     2.25 -3  3
 0        2.25  0  3
 2.25     2.25  3  3
 4.5      2.25  6  3
"""

# Each tuple in the list contains the following fields:
#  1. Azimuth [deg],
#  2. Elevation [deg],
#  3. θ [rad],
#  4. φ [rad],
#  5. E_co,
#  6. E_cx,
#  7. I in Ludwig’s third convention reference frame,
#  8. Q in Ludwig’s third convention reference frame,
#  9. U in Ludwig’s third convention reference frame,
# 10. V in Ludwig’s third convention reference frame,
# 11. I in the θ/φ reference frame,
# 12. Q in the θ/φ reference frame,
# 13. U in the θ/φ reference frame,
# 14. V in the θ/φ reference frame,

# fmt: off
AZ_EL_GRID_TEST_DATA = [
    (-90.0, -90.0, 2.22144, -0.785398, -4.5 - 2.25j, -6.0 - 3.0j, 70.3125, \
     -19.6875, 67.5, 0.0, 70.3125, -67.5, -19.6875, 0.0),
    (-45.0, -90.0, 1.7562, -1.10715, -2.25 - 2.25j, -3.0 - 3.0j, 28.125, \
     -7.875, 27.0, 0.0, 28.125, -16.875, -22.5, 0.0),
    (0.0, -90.0, 1.5708, -1.5708, 0.0 - 2.25j, 0.0 - 3.0j, 14.0625, \
     -3.9375, 13.5, 0.0, 14.0625, 3.9375, -13.5, 0.0),
    (45.0, -90.0, 1.7562, -2.03444, 2.25 - 2.25j, 3.0 - 3.0j, 28.125, \
     -7.875, 27.0, 0.0, 28.125, 26.325, -9.9, 0.0),
    (90.0, -90.0, 2.22144, -2.35619, 4.5 - 2.25j, 6.0 - 3.0j, 70.3125, \
     -19.6875, 67.5, 0.0, 70.3125, 67.5, 19.6875, 0.0),
    (-90.0, 0.0, 1.5708, 0.0, -4.5, -6.0, 56.25, -15.75, 54.0, 0.0, \
     56.25, -15.75, 54.0, 0.0),
    (-45.0, 0.0, 0.785398, 0.0, -2.25, -3.0, 14.0625, -3.9375, 13.5, 0.0, \
     14.0625, -3.9375, 13.5, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
     0.0),
    (45.0, 0.0, 0.785398, 3.14159, 2.25, 3.0, 14.0625, -3.9375, 13.5, \
     0.0, 14.0625, -3.9375, 13.5, 0.0),
    (90.0, 0.0, 1.5708, 3.14159, 4.5, 6.0, 56.25, -15.75, 54.0, 0.0, \
     56.25, -15.75, 54.0, 0.0),
    (-90.0, 90.0, 2.22144, 0.785398, -4.5 + 2.25j, -6.0 + 3.0j, 70.3125, \
     -19.6875, 67.5, 0.0, 70.3125, 67.5, 19.6875, 0.0),
    (-45.0, 90.0, 1.7562, 1.10715, -2.25 + 2.25j, -3.0 + 3.0j, 28.125, \
     -7.875, 27.0, 0.0, 28.125, 26.325, -9.9, 0.0),
    (0.0, 90.0, 1.5708, 1.5708, 0.0 + 2.25j, 0.0 + 3.0j, 14.0625, \
     -3.9375, 13.5, 0.0, 14.0625, 3.9375, -13.5, 0.0),
    (45.0, 90.0, 1.7562, 2.03444, 2.25 + 2.25j, 3.0 + 3.0j, 28.125, \
     -7.875, 27.0, 0.0, 28.125, -16.875, -22.5, 0.0),
    (90.0, 90.0, 2.22144, 2.35619, 4.5 + 2.25j, 6.0 + 3.0j, 70.3125, \
     -19.6875, 67.5, 0.0, 70.3125, -67.5, -19.6875, 0.0),
]
# fmt: on


def _test_value_error(text: str, beam_class: Type[BeamCut] | Type[BeamGrid]) -> None:
    txt_with_error = StringIO(text)
    txt_with_error.seek(0)
    with pytest.raises(ValueError):
        beam_class(txt_with_error)


class TestBeamCut(unittest.TestCase):
    """
    Unit tests for the BeamCut class to ensure proper handling of beam cut files and
    correct exception raising for invalid inputs.
    """

    def test_ncomp_exception(self):
        """
        Tests that the BeamCut class raises a ValueError
        for an incorrect number of components in the cut file.

        Asserts:
            Raises ValueError when the number of components (ncomp) is incorrect.
        """
        _test_value_error(
            """Test header
            -180 90 3 0 3 1 100
            1 2 3 4
            -1.0 -2.0 -3.0 -4.0
            1 1 1 1
        """,
            BeamCut,
        )

    def test_vnum_exception(self):
        """
        Tests that the BeamCut class raises a ValueError
        for an incorrect number of data points (vnum) in the cut file.

        Asserts:
            Raises ValueError when the number of data points (vnum) is incorrect.
        """
        _test_value_error(
            """Test header
            -180 90 10 0 3 1 2
            1 2 3 4
            -1.0 -2.0 -3.0 -4.0
            1 1 1 1
        """,
            BeamCut,
        )

    def test_nan_exception(self):
        """
        Tests that the BeamCut class raises a ValueError
        when the cut file contains NaN values.

        Asserts:
            Raises ValueError when the cut file contains NaN values.
        """
        _test_value_error(
            """Test header
            -180 90 3 0 3 1 2
            1 2 3 4
            -1.0 nan -3.0 -4.0
            1 1 1 1
        """,
            BeamCut,
        )

    def test_cut_reading(self):
        """
        Tests that the BeamCut class correctly reads a properly formatted cut file
        and verifies the contents.

        Asserts:
            Asserts that the header, vini, vinc, vnum, c,
            icomp, icut, ncomp, ncut, and amp
            are correctly read from the cut file.
        """
        txt = StringIO("""Test header
            -180 90 3 0 3 1 2
            1 2 3 4
            -1.0 -2.0 -3.0 -4.0
            1 1 1 1
        """)
        txt.seek(0)
        test_cut = BeamCut(txt)
        expected_amp = np.array(
            [[[(1 + 2j), (-1 - 2j), (1 + 1j)]], [[(3 + 4j), (-3 - 4j), (1 + 1j)]]]
        )
        self.assertTrue(test_cut.theta0_deg == -180.0)
        self.assertTrue(test_cut.delta_theta_deg == 90.0)
        self.assertTrue(test_cut.n_theta == 3)
        self.assertTrue(test_cut.phi_values_rad == np.array([0.0]))
        self.assertTrue(test_cut.ncomp == 2)
        self.assertTrue(test_cut.num_of_phi_cuts == 1)
        npt.assert_array_equal(test_cut.amp, expected_amp)


class TestBeamGrid(unittest.TestCase):
    """
    Unit tests for the BeamGrid class to ensure proper handling of beam grid files and
    correct exception raising for invalid inputs.
    """

    def test_input_grid_format(self):
        """
        Tests that the BeamGrid class raises a ValueError for an incorrect number for grid format in the grid file.

        Asserts:
            Raises ValueError when the grid format (ktype) is not 1.
        """
        _test_value_error("Test header\n++++\n2", BeamGrid)

    def test_nan_exception(self):
        """
        Tests that the BeamGrid class raises a ValueError
        when the grid file contains NaN values.

        Asserts:
            Raises a ValueError when the gridfile contains NaN values

        """
        _test_value_error(
            """
Test header
++++
1
1 3 2 7
0 0
0.0 0.0 360.0 90.0
2 2 0
1 1 1 1
1 Nan 1 1
1 1 1 1
1 1 1 1
""",
            BeamGrid,
        )

    def test_thetaphi_grid_reading(self):
        """
        Tests that the BeamGrid class correctly reads a
        properly formatted grid file and verifies the contents.

        Asserts:
            Asserts that the header, ktype, nset, klimit,
            icomp, ncomp, igrid, ix, iy, xs, ys, xe,
            ye, nx, ny, freq, frequnit
            are correctly read from the grid file
        """
        txt = StringIO(
            """
Test header
VERSION: TICRA-EM-FIELD-V0.1
FREQUENCY_NAME: freq
FREQUENCIES [GHz]:
119.0
++++
1
1 3 2 7
0 0
0.0 0.0 360.0 90.0
3 3 0
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
"""
        )

        test_grid = BeamGrid(txt)

        assert test_grid.field_component_type == 3
        assert test_grid.num_of_components == 2
        assert test_grid.grid_type == 7
        assert test_grid.ix == 0
        assert test_grid.iy == 0
        assert test_grid.xs == 0.0
        assert test_grid.ys == 0.0
        assert test_grid.xe == 360.0
        assert test_grid.ye == 90.0
        assert test_grid.nx == 3
        assert test_grid.ny == 3
        assert test_grid.frequency == 119.0
        assert test_grid.frequency_unit == "GHz"

        expected_amp = np.array(
            [
                [
                    [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j],
                    [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j],
                    [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j],
                ],
                [
                    [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j],
                    [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j],
                    [1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j],
                ],
            ]
        )
        self.assertTrue(np.array_equal(test_grid.amp, expected_amp))

    def test_az_el_grid_reading(self):
        """
        Tests that the BeamGrid class correctly reads a
        properly formatted grid file and verifies the contents.

        Asserts:
            Asserts that the header, ktype, nset, klimit,
            icomp, ncomp, igrid, ix, iy, xs, ys, xe,
            ye, nx, ny, freq, frequnit
            are correctly read from the grid file
        """

        txt = StringIO(AZ_EL_GRID_GRASP_FILE_CONTENTS)

        test_grid = BeamGrid(txt)
        test_grid_polar = test_grid.to_polar(copol_axis="y")
        test_grid_rotated = test_grid_polar.convert_to_polar_basis()

        assert test_grid.field_component_type == 3
        assert test_grid.num_of_components == 2
        assert test_grid.grid_type == 5
        assert test_grid.ix == 0
        assert test_grid.iy == 0
        assert test_grid.xs == -90.0
        assert test_grid.ys == -90.0
        assert test_grid.xe == 90.0
        assert test_grid.ye == 90.0
        assert test_grid.nx == 5
        assert test_grid.ny == 3
        assert test_grid.frequency == 123.0
        assert test_grid.frequency_unit == "GHz"

        cur_x = 0
        cur_y = 0
        cur_row = 0
        for (
            az_deg,
            el_deg,
            theta_rad,
            phi_rad,
            E_co,
            E_cx,
            stokes_I,
            stokes_Q,
            stokes_U,
            stokes_V,
            stokes_I_rotated,
            stokes_Q_rotated,
            stokes_U_rotated,
            stokes_V_rotated,
        ) in AZ_EL_GRID_TEST_DATA:
            npt.assert_allclose(test_grid.amp[0, cur_x, cur_y], E_co)
            npt.assert_allclose(test_grid.amp[1, cur_x, cur_y], E_cx)

            npt.assert_allclose(test_grid_polar.stokes[0, cur_row], stokes_I)
            npt.assert_allclose(test_grid_polar.stokes[1, cur_row], stokes_Q)
            npt.assert_allclose(test_grid_polar.stokes[2, cur_row], stokes_U)
            npt.assert_allclose(test_grid_polar.stokes[3, cur_row], stokes_V)

            npt.assert_allclose(test_grid_rotated.stokes[0, cur_row], stokes_I_rotated)
            npt.assert_allclose(test_grid_rotated.stokes[1, cur_row], stokes_Q_rotated)
            npt.assert_allclose(test_grid_rotated.stokes[2, cur_row], stokes_U_rotated)
            npt.assert_allclose(test_grid_rotated.stokes[3, cur_row], stokes_V_rotated)

            cur_x += 1
            if cur_x == test_grid.nx:
                cur_x = 0
                cur_y += 1

            cur_row += 1


@pytest.mark.parametrize(
    "param",
    [
        (BeamGrid, ticra_grid_to_alm, "test.grd", "test-grid-alm.npy"),
        (BeamCut, ticra_cut_to_alm, "test.cut", "test-cut-alm.npy"),
    ],
)
def test_grasp_to_alm(param, regenerate_grasp_alm: bool):
    beam_class, function_to_test, grasp_file_name, alm_file_name = param

    nside = 128
    with (REFERENCE_BEAM_FILES_PATH / grasp_file_name).open("rt") as grasp_file:
        grasp2alm_result: SphericalHarmonics = function_to_test(
            grasp_file,
            nside=nside,
        )

    reference_file = REFERENCE_BEAM_FILES_PATH / alm_file_name
    if regenerate_grasp_alm:
        grasp2alm_result.write_fits(
            REFERENCE_BEAM_FILES_PATH / reference_file, overwrite=True
        )
        pytest.skip("Regenerated reference file.")
    else:
        expected_result = SphericalHarmonics.read_fits(reference_file)

        # Since a_ℓm coefficients are complex but npt.assert_allclose only supports reals,
        # we must compare their real and imaginary parts separately
        grasp2alm_result.allclose(expected_result)
