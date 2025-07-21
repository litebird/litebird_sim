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

    def test_altaz_grid_reading(self):
        """
        Tests that the BeamGrid class correctly reads a
        properly formatted grid file and verifies the contents.

        Asserts:
            Asserts that the header, ktype, nset, klimit,
            icomp, ncomp, igrid, ix, iy, xs, ys, xe,
            ye, nx, ny, freq, frequnit
            are correctly read from the grid file
        """

        # This file samples the “azimuth” angle at five values:
        #     −90°, −45°, 0°, 45°, 90°
        # and the “elevation” angle at three values:
        #     −90°, 0°, 90°
        #
        # The values of the field are the following:
        #
        #  |   Az |   El | θ [rad]|     φ [rad] | Value (co, cx) | Value (E_θ, E_φ) |         (I, Q, U) in θ/φ |
        #  |------|------|--------|-------------|----------------|------------------|--------------------------|
        #  | −90° | −90° |   π/√2 |        −π/4 |         (1, 2) |    (−1/√2, 3/√2) |              (5, -4, -3) |
        #  | −45° | −90° | √5 π/4 |    −atan(2) |         (3, 4) |       (-√5, 2√5) |           (25, -15, -20) |
        #  |   0° | −90° |    π/2 |        −π/2 |         (5, 6) |          (−6, 5) |            (61, 11, -60) |
        #  |  45° | −90° | √5 π/4 | atan(2) − π |         (7, 8) |   (−23/√5, 6/√5) |      (113, 493/5, 276/5) |
        #  |  90° | −90° |   π/√2 |       −3π/4 |        (9, 10) |  (−19/√2, −1/√2) |           (181, 180, 19) |
        #  | −90° |   0° |    π/2 |           0 |       (11, 12) |         (11, 12) |          (265, -23, 264) |
        #  | −45° |   0° |    π/4 |           0 |       (13, 14) |         (13, 14) |          (365, -27, 364) |
        #  |   0° |   0° |      0 |           0 |       (15, 16) |         (15, 16) |          (481, -31, 480) |
        #  |  45° |   0° |    π/4 |           π |       (17, 18) |       (−17, −18) |          (613, -35, 612) |
        #  |  90° |   0° |    π/2 |           π |       (19, 20) |       (−19, −20) |          (761, -39, 760) |
        #  | −90° |  90° |   π/√2 |         π/4 |       (21, 22) |    (43/√2, 1/√2) |           (925, 924, 43) |
        #  | −45° |  90° | √5 π/4 |     atan(2) |       (23, 24) |  (71/√5, −22/√5) |  (1105, 4557/5, -3124/5) |
        #  |   0° |  90° |    π/2 |         π/2 |       (25, 26) |        (26, −25) |        (1301, 51, -1300) |
        #  |  45° |  90° | √5 π/4 | π − atan(2) |       (27, 28) |  (29/√5, −82/√5) | (1513, -5883/5, -4756/5) |
        #  |  90° |  90° |   π/√2 |        3π/4 |       (29, 30) |   (1/√2, −59/√2) |       (1741, -1740, -59) |
        #
        # To compute the value of E_θ, E_φ, apply a rotation by −φ to (E_co, E_cx), as explained in
        # <https://ziotom78.github.io/electromagnetics/2025/07/17/ludwig-polarization-definition.html>
        # (section “Converting back and forth between the spherical basis and Ludwig’s”)

        txt = StringIO(
            """
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
1 0 2 0
3 0 4 0
5 0 6 0
7 0 8 0
9 0 10 0
11 0 12 0
13 0 14 0
15 0 16 0
17 0 18 0
19 0 20 0
21 0 22 0
23 0 24 0
25 0 26 0
27 0 28 0
29 0 30 0
"""
        )

        test_grid = BeamGrid(txt)

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

        expected_amp = np.array(
            [
                [
                    # Each row contains E_co for each of the three elevations
                    [1.0 + 0.0j, 11.0 + 0.0j, 21.0 + 0.0j],
                    [3.0 + 0.0j, 13.0 + 0.0j, 23.0 + 0.0j],
                    [5.0 + 0.0j, 15.0 + 0.0j, 25.0 + 0.0j],
                    [7.0 + 0.0j, 17.0 + 0.0j, 27.0 + 0.0j],
                    [9.0 + 0.0j, 19.0 + 0.0j, 29.0 + 0.0j],
                ],
                [
                    # Each row contains E_cx for each of the three elevations
                    [2.0 + 0.0j, 12.0 + 0.0j, 22.0 + 0.0j],
                    [4.0 + 0.0j, 14.0 + 0.0j, 24.0 + 0.0j],
                    [6.0 + 0.0j, 16.0 + 0.0j, 26.0 + 0.0j],
                    [8.0 + 0.0j, 18.0 + 0.0j, 28.0 + 0.0j],
                    [10.0 + 0.0j, 20.0 + 0.0j, 30.0 + 0.0j],
                ],
            ]
        )
        np.testing.assert_allclose(test_grid.amp, expected_amp)

        polar_beam = test_grid.to_polar()
        # TODO: This cannot work! The class BeamPolar must contain a *list* of 2-tuples (θ, φ) instead
        #  of two linear vectors, otherwise it will NEVER work with IGRID≠7
        np.testing.assert_allclose(polar_beam.theta_values, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(
            polar_beam.phi_values, np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )


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
        grasp2alm_result = function_to_test(
            grasp_file,
            nside=nside,
        )  # type: SphericalHarmonics

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
