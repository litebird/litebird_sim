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

    def test_grid_reading(self):
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

        assert test_grid.ktype == 1
        assert test_grid.nset == 1
        assert test_grid.klimit == 0
        assert test_grid.icomp == 3
        assert test_grid.ncomp == 2
        assert test_grid.igrid == 7
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
