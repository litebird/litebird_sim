# -*- encoding: utf-8 -*-

from pathlib import Path
import unittest

import numpy as np
import numpy.testing as npt

from litebird_sim.grasp2alm import grasp2alm, BeamGrid, BeamCut

REFERENCE_BEAM_FILES_PATH = Path(__file__).parent / "grasp2alm_reference"


class TestBeamCut(unittest.TestCase):
    """
    Unit tests for the BeamCut class to ensure proper handling of beam cut files and
    correct exception raising for invalid inputs.

    Methods:
        setUp():
            Sets up the test environment by defining the path to the test cut file.

        tearDown():
            Cleans up the test environment by removing the test cut file after each test.

        write_to_test_cut(txt: str):
            Writes the provided text to the test cut file.

        test_input_extension_exception():
            Tests that the BeamCut class raises a ValueError
            for files with an incorrect extension.

        test_ncomp_exception():
            Tests that the BeamCut class raises a ValueError
            for an incorrect number of components in the cut file.

        test_vnum_exception():
            Tests that the BeamCut class raises a ValueError
            for an incorrect number of data points (vnum) in the cut file.

        test_nan_exception():
            Tests that the BeamCut class raises a ValueError
            when the cut file contains NaN values.

        test_cut_reading():
            Tests that the BeamCut class correctly reads a
            properly formatted cut file and verifies the contents.
    """

    def setUp(self):
        """
        Sets up the test environment by defining the path to the test cut file.
        """
        self.path_to_test_cut = str(REFERENCE_BEAM_FILES_PATH / "unit_test.cut")

    def tearDown(self):
        """
        Cleans up the test environment by removing
        the test cut file after each test method is executed.
        """
        test_cut_path = Path(self.path_to_test_cut)
        if test_cut_path.exists():
            test_cut_path.unlink()

    def write_to_test_cut(self, txt: str):
        """
        Writes the provided text to the test cut file.

        Args:
            txt (str): The text to write to the test cut file.
        """
        with open(self.path_to_test_cut, "w", encoding="utf-8") as text_file:
            text_file.write(txt)

    def test_ncomp_exception(self):
        """
        Tests that the BeamCut class raises a ValueError
        for an incorrect number of components in the cut file.

        Asserts:
            Raises ValueError when the number of components (ncomp) is incorrect.
        """
        txt_with_error = """Test header
            -180 90 3 0 3 1 100
            1 2 3 4
            -1.0 -2.0 -3.0 -4.0
            1 1 1 1
        """
        self.write_to_test_cut(txt_with_error)
        with self.assertRaises(ValueError):
            BeamCut(self.path_to_test_cut)

    def test_vnum_exception(self):
        """
        Tests that the BeamCut class raises a ValueError
        for an incorrect number of data points (vnum) in the cut file.

        Asserts:
            Raises ValueError when the number of data points (vnum) is incorrect.
        """
        txt_with_error = """Test header
            -180 90 10 0 3 1 2
            1 2 3 4
            -1.0 -2.0 -3.0 -4.0
            1 1 1 1
        """
        self.write_to_test_cut(txt_with_error)
        with self.assertRaises(ValueError):
            BeamCut(self.path_to_test_cut)

    def test_nan_exception(self):
        """
        Tests that the BeamCut class raises a ValueError
        when the cut file contains NaN values.

        Asserts:
            Raises ValueError when the cut file contains NaN values.
        """
        txt_with_error = """Test header
            -180 90 3 0 3 1 2
            1 2 3 4
            -1.0 nan -3.0 -4.0
            1 1 1 1
        """
        self.write_to_test_cut(txt_with_error)
        with self.assertRaises(ValueError):
            BeamCut(self.path_to_test_cut)

    def test_cut_reading(self):
        """
        Tests that the BeamCut class correctly reads a properly formatted cut file
        and verifies the contents.

        Asserts:
            Asserts that the header, vini, vinc, vnum, c,
            icomp, icut, ncomp, ncut, and amp
            are correctly read from the cut file.
        """
        txt = """Test header
            -180 90 3 0 3 1 2
            1 2 3 4
            -1.0 -2.0 -3.0 -4.0
            1 1 1 1
        """
        self.write_to_test_cut(txt)
        test_cut = BeamCut(self.path_to_test_cut)
        expected_amp = np.array(
            [
                [[1.0 + 2.0j], [-1.0 - 2.0j], [1.0 + 1.0j]],
                [[3.0 + 4.0j], [-3.0 - 4.0j], [1.0 + 1.0j]],
            ]
        )
        self.assertTrue(test_cut.header == "Test header")
        self.assertTrue(test_cut.vini == -180.0)
        self.assertTrue(test_cut.vinc == 90.0)
        self.assertTrue(test_cut.vnum == 3)
        self.assertTrue(test_cut.c == np.array([0.0]))
        self.assertTrue(test_cut.icomp == 3)
        self.assertTrue(test_cut.icut == 1)
        self.assertTrue(test_cut.ncomp == 2)
        self.assertTrue(test_cut.ncut == 1)
        self.assertTrue(np.array_equal(test_cut.amp, expected_amp))


class TestBeamGrid(unittest.TestCase):
    """
    Unit tests for the BeamGrid class to ensure proper handling of beam grid files and
    correct exception raising for invalid inputs.

    Methods:
        setUp():
            Sets up the test environment by defining the path to the test grid file.

        tearDown():
            Cleans up the test environment by removing the test grid file after each test.

        write_to_test_grid(txt: str):
            Writes the provided text to the test grid file.

        test_input_extension_exception():
            Tests that the BeamGrid class raises a ValueError
            for files with an incorrect extension.

        test_input_grid_format():
            Tests that the BeamGrid class raises a ValueError
            for an incorrect number for grid format (ktype) in the grid file.

        test_input_beams_number():
            Tests that the BeamGrid class raises a Warning
            for an incorrect number of beams (nset) in the grid file.

        test_input_beam_solid_angle():
            Tests that the BeamGrid class raises a Warning
            for an incorrect value of beam solid angle in the grid file.


        test_nan_exception():
            Tests that the BeamGrid class raises a ValueError
            when the grid file contains NaN values.

        test_grid_reading():
            Tests that the BeamGrid class correctly reads a
            properly formatted grid file and verifies the contents."""

    def setUp(self):
        """
        Sets up the test environment by defining the path to the test grid file.
        """
        self.path_to_test_grid = str(REFERENCE_BEAM_FILES_PATH / "unit_test.grd")

    def tearDown(self):
        """
        Cleans up the test environment by removing
        the test grid file after each test method is executed.
        """
        test_grid_path = Path(self.path_to_test_grid)
        if test_grid_path.exists():
            test_grid_path.unlink()

    def write_to_test_grid(self, txt: str):
        """
        Writes the provided text to the test grid file.

        Args:
            txt (str): The text to write to the test cut file.
        """
        text_file = open(self.path_to_test_grid, "w", encoding="utf-8")
        text_file.write(txt)
        text_file.close()

    def test_input_grid_format(self):
        """
        Tests that the BeamGrid class raises a ValueError for an incorrect number for grid format in the grid file.

        Asserts:
            Raises ValueError when the grid format (ktype) is not 1.
        """
        txt_with_error = "Test header\n++++\n2"
        self.write_to_test_grid(txt_with_error)
        with self.assertRaises(ValueError):
            BeamGrid(self.path_to_test_grid)

    def test_input_beams_number(self):
        """
        Tests that the BeamGrid class raises a Warning for an incorrect number of beams in the grid file.

        Asserts:
            Raises a Warning when the number of beams (nset) is not 1.
        """
        txt_with_error = (
            "Test header\n"
            + "++++\n"
            + "1\n"
            + "2 3 2 7\n"
            + "0 0\n\n"
            + "0.0 0.0 360.0 90.0\n"
            + "2 2 0\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1"
        )
        self.write_to_test_grid(txt_with_error)
        with self.assertWarns(Warning):
            BeamGrid(self.path_to_test_grid)

    def test_input_beam_solid_angle(self):
        """
        Tests that the BeamGrid class raises a Warning for an incorrect value of beam solid angle in the grid file.

        Asserts:
            Raises a Warning if the beam solid angle is different from 2pi or 4pi
        """
        txt_with_error = (
            "Test header\n"
            + "++++\n"
            + "1\n"
            + "1 3 2 7\n"
            + "0 0\n"
            + "0.0 0.0 340.0 80.0\n"
            + "2 2 0\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1"
        )
        self.write_to_test_grid(txt_with_error)
        with self.assertWarns(Warning):
            BeamGrid(self.path_to_test_grid)

    def test_nan_exception(self):
        """
        Tests that the BeamGrid class raises a ValueError
        when the grid file contains NaN values.

        Asserts:
            Raises a ValueError when the gridfile contains NaN values

        """
        txt_with_error = (
            "Test header\n"
            + "++++\n"
            + "1\n"
            + "1 3 2 7\n"
            + "0 0\n"
            + "0.0 0.0 360.0 90.0\n"
            + "2 2 0\n"
            + "1 1 1 1\n"
            + "1 Nan 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1"
        )
        self.write_to_test_grid(txt_with_error)
        with self.assertRaises(ValueError):
            BeamGrid(self.path_to_test_grid)

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
        txt = (
            "Test header\n"
            + "VERSION: TICRA-EM-FIELD-V0.1\n"
            + "FREQUENCY_NAME: freq\n"
            + "FREQUENCIES [GHz]:\n"
            + "119.0\n"
            + "++++\n"
            + "1\n"
            + "1 3 2 7\n"
            + "0 0\n"
            + "0.0 0.0 360.0 90.0\n"
            + "3 3 0\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1\n"
            + "1 1 1 1"
        )

        self.write_to_test_grid(txt)
        test_grid = BeamGrid(self.path_to_test_grid)

        assert (
            test_grid.header
            == "Test header\n"
            + "VERSION: TICRA-EM-FIELD-V0.1\n"
            + "FREQUENCY_NAME: freq\n"
        )
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


class TestGrasp2Alm(unittest.TestCase):
    def test_grasp2alm_with_gridfile(self):
        file = REFERENCE_BEAM_FILES_PATH / "test.grd"
        file = str(file)
        beam = BeamGrid(file)
        polar = beam.to_polar()
        nside = 128
        beammap = polar.to_map(nside)
        expected_result = beammap.to_alm(
            lmax=5 * nside - 1, epsilon=1e-5, max_num_of_iterations=20
        )

        grasp2alm_result = grasp2alm(
            file, nside, lmax=5 * nside - 1, epsilon=1e-5, max_num_of_iterations=20
        )

        # Since a_ℓm coefficients are complex but npt.assert_allclose only supports reals,
        # we must compare their real and imaginary parts separately
        npt.assert_allclose(
            actual=np.real(grasp2alm_result), desired=np.real(expected_result)
        )
        npt.assert_allclose(
            actual=np.imag(grasp2alm_result), desired=np.imag(expected_result)
        )

    def test_grasp2alm_with_cutfile(self):
        file = REFERENCE_BEAM_FILES_PATH / "test.cut"
        file = str(file)
        beam = BeamCut(file)
        polar = beam.to_polar()
        nside = 128
        beammap = polar.to_map(nside)
        expected_result = beammap.to_alm(
            lmax=5 * nside - 1, epsilon=1e-5, max_num_of_iterations=20
        )

        grasp2alm_result = grasp2alm(
            file, nside, lmax=5 * nside - 1, epsilon=1e-5, max_num_of_iterations=20
        )

        # Since a_ℓm coefficients are complex but npt.assert_allclose only supports reals,
        # we must compare their real and imaginary parts separately
        npt.assert_allclose(
            actual=np.real(grasp2alm_result), desired=np.real(expected_result)
        )
        npt.assert_allclose(
            actual=np.imag(grasp2alm_result), desired=np.imag(expected_result)
        )
