# HEAD

# Version 0.15.3

-   Fix the computation of HWP angles [#444](https://github.com/litebird/litebird_sim/pull/444)

-   Add low-level interface to `BrahMap` [#440](https://github.com/litebird/litebird_sim/pull/440)

-   Set Mueller matrix phases in Hwp_sys module as class attributes, instead of being hardcoded [#442](https://github.com/litebird/litebird_sim/pull/442)

# Version 0.15.2

-   Add more tutorials [#443](https://github.com/litebird/litebird_sim/pull/443)

# Version 0.15.1

-   Fixed s4,s5,d9,d10 and added s7 in Mbs [#437](https://github.com/litebird/litebird_sim/pull/437)

-   Make sure that the PTEP IMo, the Madam templates, and the static files required to create the HTML reports are installed properly by `pip` and enable Binder/Google Colab [#436](https://github.com/litebird/litebird_sim/pull/436)

-   Make the HWP_sys module able to deal with missing pixels, let the output maps to use a different NSIDE than the one of the inputs [#432](https://github.com/litebird/litebird_sim/pull/432)

-   Fix bug in the computation of pointings for the HWP_sys module [#429](https://github.com/litebird/litebird_sim/pull/429)

-   Upgrade PySM to 3.4.2 [#431](https://github.com/litebird/litebird_sim/pull/431)

# Version 0.15.0

-   **Breaking change**: New improved seeding strategy using `RNGHierarchy` [#414](https://github.com/litebird/litebird_sim/pull/414)

-   **Breaking change**: Drop support for Python 3.9, add support for 3.13. Note that this implies that we do no longer support TOAST2, as it does not work with Python 3.10 [#409](https://github.com/litebird/litebird_sim/pull/409)

-   **Breaking change**: The HWP and ψ angles in the HWP Systematics module have been redefined to be consistent [#377](https://github.com/litebird/litebird_sim/pull/377)

-   **Breaking change**: The two parameters `Channel` and `Mbsparams` used in `HwpSys.set_parameters()` have been renamed to `channel` and `mbs_params` [#422](https://github.com/litebird/litebird_sim/pull/422)

-   Remove extra dependencies from `pyproject.toml` [#423](https://github.com/litebird/litebird_sim/pull/423)

-   Add interpolation to `hwp_sys` [#420](https://github.com/litebird/litebird_sim/pull/420)

-   Change default ellipticity in IMo vPTEP [#419](https://github.com/litebird/litebird_sim/pull/419) and improved documentation for beam synthesis [#416](https://github.com/litebird/litebird_sim/pull/416)

-   Measure code coverage in PRs [#415](https://github.com/litebird/litebird_sim/pull/415)

-   Move map rotation from galactic to ecliptic at generation in `mbs` and test [#413](https://github.com/litebird/litebird_sim/pull/413)

-   Make actual use of `[skipci]` and `skip ci]` in commit messages [#410](https://github.com/litebird/litebird_sim/pull/410)

-   New option in `Simulation.get_sky()` for generating maps per channel, option for storing sky and beam alms in the observations [#408](https://github.com/litebird/litebird_sim/pull/408)

-   Make MPI tests more robusts and produce a “MPI” section in the output report [#405](https://github.com/litebird/litebird_sim/pull/405/)

-   Add interface to BrahMap GLS mapmaker and relative tests [#400](https://github.com/litebird/litebird_sim/pull/400).

-   Replace allreduce with Allreduce in binning.py [#407](https://github.com/litebird/litebird_sim/pull/407).

-   Function for nullifying tods in the class `Simulation` [#389](https://github.com/litebird/litebird_sim/pull/389). 

-   Save `det_idx` in HDF5 files and make MPI test fail properly [#402](https://github.com/litebird/litebird_sim/pull/402)

-   Option for centering the pointing in the beam convolution plus some reworking of the pointing API (new methods added to the class Observation) [#397](https://github.com/litebird/litebird_sim/pull/397)

-   Bug in `Simulation.get_sky()` fixed [#398](https://github.com/litebird/litebird_sim/pull/398)

-   Fix a bug in `prepare_pointings` [#396](https://github.com/litebird/litebird_sim/pull/396)

-   Deprecate `apply_hwp_to_obs()` and `HWP.add_hwp_angle()`, remove deprecated functions `Simulation.generate_spin2ecl_quaternions()` and `write_observations()`

-   Fix docstrings and type hints for `mueller_hwp` [#381](https://github.com/litebird/litebird_sim/pull/381)

-   Fix the formatting of a few docstrings [#391](https://github.com/litebird/litebird_sim/pull/391)

-   Common interface to compute pointings and detector polarization angle [#378](https://github.com/litebird/litebird_sim/pull/378)

-   Module for performing the 4π beam convolution [#338](https://github.com/litebird/litebird_sim/pull/338)

-   Make `@_profile` preserve docstrings [#371](https://github.com/litebird/litebird_sim/pull/371)

-   Upgrade Ducc0 to 0.38.0 and clarify how to compile Ducc0 from sources [#383](https://github.com/litebird/litebird_sim/pull/383), [#390](https://github.com/litebird/litebird_sim/pull/390), [#394](https://github.com/litebird/litebird_sim/pull/394)

-   Update documentation for hwp_sys and non_linearity modules [#404](https://github.com/litebird/litebird_sim/pull/404)

-   Add non-linearity coupling with hwp systematic effects [#395](https://github.com/litebird/litebird_sim/pull/395)

-   Set numba parallelization (only) in compute_signal_for_one_detector in hwp_sys [#395](https://github.com/litebird/litebird_sim/pull/395)

-   Update test_hwp_sys.py [#395](https://github.com/litebird/litebird_sim/pull/395)

-   Change hwp angle variable name in hwp_sys.py so it is cohererent with documentation [#395](https://github.com/litebird/litebird_sim/pull/395)

-   Remove the optical optical power argument from the 2f hwpss code [#395](https://github.com/litebird/litebird_sim/pull/395 )

-   Update test_hwp_diff_emiss.py [#395](https://github.com/litebird/litebird_sim/pull/395)

-   Add option to pass a seed to generate a random g_1 term in the non-linearity module [#395](https://github.com/litebird/litebird_sim/pull/395)
  
-   Fix a bug in using MPI parallelization in hwp_sys [#395](https://github.com/litebird/litebird_sim/pull/395)
 
# Version 0.14.0

-   **Breaking change**: Bug in the 1/f noise generation has been corrected. Previously, the frequency array was miscalculated due to an incorrect factor of 2 in the sample spacing passed to the SciPy function fft.rfftfreq. [#362](https://github.com/litebird/litebird_sim/pull/362). 

-   **Breaking change**: Change to the pointing API [#358](https://github.com/litebird/litebird_sim/pull/358), in detail:

    1. DetectorInfo has three new attributes: pol_angle_rad (polarization angle), pol_efficiency (polarization efficiency)and mueller_hwp (mueller matrix of the HWP).

    2. get_pointings() return only the orientation ψ of the detector, the polarization angle is a separate variable stored in the `Observation` class. The same class also handles the mueller_hwp for each detector, and it has a new bool variable `has_hwp` that is set to true if an HWP object is passed to `prepare_pointings()`.

    3. The mock vPTEP IMo has been updated accordingly.

    4. The `HWP` class has a new field called mueller, that contains the mueller matrix of the HWP.

    5. The function `scan_map()` now handles three possible algebras: (i) no HWP, (ii) ideal HWP, (iii) generic optical chain.

-   Implementation of distributing detectors across the MPI processes by grouping them according to given attributes [#334](https://github.com/litebird/litebird_sim/pull/334)

-   **Breaking change**: `PointingSys()` now requires `Observation` as an argument. And several functions to add pointing systematics are merged into `left_multiply_syst_quats()`.

-   **Breaking change**: Redefinition (ωt instead of 2ωt) of the hwp angle returned by `get_pointings()`. Change in the pointing returned by `Observation.get_pointings()`, now behaving as it was before this [commit](https://github.com/litebird/litebird_sim/pull/319/commits/b3bc3bb2049c152cc183d6cfc68f4598f5b93ec0). Documentation updated accordingly. [#340](https://github.com/litebird/litebird_sim/pull/340)

-   Restructure the manual and use a new, cleaner style [#342](https://github.com/litebird/litebird_sim/pull/342)

-   Module for including nonlinearity in the simulations [#331](https://github.com/litebird/litebird_sim/pull/331)

-   Improve the documentation of the binner and the destriper [#333](https://github.com/litebird/litebird_sim/pull/333)

-   Make the code compatible with Python 3.12 [#332](https://github.com/litebird/litebird_sim/pull/332)

-   plot_fp.py which visualizes focal plane and `DetectorInfo` is implemented. Also it can generate a dector list file by clicking visualized detectors. The function is executable by: `python -m litebird_sim.plot_fp` [#345](https://github.com/litebird/litebird_sim/pull/345)

-   Simulation.add_noise() uses self.random as default random number generator [#349](https://github.com/litebird/litebird_sim/pull/349)

-   Mbs updated, code aligned to pysm 3.4.0, CO lines included plus other new foreground models [#347](https://github.com/litebird/litebird_sim/pull/347)

# Version 0.13.0

-   **Breaking change**: new API for pointing computation [#319](https://github.com/litebird/litebird_sim/pull/319). Here is a in-depth list of all the breaking changes in this PR:

    1.  Quaternions describing the orientation of the detectors must now be encoded using a `RotQuaternion` object; plain NumPy arrays are no longer supported.

    2.  Quaternions are now computed using the function `prepare_pointings()` (low-level) and the method `Simulation.prepare_pointings()` (high-level, you should use this). Pointings are no longer kept in memory until you retrieve them using `Observation.get_pointings()`.

    3.  Pointings are no longer accessible using the field `pointings` in the `Observation` class. (Not 100% true, see below.) They are computed on the fly by the method `Observation.get_pointings()`.

    4.  The way pointings are returned differs from how they were stored before. The result of a call to `Observation.get_pointings()` is a 2-element tuple: the first element contains a `(N, 3)` NumPy array containing the colatitude θ, the longitude φ, and the orientation ψ, while the second element is an array of the angles of the HWP. Thus, the orientation angle ψ is now stored together with θ and φ.

    5.  If you want to pre-compute all the pointings instead of computing them on the fly each time you call `Observation.get_pointings()`, you can use the function `precompute_pointings()` (low-level) and the method `Simulation.precompute_pointings()` (high-level). This initializes a number of fields in each `Observation` object, but they are shaped as described in the previous point, i.e., ψ is kept in the same matrix as θ and φ.

    6.  The argument `dtype_tod` of the method `Simulation.create_observations` has become `tod_type` for consistency with other similar parameters.

    7.  The format of the HDF5 files has been slightly changed to let additional information about pointings to be stored.

    See the comments in [PR#319](https://github.com/litebird/litebird_sim/pull/319) and discussion [#312](https://github.com/litebird/litebird_sim/discussions/312) for more details.

-   Add data splits in time and detector space to destriped maps [#309](https://github.com/litebird/litebird_sim/pull/309)

-   Fix issue [#317](https://github.com/litebird/litebird_sim/issues/317)

-   Implement a time profiler [#308](https://github.com/litebird/litebird_sim/pull/308)

# Version 0.12.0

-   **Breaking change**: Disambiguate between “polarization angle” and “orientation” [#305](https://github.com/litebird/litebird_sim/pull/305). A few functions have been renamed as a consequence of this change; however, they are low-level functions that are used internally (`compute_pointing_and_polangle`, `all_compute_pointing_and_polangle`, `polarization_angle`), so external codes should be unaffected by this PR.

-   **Breaking change**: Reworking of the IO, `write_observations` and `read_observations` are now part of the class simulation [#293](https://github.com/litebird/litebird_sim/pull/293)

-   Mbs optionally returns alms instead of maps [#306](https://github.com/litebird/litebird_sim/pull/306)

-   Include the possibility to pass components to fill_tods,  add_dipole and add_noise [#302](https://github.com/litebird/litebird_sim/issues/302)

-   Add data splits in time and detector space to binned maps [#291](https://github.com/litebird/litebird_sim/pull/291)

-   Add support for partial multithreading using Numba [#276](https://github.com/litebird/litebird_sim/pull/276)

-   Fixing bug in mbs to pass general bandpass to mbs [#271](https://github.com/litebird/litebird_sim/pull/271)

-   Support for numpy.float128 made optional, this fixes importing issue on ARM architectures [#286](https://github.com/litebird/litebird_sim/pull/286)

-   Improve the documentation about noise simulations [#283](https://github.com/litebird/litebird_sim/pull/283)

-   Use libinsdb to access the IMO [#282](https://github.com/litebird/litebird_sim/pull/282)

-   Move from `flake8`/`black` to `ruff` [#281](https://github.com/litebird/litebird_sim/pull/281/)

-   New module to simulate HWP systematics [#232](https://github.com/litebird/litebird_sim/pull/232)

# Version 0.11.0

-   **Breaking change**: Change the interface to the binner, implement a new destriper, and make the dependency on TOAST optional [#260](https://github.com/litebird/litebird_sim/pull/260)

-   **Breaking change**: Drop support for Python 3.7 and 3.8 [#254](https://github.com/litebird/litebird_sim/pull/254)

-   **Breaking change**: Fix noise seed inconsistency [#256](https://github.com/litebird/litebird_sim/pull/256)

-   Be more robust when parsing UUIDs and URLs coming from the IMo [#274](https://github.com/litebird/litebird_sim/pull/274)

-   Solve typing error in destriper [#272](https://github.com/litebird/litebird_sim/pull/272)

-   Include default PTEP IMO for tests and demos [#230](https://github.com/litebird/litebird_sim/pull/230)

-   Fixed typo in timeordered.rst [#250](https://github.com/litebird/litebird_sim/pull/250)

-   Fix error in reading observation when it does not have tod field [#262](https://github.com/litebird/litebird_sim/pull/262) 

-   Bug in mbs for band integration solved [#251](https://github.com/litebird/litebird_sim/pull/251)

-   Implement a bandpass generator [#160](https://github.com/litebird/litebird_sim/pull/160), [#200](https://github.com/litebird/litebird_sim/pull/200)

# Version 0.10.0

-   Some memory optimization [#245](https://github.com/litebird/litebird_sim/pull/245)

-   Improve the docstring for `scan_map_in_observations` [#248](https://github.com/litebird/litebird_sim/pull/248)

-   New interface for `make_bin_map` in `Simulation` [#244](https://github.com/litebird/litebird_sim/pull/244)

-   Added gain drift simulation module [#243](https://github.com/litebird/litebird_sim/pull/243)

-   Enable the use of other names than `tod` when calling the TOAST2 destriper [#242](https://github.com/litebird/litebird_sim/pull/242)

-   Use Poetry instead of Pip to specify the dependencies for the documentation [#237](https://github.com/litebird/litebird_sim/pull/237)

-   Remove bandpass-related warnings [#236](https://github.com/litebird/litebird_sim/pull/236)

-   Add TOD interpolation [#233](https://github.com/litebird/litebird_sim/pull/233)

-   Improve the documentation [#231](https://github.com/litebird/litebird_sim/pull/231)

-   Mbs supports generic bandpasses and can generate solar dipole [#227](https://github.com/litebird/litebird_sim/pull/227)

-   Improve the support for multiple TODs in the same `Observation` [#225](https://github.com/litebird/litebird_sim/pull/225)


# Version 0.9.0

-   Some memory optimization in pointing production [#222](https://github.com/litebird/litebird_sim/pull/222), coordinate rotation and noise [#223](https://github.com/litebird/litebird_sim/pull/223)

-   Implement new methods in the `Simulation` class: `fill_tods`, `compute_pos_and_vel`, `add_dipole` and `add_noise` [#221](https://github.com/litebird/litebird_sim/pull/221)

-   **Breaking change**: add multiple TOD support to `describe_mpi_distribution` and make the field `MpiObservationDescr.tod_dtype` a list of strings [#220](https://github.com/litebird/litebird_sim/pull/220)

-   Add links to the manual in the example notebook [#219](https://github.com/litebird/litebird_sim/pull/219)

-   Implement new methods in the `Simulation` class: `set_scanning_strategy`, `set_instrument`, `set_hwp`, and deprecate `generate_spin2ecl_quaternions` [#217](https://github.com/litebird/litebird_sim/pull/217)

-   Add `gzip_compression` keyword to `write_observations` [#214](https://github.com/litebird/litebird_sim/pull/214)

-   Run more comprehensive tests on different TOD components [#212](https://github.com/litebird/litebird_sim/pull/212)

-   Add a link to the IMO webpage @SSDC for each entity/quantity/data file included in simulation reports [#211](https://github.com/litebird/litebird_sim/pull/211)

-   Fix issue #209 [#210](https://github.com/litebird/litebird_sim/pull/210)

-   Add flag for coordinate system choice of madam output maps [#208](https://github.com/litebird/litebird_sim/pull/208)

-   Improve support for multiple TODs [#205](https://github.com/litebird/litebird_sim/pull/205)

# Version 0.8.0

-   **Breaking change** Interface of `get_pointings` modified, new function `get_pointings_for_observation` simplifies the pointing generation for a list of observations [#198](https://github.com/litebird/litebird_sim/pull/198)

-   Ensure chronological order for Madam FITS files and make sure that exporting them to Madam works with MPI [#204](https://github.com/litebird/litebird_sim/pull/204) 

-   Properly install Madam template files [#202](https://github.com/litebird/litebird_sim/pull/202)

-   Mark installation errors for rich traceback in CI builds as non fatal [#199](https://github.com/litebird/litebird_sim/pull/199)

-   Fix bug in `make_bin_map` [#196](https://github.com/litebird/litebird_sim/pull/196)

# Version 0.7.0

-   Update and fix dependencies [#192](https://github.com/litebird/litebird_sim/pull/192)

-   Allow nnz=1 in the destriper [#191](https://github.com/litebird/litebird_sim/pull/191)

-   Improve the performance of the pointing generator [#190](https://github.com/litebird/litebird_sim/pull/190)

-   Add support for Madam (through an external call) [#186](https://github.com/litebird/litebird_sim/pull/186)

# Version 0.6.0

-   **Breaking change** The wrapper to the TOAST2 mapmaker has been fixed, and the parameter `baseline_length` was renamed to `baseline_length_s` to make clear what the measurement unit is [#182](https://github.com/litebird/litebird_sim/pull/182)

# Version 0.5.0

-   **Breaking change** New API for noise module [#151](https://github.com/litebird/litebird_sim/pull/151):

    -   Function `add_noise` has been renamed to `add_noise_to_observations`, and its parameter `noisetype` has been renamed into `noise_type` for consistency with other parameters (**breaking**)

    -   New functions `add_white_noise` and `add_one_over_f_noise` are exported (they were already implemented but were not visible)

    -   Each `Simulation` object creates random number generators (field `Simulation.random`), in a way that is safe even for MPI applications

-   **Breaking change** New API for `scan_map_in_observations` and `add_dipole_to_observations`, which now accept list of pointing matrices and simplify the parameters describing the HWP [#171](https://github.com/litebird/litebird_sim/pull/171)

-   Add a notebook to show an example of how to use the framework ([#178](https://github.com/litebird/litebird_sim/pull/178))

-   Support the production of maps in Galactic coordinates through the TOAST2 wrapper to the Madam map-maker ([#177](https://github.com/litebird/litebird_sim/pull/177))

-   Make `make_bin_map` compute pixel indices instead of requiring them as input, add support for Galactic coordinates [#176](https://github.com/litebird/litebird_sim/pull/176)

-   Use a more robust algorithm to compute pointings [#175](https://github.com/litebird/litebird_sim/pull/175)

-   Improve the documentation for the destriper [#172](https://github.com/litebird/litebird_sim/pull/172)

-   Add a high-pass filter for the noise [#169](https://github.com/litebird/litebird_sim/pull/169)

-   Upgrade NumPy from 1.20 to 1.21, Numba from 0.54 to 0.55, Rich from 6.2 to 11.0 [#152](https://github.com/litebird/litebird_sim/pull/152)

-   Add the ability to create Singularity container from branches different than `master` [#163](https://github.com/litebird/litebird_sim/pull/163)

-   Make MBS tests more robust against disappearing temporary directories [#162](https://github.com/litebird/litebird_sim/pull/162)

-   Remove NumPy's and Healpy's deprecation warnings [#158](https://github.com/litebird/litebird_sim/pull/158)

-   Use a cache to speed up CI builds [PR#147](https://github.com/litebird/litebird_sim/pull/147)

-   Create a script that fetches information about the latest release and produce a release announcement [PR#156](https://github.com/litebird/litebird_sim/pull/156)

-   Option for rotating the pointing from ecliptic to galactic coordinates in scan_map [#164](https://github.com/litebird/litebird_sim/pull/164)

-   Fix issue [#148](https://github.com/litebird/litebird_sim/issues/148)

# Version 0.4.0

- **Breaking change** Drop support for Python 3.6, enable Python 3.9 [#136](https://github.com/litebird/litebird_sim/pull/136)

- **Breaking change** Rename keyword `distribute` to `split_list_over_processes` in `Simulation.create_observations` [#110](https://github.com/litebird/litebird_sim/pull/110)

- **Breaking change** Switch to thermodynamic units in the MBS module [#123](https://github.com/litebird/litebird_sim/pull/123)

- Functions to write/load TODs to HDF5 files [#139](https://github.com/litebird/litebird_sim/pull/139) 

- Module for simulating hwp systematics (hwp_sys) [PR#117](https://github.com/litebird/litebird_sim/pull/117). The algebra is described in [Giardiello et al.](https://arxiv.org/abs/2106.08031)

- Fix Singularity builds [#145](https://github.com/litebird/litebird_sim/issues/145)

- Make the TOAST destriper more robust when MPI is/isn't present [#106](https://github.com/litebird/litebird_sim/pull/106)

- Option in Mbs for maps in ecliptic coordinates [#133](https://github.com/litebird/litebird_sim/pull/133)

- Module for scanning a map and filling TOD [#131](https://github.com/litebird/litebird_sim/pull/131)

# Version 0.3.0

- Spacecraft simulator and dipole computation [#122](https://github.com/litebird/litebird_sim/pull/122)

- Improve the way code is checked [#130](https://github.com/litebird/litebird_sim/pull/130)

- Fix bugs [#126](https://github.com/litebird/litebird_sim/issues/126), [#124](https://github.com/litebird/litebird_sim/issues/124), [#120](https://github.com/litebird/litebird_sim/issues/120), [#111](https://github.com/litebird/litebird_sim/pull/111)

# Version 0.2.1

- Fix bug [#107](https://github.com/litebird/litebird_sim/pull/107) [PR#108](https://github.com/litebird/litebird_sim/pull/108)

# Version 0.2.0

- White and 1/f noise generation [PR#100](https://github.com/litebird/litebird_sim/pull/100)

- Fix bug #104 [PR#105](https://github.com/litebird/litebird_sim/pull/105)

# Version 0.2.0 alpha

- Add a text-mode browser for the IMO [PR#103](https://github.com/litebird/litebird_sim/pull/103)

- Implement the tools to build a Singularity container [PR#96](https://github.com/litebird/litebird_sim/pull/96/)

- Implement an interface to the TOAST mapmaker [PR#86](https://github.com/litebird/litebird_sim/pull/86/)

- Fix issue [#101](https://github.com/litebird/litebird_sim/issues/101) (*No proper "parents" in Entry objects*) [PR#102](https://github.com/litebird/litebird_sim/pull/102)

- Make the README point to the latest version of the documentation [PR#99](https://github.com/litebird/litebird_sim/pull/99)

- Ensure that tests do not write within the source-code directory [PR#97](https://github.com/litebird/litebird_sim/pull/97)

- Add a [`STATUS.md`](https://github.com/litebird/litebird_sim/blob/be2ddfc3dfcc8d98711de72c56d8bc140bf8e7ce/STATUS.md) file showing the overall implementation status of the simulation modules [PR#87](https://github.com/litebird/litebird_sim/pull/87)

- Clarify how the IMO is used by `litebird_sim` [PR#94](https://github.com/litebird/litebird_sim/pull/94)

- Make tests run faster by using ducc0 0.8.0 [PR#92](https://github.com/litebird/litebird_sim/pull/92)

- Misc minor changes: gitignore .DS_Store; losslessly compress some assets [PR#88](https://github.com/litebird/litebird_sim/pull/88)

- Improve the `Observation` API. Deprecate the pointing-related methods (moved to `scanning`), quantities are local by default [PR#84](https://github.com/litebird/litebird_sim/pull/84)

- Permit to use pre-allocated buffers when generating quaternions and pointing angles [PR#83](https://github.com/litebird/litebird_sim/pull/83)

- Add support for PySM3 in new class `Mbs` [PR#76](https://github.com/litebird/litebird_sim/pull/76)

- Add the parameter `include_git_diff` in `Simulation.flush()` [PR#81](https://github.com/litebird/litebird_sim/pull/81)

- Add the ability to specify the size of the floating-point type used in `Observation` objects [PR#79](https://github.com/litebird/litebird_sim/pull/79)

- Simple bin map-maker [PR#73](https://github.com/litebird/litebird_sim/pull/73)

- Use SI units in class `SpinningScanningStrategy` (**breaking change**) [PR#69](https://github.com/litebird/litebird_sim/pull/69)

- Use dataclasses, rename `Detector` to `DetectorInfo` and `Instrument` to `InstrumentInfo` (**breaking change**) [PR#60](https://github.com/litebird/litebird_sim/pull/60)

- Improve the docs [PR#72](https://github.com/litebird/litebird_sim/pull/72), [PR#82](https://github.com/litebird/litebird_sim/pull/82)

- Code cleanups [PR#71](https://github.com/litebird/litebird_sim/pull/71)

- Improve the README [PR#70](https://github.com/litebird/litebird_sim/pull/70)

- [Fix issue #61](https://github.com/litebird/litebird_sim/pull/62)

# Version 0.1.0

- First release
