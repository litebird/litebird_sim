# HEAD

-   **Breaking change**: Drop support for Python 3.7 and 3.8 [#254](https://github.com/litebird/litebird_sim/pull/254)

-   Fix noise seed inconsistency [#256](https://github.com/litebird/litebird_sim/pull/256)

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
