# HEAD

-   **Breaking change** New API for noise module [#151](https://github.com/litebird/litebird_sim/pull/151):

    -   Function `add_noise` has been renamed to `add_noise_to_observations`, and its parameter `noisetype` has been renamed into `noise_type` for consistency with other parameters (**breaking**)

    -   New functions `add_white_noise` and `add_one_over_f_noise` are exported (they were already implemented but were not visible)

    -   Each `Simulation` object creates random number generators (field `Simulation.random`), in a way that is safe even for MPI applications

-   Upgrade NumPy from 1.20 to 1.21, Numba from 0.54 to 0.55, Rich from 6.2 to 11.0 [#152](https://github.com/litebird/litebird_sim/pull/152)

-   Fix issue [#148](https://github.com/litebird/litebird_sim/issues/148)

# Version 0.4.0

- **Breaking change** Drop support for Python 3.6, enable Python 3.9 [#136](https://github.com/litebird/litebird_sim/pull/136)

- **Breaking change** Rename keyword `distribute` to `split_list_over_processes` in `Simulation.create_observations` [#110](https://github.com/litebird/litebird_sim/pull/110)

- **Breaking change** Switch to thermodynamic units in the MBS module [#123](https://github.com/litebird/litebird_sim/pull/123)

- Functions to write/load TODs to HDF5 files [#139](https://github.com/litebird/litebird_sim/pull/139) 

- Module for simulating hwp systematics (hwp_sys) [PR#117](https://github.com/litebird/litebird_sim/pull/117). The algebra is described in [Giardiello et al.](https://arxiv.org/abs/2106.08031)

- Fix Singularity builds [#145](https://github.com/litebird/litebird_sim/issues/145)

- Make the TOAST destriper more robust when MPI is/isn't present [#106](https://github.com/litebird/litebird_sim/pull/106)

- Option in Mbs for maps in ecliptoc coordinates [#133](https://github.com/litebird/litebird_sim/pull/133)

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
