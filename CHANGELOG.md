# HEAD

# Version 0.2.0

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
