# Implementation status

This document details the list of simulation modules that the
Simulation Team plans to implement in `litebird_sim`.

## List of modules

| Module                           | Status   | Priority | Notes                           | Links                                                                                                                |
|----------------------------------|----------|----------|---------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Pointing generation              | Complete |          |                                 | [#48](https://github.com/litebird/litebird_sim/pull/48)                                                              |
| Interface with Ephemerides       | Complete |          | Through AstroPy                 | [#48](https://github.com/litebird/litebird_sim/pull/48)                                                              |
| Synthetic sky map generation     | Complete |          | Based on PySM3                  | [#76](https://github.com/litebird/litebird_sim/pull/76)                                                              |
| Binning map-maker                | Complete |          |                                 | [#73](https://github.com/litebird/litebird_sim/pull/76)                                                              |
| Destriping+calibration map-maker | Complete |          | Internal destriper +            | [#260](https://github.com/litebird/litebird_sim/pull/260)                                                            |
|                                  |          |          | interface with TOAST and Madam  | [#86](https://github.com/litebird/litebird_sim/pull/86)                                                              |
|                                  |          |          |                                 | [#186](https://github.com/litebird/litebird_sim/pull/186)                                                            |
| Splits in map-makers             | Complete |          |                                 | [#291](https://github.com/litebird/litebird_sim/pull/291)                                                            |
| Spacecraft simulator             | Complete |          |                                 | [#122](https://github.com/litebird/litebird_sim/pull/122)                                                            |
| Dipole calculation               | Complete |          |                                 | [#122](https://github.com/litebird/litebird_sim/pull/122)                                                            |
| Map scanning                     | Complete |          |                                 | [#131](https://github.com/litebird/litebird_sim/pull/131)                                                            |
| White+1/f noise generation       | Complete |          |                                 | [#100](https://github.com/litebird/litebird_sim/pull/100)                                                            |
| Gain drift simulation            | Complete |          |                                 | [#243](https://github.com/litebird/litebird_sim/pull/243)                                                            |
| Synthetic bandpass generation    | Complete |          |                                 | [#160](https://github.com/litebird/litebird_sim/pull/160), [#200](https://github.com/litebird/litebird_sim/pull/200) |
| Calibration non-idealities       | Complete |          |                                 | [#243](https://github.com/litebird/litebird_sim/pull/243)                                                            |
| Pointing systematics             | Complete |          |                                 | [#319](https://github.com/litebird/litebird_sim/pull/319)                                                            |
| Beam convolution                 | Partial  |          | Through ducc0                   | [ducc.totalconvolve](https://gitlab.mpcdf.mpg.de/mtr/ducc/-/tree/ducc0/)                                             |
| Cosmic-ray glitch generation     | Partial  |          |                                 | No PRs yet                                                                                                           |
| HWP simulation                   | Partial  |          |                                 | [#117](https://github.com/litebird/litebird_sim/pull/117)                                                            |
| ADC simulation                   | Partial  |          | Through the CR glitch generator | No PRs yet                                                                                                           |
| Correlated noise generation      | Missing  |          |                                 |                                                                                                                      |
| Dipole calibration               | Missing  |          |                                 |                                                                                                                      |

## Beam convolution

-   `ducc0` already provides a 4π convolution code, and it is already available within `litebird_sim`
-   A high-level interface to `ducc0` is still missing

## Destriping+calibration map-maker

-   Internal destriper, [#260](https://github.com/litebird/litebird_sim/pull/260)
-   Provided by TOAST2, [PR#86](https://github.com/litebird/litebird_sim/pull/86)
-   [PR#186](https://github.com/litebird/litebird_sim/pull/186) adds the possibility to interface Madam

## Calibration non-idealities

-   Code available in toast-litebird
-   Still not integrated in `litebird_sim`
-   No PR yet

## Cosmic-ray glitch generation

-   Simulation code available
-   Not integrated with the IMO nor with `litebird_sim`
-   No PR yet

## HWP simulation

-   A mathematical model is already available, based on [Giardiello et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021arXiv210608031G/abstract).
-   [PR#117](https://github.com/litebird/litebird_sim/pull/117).
-   This supports both Jones and Mueller formalisms, and it's interfaced with the bandpass generation module

## ADC simulation

-   Need to simulate the following effects:
    -   Signal quantization
    -   Clipping of the signal outside the dynamic range
    -   Non-linearity effects
-   Signal clipping is already available in the Cosmic-ray glitch
    generator (see above)

## Correlated noise generation

-   Not implemented yet

## Dipole calibration

-   The interface with Ephemerides codes is already in place

-   No PRs yet
