# Implementation status

This document details the list of simulation modules that the
Simulation Team plans to implement in `litebird_sim`.

## List of modules

| Module                           | Status   | Priority   | Notes                           | Links                                                                                           |
| -------------------------------- | -------- | ---------- | ------------------------------- | ---------------------------------------------------------                                       |
| Pointing generation              | Complete |            |                                 | [PR#48](https://github.com/litebird/litebird_sim/pull/48)                                       |
| Interface with Ephemerides       | Complete |            | Through AstroPy                 | [PR#48](https://github.com/litebird/litebird_sim/pull/48)                                       |
| Synthetic sky map generation     | Complete |            | Based on PySM3                  | [PR#76](https://github.com/litebird/litebird_sim/pull/76)                                       |
| Binning map-maker                | Complete |            |                                 | [PR#73](https://github.com/litebird/litebird_sim/pull/76)                                       |
| Destriping+calibration map-maker | Complete |            | Provided by TOAST               | [PR#86](https://github.com/litebird/litebird_sim/pull/86)                                       |
| Spacecraft simulator             | Complete |            |                                 | [PR#122](https://github.com/litebird/litebird_sim/pull/122)                                     |
| Dipole calculation               | Complete |            |                                 | [PR#122](https://github.com/litebird/litebird_sim/pull/122)                                     |
| Map scanning                     | Complete |            |                                 | [PR#131](https://github.com/litebird/litebird_sim/pull/131)                                    |
| White+1/f noise generation       | Complete |            |                                 | [PR#100](https://github.com/litebird/litebird_sim/pull/100)                                    |
| Beam convolution                 | Partial  |            | Through ducc0                   | [ducc.totalconvolve](https://gitlab.mpcdf.mpg.de/mtr/ducc/-/tree/ducc0/)                        |
| Calibration non-idealities       | Partial  |            | Code in toast-litebird          | [`OpGainDrifter`](https://github.com/hpc4cmb/toast-litebird/blob/master/toast_litebird/gain.py) |
| Cosmic-ray glitch generation     | Partial  |            |                                 | No PRs yet                                                                                      |
| HWP simulation                   | Partial  |            |                                 | [PR#117](https://github.com/litebird/litebird_sim/pull/117)                                    |
| ADC simulation                   | Partial  |            | Through the CR glitch generator | No PRs yet                                                                                      |
| Correlated noise generation      | Missing  |            |                                 |                                                                                                |
| Dipole calibration               | Missing  |            |                                 |                                                                                                |

## Beam convolution

-   `ducc0` already provides a 4π convolution code, and it is already
    available within `litebird_sim`
-   A high-level interface to `ducc0` is still missing

## Destriping+calibration map-maker

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
