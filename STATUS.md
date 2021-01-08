# Implementation status

This document details the list of simulation modules that the
Simulation Team plans to implement in `litebird_sim`.

## List of modules

| Module                           | Status   | Priority   | Notes                           |
| -------------------------------- | -------- | ---------- | ------------------------------- |
| Pointing generation              | Complete |            |                                 |
| Interface with Ephemerides       | Complete |            | Through AstroPy                 |
| Synthetic sky map generation     | Complete |            | Based on PySM3                  |
| Binning map-maker                | Complete |            |                                 |
| Beam convolution                 | Partial  |            | Through ducc0                   |
| Destriping+calibration map-maker | Partial  |            | Provided by TOAST               |
| Calibration non-idealities       | Partial  |            | Code in toast-litebird          |
| Cosmic-ray glitch generation     | Partial  |            |                                 |
| HWP simulation                   | Partial  |            |                                 |
| ADC simulation                   | Partial  |            | Through the CR glitch generator |
| Map scanning                     | Missing  |            |                                 |
| White+1/f noise generation       | Missing  |            |                                 |
| Correlated noise generation      | Missing  |            |                                 |
| Dipole calibration               | Missing  |            |                                 |

## Beam convolution

-   `ducc0` already provides a 4π convolution code, and it is already
    available within `litebird_sim`
-   A high-level interface to `ducc0` is still missing

## Destriping+calibration map-maker

-   Provided by TOAST2
-   [PR#86](https://github.com/litebird/litebird_sim/pull/86)

## Calibration non-idealities

-   Code available in toast-litebird
-   Still not integrated in `litebird_sim`
-   No PR yet

## Cosmic-ray glitch generation

-   Simulation code available
-   Not integrated with the IMO nor with `litebird_sim`
-   No PR yet

## HWP simulation

-   A mathematical model is already available
-   Code is being written by Luca Pagano and Serena Giardiello
-   No PR yet

## ADC simulation

-   Need to simulate the following effects:

    -   Signal quantization
    
    -   Clipping of the signal outside the dynamic range
    
    -   Non-linearity effects
    
-   Signal clipping is already available in the Cosmic-ray glitch
    generator (see above)

## Map scanning

-   Not implemented yet

## White+1/f noise generation

-   Work is in progress (Mathew Galloway)

-   No PRs yet

## Correlated noise generation

-   Not implemented yet

## Dipole calibration

-   The interface with Ephemerides codes is already in place

-   No PRs yet
