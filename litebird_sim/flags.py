"""Flags of the samples in the TODs.

Flags are bitmasks of dtype :data:`FLAG_DTYPE`: zero means a good sample,
and each bit is a different reason why a sample is flagged. Modules that
flag samples set their own bit with ``|=``, and consumers select the
reasons to exclude with ``(flags & mask) != 0``.

``Observation.local_flags`` (shape ``(n_detectors, n_samples)``) and
``Observation.global_flags`` (shape ``(n_samples,)``) are ``None`` until
somebody writes them, meaning that all the samples are good: use
:func:`get_local_flags` and :func:`get_global_flags` to allocate them on
the first write.
"""

import numpy as np
import numpy.typing as npt
from numba import njit

from .observations import Observation
from .planets import (
    PLANETS,
    _angle_to_planet,
    check_observations_for_planets,
    iter_planet_segments,
)
from .spacecraft import SpacecraftPositionAndVelocity

#: dtype of ``Observation.local_flags`` and ``Observation.global_flags``
FLAG_DTYPE = np.uint16

# Bits of the flags, one per reason

#: Sample close to a planet
FLAG_PLANET = 1 << 0


def _local_shape(obs: Observation) -> tuple[int, int]:
    # Same shape as the TODs, also when they are not allocated
    return obs._get_tod_shape(obs.n_blocks_det, obs.n_blocks_time)


def get_local_flags(obs: Observation) -> npt.NDArray:
    """Return ``obs.local_flags``, allocating it with zeros if it is ``None``"""
    if obs.local_flags is None:
        obs.local_flags = np.zeros(_local_shape(obs), dtype=FLAG_DTYPE)
    return obs.local_flags


def get_global_flags(obs: Observation) -> npt.NDArray:
    """Return ``obs.global_flags``, allocating it with zeros if it is ``None``"""
    if obs.global_flags is None:
        obs.global_flags = np.zeros(_local_shape(obs)[1], dtype=FLAG_DTYPE)
    return obs.global_flags


@njit
def flag_planet(flags, pointings, planet_dir, radius_rad, bit=FLAG_PLANET) -> int:
    """Set `bit` in the samples closer than `radius_rad` to a planet

    `flags` (``(n_detectors, n_samples)``) is modified in place; `pointings`
    (``(n_detectors, n_samples, ≥2)``) contains θ and φ, `planet_dir`
    (``(n_samples, 3)``) the unit vectors pointing to the planet in the same
    coordinates, and `radius_rad` (``(n_detectors,)``) the radius around the
    planet for each detector. Return the number of flagged samples.
    """
    num_of_flagged = 0
    for det_idx in range(flags.shape[0]):
        cos_radius = np.cos(radius_rad[det_idx])
        for sample_idx in range(flags.shape[1]):
            if (
                _angle_to_planet(pointings, planet_dir, det_idx, sample_idx)
                > cos_radius
            ):
                flags[det_idx, sample_idx] |= bit
                num_of_flagged += 1
    return num_of_flagged


def flag_planets_in_observations(
    observations: Observation | list[Observation],
    pos_and_vel: SpacecraftPositionAndVelocity,
    planets: str | list[str] | tuple[str, ...] = PLANETS,
    radius_fwhm: float | dict[str, float] = 3.0,
    bit: int = FLAG_PLANET,
) -> dict[str, int]:
    """Flag the samples close to the planets

    Set `bit` in ``local_flags`` (allocated if needed) for the samples whose
    pointing is closer than ``radius_fwhm × fwhm_arcmin`` of the detector to
    one of the `planets`, seen from the position of the spacecraft in
    `pos_and_vel`. `radius_fwhm` can be a dictionary with the radius of each
    planet.

    The pointings of the detectors are computed only where some of them can
    see a planet: the boresight is first sampled every second to find the
    segments of time close to the planets. The observations must have a
    pointing provider (see :meth:`.Simulation.prepare_pointings`), times in
    ``astropy.time.Time`` and ``fwhm_arcmin``.

    Return the number of flagged samples (over all the detectors) for each
    planet, local to this MPI process.
    """
    obs_list = [observations] if isinstance(observations, Observation) else observations
    planets = [planets] if isinstance(planets, str) else list(planets)
    check_observations_for_planets(obs_list)

    num_of_flagged = dict.fromkeys(planets, 0)
    for obs in obs_list:
        flags = get_local_flags(obs)
        fwhm_rad = np.deg2rad(np.asarray(obs.fwhm_arcmin) / 60.0)
        for planet in planets:
            k = radius_fwhm[planet] if isinstance(radius_fwhm, dict) else radius_fwhm
            radius_rad = k * fwhm_rad
            for start, stop, pointings, planet_dir, _ in iter_planet_segments(
                obs, pos_and_vel, planet, radius_rad.max()
            ):
                num_of_flagged[planet] += flag_planet(
                    flags[:, start:stop], pointings, planet_dir, radius_rad, bit
                )

    return num_of_flagged
