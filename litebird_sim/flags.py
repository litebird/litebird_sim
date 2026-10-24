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

import astropy.time
import numpy as np
import numpy.typing as npt
from astropy.coordinates import ICRS, get_body_barycentric
from numba import njit

from .coordinates import DEFAULT_COORDINATE_SYSTEM
from .observations import Observation
from .quaternions import all_rotate_z_vectors
from .scanning import RotQuaternion
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


#: Planets that can enter the field of view from L2 (Mercury and Venus are
#: always too close to the Sun)
PLANETS = ("mars", "jupiter", "saturn", "uranus", "neptune")

# The ephemerides are computed every _EPHEMERIS_STEP_S seconds and
# interpolated linearly, and the boresight is sampled every
# _PREFILTER_STEP_S seconds to find the planets in the field of view
_EPHEMERIS_STEP_S = 3600.0
_PREFILTER_STEP_S = 1.0


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
            theta = pointings[det_idx, sample_idx, 0]
            phi = pointings[det_idx, sample_idx, 1]
            sin_theta = np.sin(theta)
            cos_angle = (
                sin_theta * np.cos(phi) * planet_dir[sample_idx, 0]
                + sin_theta * np.sin(phi) * planet_dir[sample_idx, 1]
                + np.cos(theta) * planet_dir[sample_idx, 2]
            )
            if cos_angle > cos_radius:
                flags[det_idx, sample_idx] |= bit
                num_of_flagged += 1
    return num_of_flagged


def _ang2vec(theta: npt.NDArray, phi: npt.NDArray) -> npt.NDArray:
    sin_theta = np.sin(theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=-1
    )


def _planet_ephemeris(
    planet: str,
    pos_and_vel: SpacecraftPositionAndVelocity,
    start_time: astropy.time.Time,
    time_span_s: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Position of `planet` with respect to the spacecraft, in km and in
    Ecliptic coordinates, on a grid of times (seconds since `start_time`)
    covering `time_span_s`"""
    # ponytail: no light-time correction, the planets move < 0.01° meanwhile
    grid_s = np.arange(-1, time_span_s / _EPHEMERIS_STEP_S + 2) * _EPHEMERIS_STEP_S
    times = start_time + astropy.time.TimeDelta(grid_s, format="sec")
    planet_km = (
        ICRS(get_body_barycentric(planet, times))
        .transform_to(DEFAULT_COORDINATE_SYSTEM)
        .cartesian.get_xyz()
        .to("km")
        .value.T
    )

    # Same interpolation as SpacecraftPositionAndVelocity.compute_velocities
    t = (start_time - pos_and_vel.start_time).sec + grid_s
    tp = np.linspace(0.0, pos_and_vel.time_span_s, pos_and_vel.positions_km.shape[0])
    spacecraft_km = np.stack(
        [np.interp(t, tp, pos_and_vel.positions_km[:, i]) for i in range(3)], axis=-1
    )
    return grid_s, planet_km - spacecraft_km


def _planet_directions(
    grid_s: npt.NDArray, relative_pos_km: npt.NDArray, times_s: npt.NDArray
) -> npt.NDArray:
    """Unit vectors pointing to the planet at times `times_s`"""
    pos = np.stack(
        [np.interp(times_s, grid_s, relative_pos_km[:, i]) for i in range(3)], axis=-1
    )
    return pos / np.linalg.norm(pos, axis=1, keepdims=True)


def _max_detector_offset_rad(obs: Observation) -> float:
    """Largest angle between the boresight and the detectors of `obs`"""
    max_offset = 0.0
    for quat in obs.quat:
        z_vectors = np.empty((quat.quats.shape[0], 3))
        all_rotate_z_vectors(z_vectors, quat.quats)
        max_offset = max(max_offset, np.arccos(np.clip(z_vectors[:, 2], -1, 1)).max())
    return max_offset


def _segments_close_to_planet(
    obs: Observation,
    grid_s: npt.NDArray,
    relative_pos_km: npt.NDArray,
    max_distance_rad: float,
) -> list[tuple[int, int]]:
    """Ranges of samples ``[start, stop)`` in which some detector of `obs`
    can be closer than `max_distance_rad` to the planet

    The boresight is sampled every _PREFILTER_STEP_S seconds: a segment
    between two consecutive points is kept if its first point is closer to the
    planet than `max_distance_rad` plus the largest offset of the detectors
    and the largest motion of the boresight between two points.
    """
    step = max(1, round(_PREFILTER_STEP_S * obs.sampling_rate_hz))
    num_of_points = (obs.n_samples - 1) // step + 1
    if num_of_points < 2:
        return [(0, obs.n_samples)]

    bore, _ = obs.pointing_provider.get_pointings(
        detector_quat=RotQuaternion(),
        start_time=obs.start_time,
        start_time_global=obs.start_time_global,
        sampling_rate_hz=obs.sampling_rate_hz / step,
        nsamples=num_of_points,
        compute_hwp_angle=False,
    )
    bore_vec = _ang2vec(bore[:, 0], bore[:, 1])
    max_motion = np.arccos(
        np.clip(np.sum(bore_vec[1:] * bore_vec[:-1], axis=1), -1, 1)
    ).max()

    times_s = np.arange(num_of_points) * step / obs.sampling_rate_hz
    planet_vec = _planet_directions(grid_s, relative_pos_km, times_s)
    distance = np.arccos(np.clip(np.sum(bore_vec * planet_vec, axis=1), -1, 1))
    close = distance < max_distance_rad + _max_detector_offset_rad(obs) + max_motion

    # Point k covers samples [k * step, (k + 1) * step); join consecutive points
    edges = np.flatnonzero(np.diff(np.concatenate([[0], close.astype(int), [0]])))
    return [
        (start * step, min(stop * step, obs.n_samples))
        for start, stop in zip(edges[::2], edges[1::2])
    ]


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

    num_of_flagged = dict.fromkeys(planets, 0)
    for obs in obs_list:
        if not isinstance(obs.start_time, astropy.time.Time):
            raise ValueError("Flagging planets needs times in astropy.time.Time")
        if np.any(obs.fwhm_arcmin <= 0):
            raise ValueError("Flagging planets needs fwhm_arcmin > 0 for all detectors")

        flags = get_local_flags(obs)
        if obs.n_samples == 0 or obs.n_detectors == 0:
            continue

        fwhm_rad = np.deg2rad(np.asarray(obs.fwhm_arcmin) / 60.0)
        time_span_s = obs.n_samples / obs.sampling_rate_hz
        for planet in planets:
            k = radius_fwhm[planet] if isinstance(radius_fwhm, dict) else radius_fwhm
            radius_rad = k * fwhm_rad
            grid_s, relative_pos_km = _planet_ephemeris(
                planet, pos_and_vel, obs.start_time, time_span_s
            )

            for start, stop in _segments_close_to_planet(
                obs, grid_s, relative_pos_km, radius_rad.max()
            ):
                pointings = np.empty((obs.n_detectors, stop - start, 3))
                for det_idx in range(obs.n_detectors):
                    obs.pointing_provider.get_pointings(
                        detector_quat=obs.quat[det_idx],
                        start_time=obs.start_time
                        + astropy.time.TimeDelta(
                            start / obs.sampling_rate_hz, format="sec"
                        ),
                        start_time_global=obs.start_time_global,
                        sampling_rate_hz=obs.sampling_rate_hz,
                        nsamples=stop - start,
                        pointing_buffer=pointings[det_idx],
                        compute_hwp_angle=False,
                    )
                times_s = np.arange(start, stop) / obs.sampling_rate_hz
                num_of_flagged[planet] += flag_planet(
                    flags[:, start:stop],
                    pointings,
                    _planet_directions(grid_s, relative_pos_km, times_s),
                    radius_rad,
                    bit,
                )

    return num_of_flagged
