"""Planets: ephemerides seen from the spacecraft and emission.

The direction and distance of the planets are computed from the position of
the spacecraft (see :class:`.SpacecraftPositionAndVelocity`), so that the
parallax is taken into account. The emission is a table of whole-disc
thermodynamic temperatures with respect to blank sky, i.e., with the CMB
monopole occulted by the planet already subtracted, which is what enters a
differential TOD.
"""

from collections.abc import Iterator

import astropy.time
import numpy as np
import numpy.typing as npt
from astropy.coordinates import ICRS, get_body_barycentric
from numba import njit

from .constants import H_OVER_K_B, T_CMB_K
from .coordinates import DEFAULT_COORDINATE_SYSTEM
from .observations import Observation
from .quaternions import all_rotate_z_vectors
from .scanning import RotQuaternion
from .spacecraft import SpacecraftPositionAndVelocity

#: Planets that can enter the field of view from L2 (Mercury and Venus are
#: always too close to the Sun)
PLANETS = ("mars", "jupiter", "saturn", "uranus", "neptune")

#: Equatorial and polar radii of the planets in km (Seidelmann et al. 2007,
#: as in Table 2 of Weiland et al. 2011, ApJS 192, 19)
PLANET_RADII_KM = {
    "mars": (3396.0, 3376.0),
    "jupiter": (71492.0, 66854.0),
    "saturn": (60268.0, 54364.0),
    "uranus": (25559.0, 24973.0),
    "neptune": (24764.0, 24341.0),
}


def _rj_to_thermodynamic(t_rj_k, freq_ghz):
    # B(ν, T) = 2kν²/c² T_RJ
    x = H_OVER_K_B * np.asarray(freq_ghz) * 1e9
    return x / np.log1p(x / np.asarray(t_rj_k))


# Whole-disc temperatures with respect to blank sky, as (frequencies in GHz,
# thermodynamic temperatures in K):
#
# - 100–857 GHz: Planck HFI, mean over the surveys (Table 4 of Planck
#   intermediate results LII, A&A 607, A122, 2017, arXiv:1612.07151)
# - < 100 GHz for Jupiter and Saturn: Planck LFI (Table 3 of LII; for Saturn
#   the rings are subtracted, while they are included in the HFI values)
# - < 100 GHz for Mars and Uranus: WMAP, Rayleigh-Jeans temperatures (Tables
#   5 and 11 of Weiland et al. 2011, ApJS 192, 19; for Mars, the mean of the
#   seven seasons; for Uranus, only the bands with errors < 5%)
# - Neptune: no measurement below 100 GHz with errors < 5%
#
# ponytail: one temperature per planet and frequency; Mars changes by ±5%
# with its distance from the Sun and Saturn by a few % with the inclination
# of the rings. Use a physical model if this matters.
PLANET_TEMPERATURES_K = {
    "mars": (
        np.array([22.85, 33.11, 40.82, 60.85, 93.32, 100, 143, 217, 353, 545, 857]),
        np.concatenate(
            [
                _rj_to_thermodynamic(
                    [186.1, 187.3, 189.4, 193.4, 195.1],
                    [22.85, 33.11, 40.82, 60.85, 93.32],
                ),
                [194.3, 198.4, 201.9, 209.9, 209.2, 213.5],
            ]
        ),
    ),
    "jupiter": (
        np.array([28.4, 44.1, 70.4, 100, 143, 217, 353, 545, 857]),
        np.array([146.6, 160.9, 173.3, 172.3, 173.6, 174.7, 166.3, 136.5, 160.3]),
    ),
    "saturn": (
        np.array([28.4, 44.1, 70.4, 100, 143, 217, 353, 545, 857]),
        np.array([138.9, 147.3, 150.6, 145.7, 147.0, 144.9, 141.5, 102.4, 115.5]),
    ),
    "uranus": (
        np.array([40.82, 60.85, 93.32, 100, 143, 217, 353, 545, 857]),
        np.concatenate(
            [
                _rj_to_thermodynamic([153.0, 138.0, 120.0], [40.82, 60.85, 93.32]),
                [120.5, 108.4, 98.5, 86.2, 73.9, 66.2],
            ]
        ),
    ),
    "neptune": (
        np.array([100, 143, 217, 353, 545, 857]),
        np.array([117.4, 106.4, 97.4, 82.6, 72.3, 65.3]),
    ),
}

# The ephemerides are computed every _EPHEMERIS_STEP_S seconds and
# interpolated linearly, and the boresight is sampled every
# _PREFILTER_STEP_S seconds to find the planets in the field of view
_EPHEMERIS_STEP_S = 3600.0
_PREFILTER_STEP_S = 1.0


def planet_temperature_k(planet: str, freq_ghz) -> npt.NDArray:
    """Thermodynamic temperature of `planet` with respect to blank sky

    The values in :data:`PLANET_TEMPERATURES_K` are interpolated linearly in
    log ν, and kept constant outside the range of the measurements.
    """
    freqs, temperatures = PLANET_TEMPERATURES_K[planet]
    return np.interp(np.log(freq_ghz), np.log(freqs), temperatures)


def planet_brightness_k_cmb(
    planet: str, bandcenter_ghz: float, bandwidth_ghz: float, num_of_freqs: int = 101
) -> float:
    """Temperature in K_CMB of a source that fills the beam with the
    emission of `planet`, for a top-hat band

    This is the ratio between the band integral of B(ν, T_planet(ν)) and
    that of ∂B/∂T at the temperature of the CMB. The signal of the planet
    at the centre of the beam is this value times Ω_planet / Ω_beam.
    """
    freqs_hz = 1e9 * np.linspace(
        bandcenter_ghz - bandwidth_ghz / 2,
        bandcenter_ghz + bandwidth_ghz / 2,
        num_of_freqs,
    )
    x = H_OVER_K_B * freqs_hz
    x_planet = x / planet_temperature_k(planet, freqs_hz / 1e9)
    x_cmb = x / T_CMB_K
    # Both B(ν, T) and T ∂B/∂T are proportional to ν³ times these factors
    radiance = 1 / np.expm1(x_planet)
    d_radiance_dt = x_cmb * np.exp(x_cmb) / np.expm1(x_cmb) ** 2 / T_CMB_K
    weights = freqs_hz**3
    return np.trapezoid(weights * radiance, freqs_hz) / np.trapezoid(
        weights * d_radiance_dt, freqs_hz
    )


def planet_solid_angle_sr(planet: str, distance_km) -> npt.NDArray:
    """Solid angle of the disc of `planet` at a distance `distance_km`

    ponytail: πr_eq r_pol / R², neglecting the projection of the polar radius
    (< 2% for Saturn, less for the others)
    """
    r_eq, r_pol = PLANET_RADII_KM[planet]
    return np.pi * r_eq * r_pol / np.asarray(distance_km) ** 2


def planet_ephemeris(
    planet: str,
    pos_and_vel: SpacecraftPositionAndVelocity,
    start_time: astropy.time.Time,
    time_span_s: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Position of `planet` with respect to the spacecraft, in km and in
    Ecliptic coordinates, on a grid of times (seconds since `start_time`)
    covering `time_span_s`

    Use :func:`planet_position` to interpolate it.
    """
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


def planet_position(
    grid_s: npt.NDArray, relative_pos_km: npt.NDArray, times_s: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Unit vectors pointing to the planet and its distance in km at times
    `times_s`, interpolating the result of :func:`planet_ephemeris`"""
    pos = np.stack(
        [np.interp(times_s, grid_s, relative_pos_km[:, i]) for i in range(3)], axis=-1
    )
    distance = np.linalg.norm(pos, axis=1)
    return pos / distance[:, None], distance


def _ang2vec(theta: npt.NDArray, phi: npt.NDArray) -> npt.NDArray:
    sin_theta = np.sin(theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=-1
    )


@njit
def _angle_to_planet(pointings, planet_dir, det_idx, sample_idx) -> float:
    """Cosine of the angle between a pointing (θ, φ) and the planet"""
    theta = pointings[det_idx, sample_idx, 0]
    phi = pointings[det_idx, sample_idx, 1]
    sin_theta = np.sin(theta)
    return (
        sin_theta * np.cos(phi) * planet_dir[sample_idx, 0]
        + sin_theta * np.sin(phi) * planet_dir[sample_idx, 1]
        + np.cos(theta) * planet_dir[sample_idx, 2]
    )


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
    planet_vec, _ = planet_position(grid_s, relative_pos_km, times_s)
    distance = np.arccos(np.clip(np.sum(bore_vec * planet_vec, axis=1), -1, 1))
    close = distance < max_distance_rad + _max_detector_offset_rad(obs) + max_motion

    # Point k covers samples [k * step, (k + 1) * step); join consecutive points
    edges = np.flatnonzero(np.diff(np.concatenate([[0], close.astype(int), [0]])))
    return [
        (start * step, min(stop * step, obs.n_samples))
        for start, stop in zip(edges[::2], edges[1::2])
    ]


def check_observations_for_planets(obs_list: list[Observation]) -> None:
    """Raise ValueError if the observations lack what planets need"""
    for obs in obs_list:
        if not isinstance(obs.start_time, astropy.time.Time):
            raise ValueError("Planets need times in astropy.time.Time")
        if np.any(obs.fwhm_arcmin <= 0):
            raise ValueError("Planets need fwhm_arcmin > 0 for all detectors")


def iter_planet_segments(
    obs: Observation,
    pos_and_vel: SpacecraftPositionAndVelocity,
    planet: str,
    max_distance_rad: float,
) -> Iterator[tuple[int, int, npt.NDArray, npt.NDArray, npt.NDArray]]:
    """Segments of `obs` in which some detector can be closer than
    `max_distance_rad` to `planet`

    Yield ``(start, stop, pointings, planet_dir, distance_km)`` for the
    samples ``[start, stop)``: the Ecliptic pointings of all the detectors
    ``(n_detectors, n, 3)``, the unit vectors pointing to the planet
    ``(n, 3)`` and its distance from the spacecraft ``(n,)``. The pointings
    are computed only in these segments, found sampling the boresight every
    second.
    """
    if obs.n_samples == 0 or obs.n_detectors == 0:
        return

    grid_s, relative_pos_km = planet_ephemeris(
        planet, pos_and_vel, obs.start_time, obs.n_samples / obs.sampling_rate_hz
    )
    for start, stop in _segments_close_to_planet(
        obs, grid_s, relative_pos_km, max_distance_rad
    ):
        pointings = np.empty((obs.n_detectors, stop - start, 3))
        for det_idx in range(obs.n_detectors):
            obs.pointing_provider.get_pointings(
                detector_quat=obs.quat[det_idx],
                start_time=obs.start_time
                + astropy.time.TimeDelta(start / obs.sampling_rate_hz, format="sec"),
                start_time_global=obs.start_time_global,
                sampling_rate_hz=obs.sampling_rate_hz,
                nsamples=stop - start,
                pointing_buffer=pointings[det_idx],
                compute_hwp_angle=False,
            )
        times_s = np.arange(start, stop) / obs.sampling_rate_hz
        planet_dir, distance_km = planet_position(grid_s, relative_pos_km, times_s)
        yield start, stop, pointings, planet_dir, distance_km


# Gaussian beams are truncated at this radius, where they are ~1e-11
_BEAM_TRUNCATION_FWHM = 3.0


def planet_beam_parameters(
    obs: Observation, planet: str
) -> tuple[npt.NDArray, npt.NDArray]:
    """Return σ of the Gaussian beams and the signal at their centre per
    steradian of `planet` (in K_CMB/sr) for each detector of `obs`"""
    fwhm_rad = np.deg2rad(np.asarray(obs.fwhm_arcmin) / 60.0)
    sigma_rad = fwhm_rad / np.sqrt(8 * np.log(2))
    brightness = np.array(
        [
            planet_brightness_k_cmb(planet, center, width)
            for center, width in zip(obs.bandcenter_ghz, obs.bandwidth_ghz)
        ]
    )
    return sigma_rad, brightness / (2 * np.pi * sigma_rad**2)


@njit
def add_planet_signal(
    tod,
    pointings,
    planet_dir,
    planet_solid_angle,
    peak_per_sr,
    sigma_rad,
    max_angle_rad,
) -> None:
    """Add the signal of a planet seen through Gaussian beams to `tod`

    `tod` (``(n_detectors, n_samples)``) is modified in place; `pointings`
    (``(n_detectors, n_samples, ≥2)``) contains θ and φ, `planet_dir`
    (``(n_samples, 3)``) the unit vectors pointing to the planet,
    `planet_solid_angle` (``(n_samples,)``) its solid angle, `peak_per_sr`
    (``(n_detectors,)``) the signal at the centre of the beam per steradian
    of planet, and `sigma_rad` (``(n_detectors,)``) the width of the beams,
    truncated at `max_angle_rad` (``(n_detectors,)``).
    """
    for det_idx in range(tod.shape[0]):
        cos_max = np.cos(max_angle_rad[det_idx])
        inv_two_sigma2 = 0.5 / sigma_rad[det_idx] ** 2
        for sample_idx in range(tod.shape[1]):
            cos_angle = _angle_to_planet(pointings, planet_dir, det_idx, sample_idx)
            if cos_angle > cos_max:
                angle = np.arccos(min(cos_angle, 1.0))
                tod[det_idx, sample_idx] += (
                    peak_per_sr[det_idx]
                    * planet_solid_angle[sample_idx]
                    * np.exp(-(angle**2) * inv_two_sigma2)
                )


def add_planets_to_observations(
    observations: Observation | list[Observation],
    pos_and_vel: SpacecraftPositionAndVelocity,
    planets: str | list[str] | tuple[str, ...] = PLANETS,
    component: str = "tod",
) -> None:
    """Add the signal of the planets to the TODs

    The planets are unpolarized point sources with the temperatures in
    :data:`PLANET_TEMPERATURES_K`, integrated over a top-hat band
    (``bandcenter_ghz``, ``bandwidth_ghz``) and seen through circular
    Gaussian beams (``fwhm_arcmin``), truncated at 3 FWHM. Their direction and
    solid angle are computed from the position of the spacecraft in
    `pos_and_vel`. As for :func:`.flag_planets_in_observations`, the pointings
    are computed only close to the planets, and the times of the
    observations must be ``astropy.time.Time`` objects.

    ponytail: Gaussian beams; the radial profile of the beam would give
    realistic sidelobes.
    """
    obs_list = [observations] if isinstance(observations, Observation) else observations
    planets = [planets] if isinstance(planets, str) else list(planets)
    check_observations_for_planets(obs_list)

    for obs in obs_list:
        tod = getattr(obs, component)
        fwhm_rad = np.deg2rad(np.asarray(obs.fwhm_arcmin) / 60.0)
        max_angle_rad = _BEAM_TRUNCATION_FWHM * fwhm_rad
        for planet in planets:
            sigma_rad, peak_per_sr = planet_beam_parameters(obs, planet)
            for start, stop, pointings, planet_dir, distance_km in iter_planet_segments(
                obs, pos_and_vel, planet, max_angle_rad.max()
            ):
                add_planet_signal(
                    tod[:, start:stop],
                    pointings,
                    planet_dir,
                    planet_solid_angle_sr(planet, distance_km),
                    peak_per_sr,
                    sigma_rad,
                    max_angle_rad,
                )
