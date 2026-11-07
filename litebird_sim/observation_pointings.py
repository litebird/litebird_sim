"""The pointings of the detectors of an observation, and its HWP angle

Effects and map-makers ask an :class:`.ObservationPointings` for the pointings
(θ, φ, ψ) of the detectors of one observation and for its HWP angle. It hides
where they come from:

- an array precomputed by :meth:`.Observation.precompute_pointings`
  (``obs.pointing_matrix``);
- the pointing provider, which computes them on the fly from the quaternions
  (:meth:`.Observation.get_pointings`);
- an array passed by the caller (the ``pointings`` argument of the public
  functions), in Ecliptic coordinates.

It never writes into the observation, and keeps only the HWP angle in memory:
the pointings are read or computed again at each request.
"""

import astropy.time
import numpy as np
import numpy.typing as npt

from litebird_sim.coordinates import CoordinateSystem, rotate_coordinates_e2g
from litebird_sim.hwp import HWP
from litebird_sim.observations import Observation, _center_pointings

#: The forms accepted to select detectors, as in :meth:`.Observation.get_pointings`
DetectorIndex = int | list[int] | str

_NOT_COMPUTED = object()


class ObservationPointings:
    """The pointings of the detectors of `observation` and its HWP angle.

    :param observation: The :class:`.Observation`
    :param pointings: An array ``(n_detectors, n_samples, 3)`` with the
        pointings in Ecliptic coordinates, passed by the caller. If None, the
        pointings precomputed by :meth:`.Observation.precompute_pointings`
        are used if present, otherwise they are computed on the fly.
    :param hwp: An :class:`.HWP` that overrides the one of the observation
    :param output_coordinate_system: The default coordinate system of the
        pointings returned by :meth:`.detector`
    :param nside_centering: If given, (θ, φ) are moved to the center of their
        Healpix pixel at this resolution, after the rotation; ψ is unchanged
    :param pointings_dtype: The dtype of the pointings and of the HWP angle
        computed on the fly
    :param nthreads: Threads for the centering; see :func:`.resolve_nthreads`
    """

    def __init__(
        self,
        observation: Observation,
        pointings: npt.NDArray | None = None,
        hwp: HWP | None = None,
        output_coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic,
        nside_centering: int | None = None,
        pointings_dtype=np.float64,
        nthreads: int | None = None,
    ):
        self.observation = observation
        self.output_coordinate_system = output_coordinate_system
        self.nside_centering = nside_centering
        self.pointings_dtype = pointings_dtype
        self.nthreads = nthreads
        self._hwp = hwp
        self._hwp_angle = _NOT_COMPUTED

        if pointings is not None:
            expected = (observation.n_detectors, observation.n_samples_process, 3)
            assert isinstance(pointings, np.ndarray) and pointings.shape == expected, (
                f"The pointings have shape {np.shape(pointings)}, but the "
                f"observation needs the shape {expected}"
            )
            self._ecliptic_array = pointings
        elif hasattr(observation, "pointing_matrix"):
            self._ecliptic_array = observation.pointing_matrix
        else:
            self._ecliptic_array = None  # Computed on the fly

        if hwp is None and not observation.has_hwp:
            if hasattr(observation, "mueller_hwp") and not observation.no_mueller_hwp():
                raise AssertionError(
                    "Detectors have been initialized with a mueller_hwp, "
                    "but no HWP is either passed or initialized in the pointing."
                )

    @property
    def hwp_angle(self) -> npt.NDArray | None:
        """The HWP angle of each sample, or None if there is no HWP.

        The HWP passed to the constructor has the precedence; otherwise the
        angle precomputed in the observation is used, or it is computed with
        the pointing provider. It is computed only once."""
        if self._hwp_angle is _NOT_COMPUTED:
            self._hwp_angle = self._compute_hwp_angle()
        return self._hwp_angle  # type: ignore[return-value]

    def _compute_hwp_angle(self) -> npt.NDArray | None:
        obs = self.observation
        if self._hwp is not None:
            start_time = obs.start_time_process - obs.start_time_simulation
            if isinstance(start_time, astropy.time.TimeDelta):
                start_time = start_time.to("s").value
            hwp_angle = np.empty(obs.n_samples_process, dtype=self.pointings_dtype)
            self._hwp.get_hwp_angle(hwp_angle, start_time, 1.0 / obs.sampling_rate_hz)
            return hwp_angle

        if not obs.has_hwp:
            return None
        if hasattr(obs, "hwp_angle"):
            return obs.hwp_angle
        return obs.get_hwp_angle(pointings_dtype=self.pointings_dtype)

    def detector(
        self,
        detector_idx: DetectorIndex,
        output_coordinate_system: CoordinateSystem | None = None,
    ) -> npt.NDArray:
        """The pointings (θ, φ, ψ) of some detectors, in radians.

        `detector_idx` is an index (the result has shape ``(n_samples, 3)``),
        a list of indices or ``"all"`` (shape ``(n_detectors, n_samples, 3)``),
        as in :meth:`.Observation.get_pointings`. `output_coordinate_system`
        overrides the one passed to the constructor. The result must not be
        modified: it may be a view of a precomputed array."""
        if output_coordinate_system is None:
            output_coordinate_system = self.output_coordinate_system
        if self._ecliptic_array is None:
            pointings, _ = self.observation.get_pointings(
                detector_idx,
                pointings_dtype=self.pointings_dtype,
                nside_centering=self.nside_centering,
                nthreads=self.nthreads,
                compute_hwp_angle=False,
                output_coordinate_system=output_coordinate_system,
            )
            return pointings

        if isinstance(detector_idx, str):
            assert detector_idx == "all", f"Unknown set of detectors: '{detector_idx}'"
            selected = self._ecliptic_array
        else:
            selected = self._ecliptic_array[detector_idx]

        if output_coordinate_system == CoordinateSystem.Galactic:
            if selected.ndim == 2:
                selected = rotate_coordinates_e2g(selected)
            else:
                selected = np.stack([rotate_coordinates_e2g(x) for x in selected])
        elif self.nside_centering is not None:
            # Do not move the pixels of the caller's array
            selected = selected.copy()

        _center_pointings(selected, self.nside_centering, self.nthreads)
        return selected


def observation_pointings(
    observations: Observation | list[Observation],
    pointings: npt.NDArray | list[npt.NDArray] | None = None,
    hwp: HWP | None = None,
    **kwargs,
) -> list[tuple[Observation, ObservationPointings]]:
    """Pair each observation with its :class:`.ObservationPointings`.

    `observations` is one :class:`.Observation` or a list of them; if given,
    `pointings` must be one array or a list of arrays with the same length.
    The other keyword arguments are those of :class:`.ObservationPointings`.
    All the checks are done here, before any pointing is computed."""
    if isinstance(observations, Observation):
        obs_list = [observations]
        if pointings is not None:
            assert isinstance(pointings, np.ndarray), (
                "For one observation you must pass one array of pointings"
            )
        ptg_list = [pointings]
    else:
        obs_list = list(observations)
        if pointings is None:
            ptg_list = [None] * len(obs_list)
        else:
            assert isinstance(pointings, list), (
                "For a list of observations you must pass a list of pointings"
            )
            assert len(obs_list) == len(pointings), (
                f"The list of observations has {len(obs_list)} elements, but "
                f"the list of pointings has {len(pointings)} elements"
            )
            ptg_list = list(pointings)

    return [
        (cur_obs, ObservationPointings(cur_obs, pointings=cur_ptg, hwp=hwp, **kwargs))
        for cur_obs, cur_ptg in zip(obs_list, ptg_list)
    ]
