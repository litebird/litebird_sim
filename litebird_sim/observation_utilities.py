from collections.abc import Callable, Iterator

import astropy.time
import numpy as np
import numpy.typing as npt
from deprecated import deprecated
from ducc0.healpix import Healpix_Base

from .coordinates import CoordinateSystem, rotate_coordinates_e2g
from .detectors import InstrumentInfo
from .hwp import HWP
from .observations import Observation, _center_pointings
from .scanning import RotQuaternion
from .seeding import regenerate_or_check_detector_generators
from .utilities import resolve_nthreads


def prepare_pointings(
    observations: Observation | list[Observation],
    instrument: InstrumentInfo,
    spin2ecliptic_quats: RotQuaternion,
    hwp: HWP | None = None,
) -> None:
    """Initialize pointing and HWP information for one or more observations.

    This function computes the boresight-to-Ecliptic quaternions for the given instrument
    and applies them to one or more :class:`.Observation` objects. It initializes the
    :class:`.PointingProvider` for each observation, enabling future computation of
    detector pointing angles and HWP angles.

    The quaternion applied is the product of the global spin-to-Ecliptic rotation and the
    instrument's internal boresight-to-spin rotation. Optionally, a Half-Wave Plate (HWP)
    model can be passed and automatically propagated to the detectors in each observation.

    Args:
        observations (Observation or list[Observation]):
            A single observation or a list of :class:`.Observation` objects to configure.

        instrument (InstrumentInfo):
            The instrument definition, including the internal rotation from boresight to spin axis.

        spin2ecliptic_quats (RotQuaternion):
            Time-dependent quaternion representing the rotation from the instrument spin frame
            to the Ecliptic reference frame. Typically generated using
            :meth:`.ScanningStrategy.generate_spin2ecl_quaternions`.

        hwp (HWP | None):
            An optional Half-Wave Plate model to attach to the observations.

    Returns:
        None

    Notes:
        This function is typically used in preparation for generating time-domain pointings.
        Once this setup is complete, calling `obs.get_pointings()` will produce properly
        oriented (θ, φ, ψ) coordinates, and if applicable, HWP angles.
    """

    if isinstance(observations, Observation):
        obs_list = [observations]
    else:
        obs_list = observations

    # The boresight quaternions span the whole mission and are the same for
    # every observation: compute them once and share them, instead of keeping
    # one copy per observation (see issue #489)
    bore2ecliptic_quats = spin2ecliptic_quats * instrument.bore2spin_quat
    for cur_obs in obs_list:
        cur_obs._set_pointing_provider(bore2ecliptic_quats=bore2ecliptic_quats, hwp=hwp)


def precompute_pointings(
    observations: Observation | list[Observation],
    pointings_dtype=np.float64,
) -> None:
    """Precompute pointing angles and HWP angles for a set of observations.

    This function triggers the computation of the full time-domain pointing matrix
    and, if applicable, the HWP angle vector for one or more :class:`.Observation`
    instances. The results are stored internally in each observation's
    ``pointing_matrix`` and ``hwp_angle`` fields.

    This is typically used to cache pointing-related data in advance, avoiding the
    need for on-the-fly computation during scanning or map-making operations.

    Args:
        observations (Observation or list[Observation]):
            A single observation or a list of observations for which pointings should be precomputed.

        pointings_dtype (data-type, optional):
            Data type to use when allocating the pointing and HWP arrays.
            Defaults to `np.float64`.

    Returns:
        None

    Raises:
        AssertionError:
            If any observation does not have a pointing provider initialized.
            Make sure to call :func:`prepare_pointings()` beforehand.

    Notes:
        - This function must be called after pointing setup (i.e., after `prepare_pointings()`).
        - Output arrays are stored in `Observation.pointing_matrix` and `Observation.hwp_angle`.
    """

    if isinstance(observations, Observation):
        obs_list = [observations]
    else:
        obs_list = observations

    for cur_obs in obs_list:
        cur_obs.precompute_pointings(pointings_dtype=pointings_dtype)


@deprecated(
    version="0.15.0",
    reason="This function adds the HWP angle to the orientation, but this is logically wrong "
    "now that LBS keeps track of ψ the orientation of the telescope, the polarization angle θ, "
    "and the HWP angle in separate places.",
)
def apply_hwp_to_obs(observations, hwp: HWP, pointing_matrix):
    """Modify a pointing matrix to consider the effect of a HWP

    This function modifies the variable `pointing_matrix` (a D×N×3 matrix,
    with D the number of detectors and N the number of samples) so that the
    orientation angle considers the behavior of the half-wave plate in
    `hwp`.
    """

    start_time = observations.start_time_process - observations.start_time_simulation
    if isinstance(start_time, astropy.time.TimeDelta):
        time_from_sim_start_s = start_time.to("s").value
    else:
        time_from_sim_start_s = start_time

    hwp.add_hwp_angle(
        pointing_matrix,
        time_from_sim_start_s,
        1.0 / observations.sampling_rate_hz,
    )


def _get_pol_angle(
    curr_pointings_det: np.ndarray,
    hwp_angle: np.ndarray | None,
    pol_angle_detectors: float,
) -> np.ndarray:
    """Computes the polarization angle of the detector

    Parameters
    ----------
    curr_pointings_det : np.ndarray
        Pointing information of the detector, here we take just the orientation
    hwp_angle : np.ndarray | None
        An array containing the HWP angle or `None`
    pol_angle_detectors : float
        Polarization angle of the detector

    Returns
    -------
    np.ndarray
        An array containing the polarization angle of the detector
    """
    if hwp_angle is None:
        pol_angle = pol_angle_detectors + curr_pointings_det[:, 2]
    else:
        pol_angle = 2 * hwp_angle - pol_angle_detectors + curr_pointings_det[:, 2]

    return pol_angle


# ---------------------------------------------------------------------------
# Shared driver for the systematic-effect modules
#
# Every module that injects an effect into the time-ordered data (noise, gain
# drift, non-linearity, HWP differential emission, the CMB dipole, …) used to
# re-implement the same preamble before getting to its own physics: coerce the
# ``observations`` argument into a list, optionally build one RNG per detector
# for the current MPI rank, then loop pulling ``getattr(obs, component)`` to
# find the TOD array to modify.  That preamble is identical across the effects,
# so it lives here once.  An effect module now writes only the part that
# differs — which detector parameters it reads and which kernel it calls.
# ---------------------------------------------------------------------------


def normalize_observations(
    observations: Observation | list[Observation],
) -> list[Observation]:
    """Coerce *observations* into a list of :class:`.Observation`.

    A single :class:`.Observation` is wrapped in a one-element list; a list is
    copied (so callers may mutate the result without touching their input).
    Anything else raises :class:`TypeError`.
    """
    if isinstance(observations, Observation):
        return [observations]
    if isinstance(observations, list):
        return list(observations)
    raise TypeError(
        "The parameter `observations` must be an `Observation` or a list of "
        "`Observation`."
    )


def _get_tod_component(obs: Observation, component: str) -> np.ndarray:
    """Fetch the named TOD component from *obs*, failing fast on a bad name.

    Effect modules pass ``component`` as a plain string.  A typo would otherwise
    reach ``getattr`` and blow up one iteration into the loop, after RNG streams
    are already built, with a bare ``AttributeError``.  Check up front instead
    and raise with the observation's registered TOD names (``obs.tod_list``) as
    a hint.  Any existing attribute is accepted, not only registered TODs, so
    callers that stash a scratch component on the observation keep working.
    """
    if not hasattr(obs, component):
        registered = [td.name for td in obs.tod_list]
        raise ValueError(
            f"'{component}' is not an attribute of this observation; "
            f"registered TOD components are {registered}."
        )
    return getattr(obs, component)


def for_each_observation(
    observations: Observation | list[Observation],
    component: str = "tod",
    *,
    user_seed: int | None = None,
    dets_random: list[np.random.Generator] | None = None,
    requires_rng: bool = False,
    comm=None,
) -> Iterator[tuple[Observation, np.ndarray, list[np.random.Generator] | None]]:
    """Iterate over observations, yielding ``(obs, tod, dets_random)``.

    This is the common driver for the systematic-effect modules.  It performs
    the bookkeeping every effect shares:

    - normalizes *observations* into a list
      (see :func:`normalize_observations`);
    - if *requires_rng* is ``True``, resolves one RNG per detector for the
      current MPI rank via
      :func:`.regenerate_or_check_detector_generators` and yields the same list
      on every iteration;
    - yields the ``component`` TOD array (validated against
      ``obs.tod_list``) so the caller does not repeat the attribute
      indirection.

    Parameters
    ----------
    observations : Observation or list of Observation
        The observation(s) to iterate over.
    component : str, optional
        Name of the TOD attribute to fetch from each observation.  Defaults to
        ``"tod"``.
    user_seed : int, optional
        Master seed used to build the per-detector RNGs when *dets_random* is
        not supplied.  Only consulted when *requires_rng* is ``True``.
    dets_random : list of numpy.random.Generator, optional
        Pre-built per-detector RNGs.  Only consulted when *requires_rng* is
        ``True``.
    requires_rng : bool, optional
        Whether the effect is stochastic.  When ``True`` exactly one of
        *user_seed* or *dets_random* must be provided.  When ``False`` (the
        default) no RNGs are built and ``None`` is yielded in their place.
    comm : optional
        MPI communicator forwarded to
        :func:`.regenerate_or_check_detector_generators`.  Only consulted
        when *requires_rng* is ``True``.  Defaults to the communicator of the
        first observation, so that the generators built from *user_seed* differ
        among MPI processes.

    Yields
    ------
    tuple
        ``(cur_obs, tod, dets_random)`` for each observation.  ``dets_random``
        is the resolved list of generators when *requires_rng* is ``True``,
        otherwise ``None``.  The same ``dets_random`` object is yielded on every
        iteration.
    """
    obs_list = normalize_observations(observations)

    if requires_rng:
        if comm is None:
            comm = obs_list[0].comm
        dets_random = regenerate_or_check_detector_generators(
            observations=obs_list,
            comm=comm,
            user_seed=user_seed,
            dets_random=dets_random,
        )
    else:
        dets_random = None

    for cur_obs in obs_list:
        yield cur_obs, _get_tod_component(cur_obs, component), dets_random


@deprecated(
    version="0.19.0",
    reason="Use observation_pointings, which gives one ObservationPointings "
    "per observation",
)
def for_each_observation_with_pointings(
    observations: Observation | list[Observation],
    pointings: np.ndarray | list[np.ndarray] | None,
    component: str = "tod",
) -> Iterator[tuple[Observation, np.ndarray, np.ndarray | Callable]]:
    """Iterate over observations paired with their pointing matrices.

    The pointing-aware counterpart of :func:`for_each_observation`. A single
    observation may be paired with a single pointing array, or a list with a
    list.  When *pointings* is ``None`` the pointing matrix is taken from each
    observation.

    Yields
    ------
    tuple
        ``(cur_obs, tod, cur_ptg)`` for each observation, where ``cur_ptg`` is
        either a pointing array or the observation's ``get_pointings`` callable
        (matching the behaviour of the underlying normalizer).
    """
    for cur_obs, cur_ptg in observation_pointings(observations, pointings):
        ecliptic = cur_ptg._ecliptic_array
        yield (
            cur_obs,
            _get_tod_component(cur_obs, component),
            ecliptic if ecliptic is not None else cur_obs.get_pointings,
        )


# ----------------------------------------------------------------------------
# The pointings of the detectors of an observation, and its HWP angle
#
# Effects and map-makers ask an :class:`.ObservationPointings` for the pointings
# (θ, φ, ψ) of the detectors of one observation and for its HWP angle. It hides
# where they come from:
#
# - an array precomputed by :meth:`.Observation.precompute_pointings`
#   (``obs.pointing_matrix``);
# - the pointing provider, which computes them on the fly from the quaternions
#   (:meth:`.Observation.get_pointings`);
# - an array passed by the caller (the ``pointings`` argument of the public
#   functions), in Ecliptic coordinates.
#
# It never writes into the observation, and keeps only the HWP angle in memory:
# the pointings are read or computed again at each request.
# ----------------------------------------------------------------------------

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

        return _select_from_ecliptic_array(
            self._ecliptic_array,
            detector_idx,
            output_coordinate_system,
            self.nside_centering,
            self.nthreads,
        )

    def pixel_index(
        self,
        detector_idx: DetectorIndex,
        nside: int,
        output_coordinate_system: CoordinateSystem | None = None,
    ) -> npt.NDArray[np.int32]:
        """The Healpix pixel (RING scheme) seen by some detectors at each sample.

        `detector_idx` is as in :meth:`.detector`; for more detectors the
        result has shape ``(n_detectors, n_samples)``, and they are computed
        one at a time. `output_coordinate_system` is the coordinate system of
        the map; if None, that passed to the constructor is used."""
        return self._for_detectors(
            detector_idx,
            output_coordinate_system,
            lambda idx, ptg: self._pixel_index(ptg, nside),
        )

    def pol_angle(
        self,
        detector_idx: DetectorIndex,
        output_coordinate_system: CoordinateSystem | None = None,
    ) -> npt.NDArray:
        """The polarization angle of some detectors at each sample, in radians.

        It is ``θ_det + ψ`` without HWP and ``2·α_HWP − θ_det + ψ`` with it,
        where θ_det is the polarization angle of the detector and ψ the
        orientation in `output_coordinate_system`. `detector_idx` is as in
        :meth:`.pixel_index`."""
        return self._for_detectors(
            detector_idx, output_coordinate_system, self._pol_angle
        )

    def pixel_index_and_pol_angle(
        self,
        detector_idx: DetectorIndex,
        nside: int,
        output_coordinate_system: CoordinateSystem | None = None,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray]:
        """:meth:`.pixel_index` and :meth:`.pol_angle` of some detectors,
        computing their pointings only once"""
        if not isinstance(detector_idx, (str, list)):
            pointings = self.detector(detector_idx, output_coordinate_system)
            return (
                self._pixel_index(pointings, nside),
                self._pol_angle(detector_idx, pointings),
            )

        indices = self._indices(detector_idx)
        n_samples = self.observation.n_samples_process
        pixidx = np.empty((len(indices), n_samples), dtype=np.int32)
        pol_angle = np.empty((len(indices), n_samples), dtype=self.pointings_dtype)
        for row, idx in enumerate(indices):
            pixidx[row], pol_angle[row] = self.pixel_index_and_pol_angle(
                idx, nside, output_coordinate_system
            )
        return pixidx, pol_angle

    def _indices(self, detector_idx: str | list[int]) -> list[int]:
        if isinstance(detector_idx, str):
            assert detector_idx == "all", f"Unknown set of detectors: '{detector_idx}'"
            return list(range(self.observation.n_detectors))
        return list(detector_idx)

    def _for_detectors(self, detector_idx, output_coordinate_system, function):
        """Apply `function(idx, pointings)` to each detector and stack the results"""
        if not isinstance(detector_idx, (str, list)):
            return function(
                detector_idx, self.detector(detector_idx, output_coordinate_system)
            )
        return np.stack(
            [
                function(idx, self.detector(idx, output_coordinate_system))
                for idx in self._indices(detector_idx)
            ]
        )

    def _pixel_index(self, pointings: npt.NDArray, nside: int) -> npt.NDArray[np.int32]:
        hpx = Healpix_Base(nside, "RING")
        return hpx.ang2pix(
            pointings[:, 0:2], nthreads=resolve_nthreads(self.nthreads)
        ).astype(np.int32)

    def _pol_angle(self, detector_idx: int, pointings: npt.NDArray) -> npt.NDArray:
        return _get_pol_angle(
            curr_pointings_det=pointings,
            hwp_angle=self.hwp_angle,
            pol_angle_detectors=self.observation.pol_angle_rad[detector_idx],
        )


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


def _select_from_ecliptic_array(
    array: npt.NDArray,
    detector_idx: DetectorIndex,
    output_coordinate_system: CoordinateSystem,
    nside_centering: int | None,
    nthreads: int | None,
) -> npt.NDArray:
    """The pointings of some detectors from an array ``(n_detectors,
    n_samples, 3)`` in Ecliptic coordinates, rotated and centered as
    requested. The array is never modified."""
    if isinstance(detector_idx, str):
        assert detector_idx == "all", f"Unknown set of detectors: '{detector_idx}'"
        selected = array
    else:
        selected = array[detector_idx]

    if output_coordinate_system == CoordinateSystem.Galactic:
        if selected.ndim == 2:
            selected = rotate_coordinates_e2g(selected)
        else:
            selected = np.stack([rotate_coordinates_e2g(x) for x in selected])
    elif nside_centering is not None:
        # Do not move the pixels of the caller's array
        selected = selected.copy()

    _center_pointings(selected, nside_centering, nthreads)
    return selected


class _ArrayPointings:
    """The pointings given to the low-level functions (:func:`.scan_map`,
    :func:`.add_convolved_sky`) without an observation, with the interface of
    :class:`.ObservationPointings` (:meth:`detector` and ``hwp_angle``).

    `pointings` is an array ``(n_detectors, n_samples, 3)`` in Ecliptic
    coordinates, or a function like :meth:`.Observation.get_pointings`. The
    HWP angle is `hwp_angle`; if None, it is that returned by the function,
    once a detector has been asked for."""

    def __init__(
        self,
        pointings: npt.NDArray | Callable,
        hwp_angle: npt.NDArray | None = None,
        nside_centering: int | None = None,
        pointings_dtype=np.float64,
        nthreads: int | None = None,
    ):
        self.pointings = pointings
        self.hwp_angle = hwp_angle
        self.nside_centering = nside_centering
        self.pointings_dtype = pointings_dtype
        self.nthreads = nthreads

    def detector(
        self,
        detector_idx: int,
        output_coordinate_system: CoordinateSystem = CoordinateSystem.Ecliptic,
    ) -> npt.NDArray:
        if isinstance(self.pointings, np.ndarray):
            return _select_from_ecliptic_array(
                self.pointings,
                detector_idx,
                output_coordinate_system,
                self.nside_centering,
                self.nthreads,
            )

        pointings, hwp_angle = self.pointings(
            detector_idx,
            pointings_dtype=self.pointings_dtype,
            compute_hwp_angle=self.hwp_angle is None,
            output_coordinate_system=output_coordinate_system,
        )
        if self.hwp_angle is None:
            self.hwp_angle = hwp_angle
        _center_pointings(pointings, self.nside_centering, self.nthreads)
        return pointings


# ----------------------------------------------------------------------------
# Compatibility helpers
#
# litebird_sim no longer uses these functions: they get the pointings and the
# HWP angle in the old way, and are kept unchanged only because BrahMap
# (brahmap.lbsim) imports them. Remove them once BrahMap uses
# ObservationPointings.
# ----------------------------------------------------------------------------


@deprecated(
    version="0.19.0",
    reason="Kept only for BrahMap; use ObservationPointings",
)
def _get_hwp_angle(
    obs: Observation,
    hwp: HWP | None = None,
    pointing_dtype=np.float64,
) -> np.ndarray | None:
    """Obtains the hwp angle for an observation

    Parameters
    ----------
    obs : Observation
        An instance of the :class:`.Observation` class
    hwp : HWP | None, optional
        An instance of the :class:`.HWP` class (optional)
    pointing_dtype : dtype, optional
        The dtype for the computed hwp angle, by default `np.float64`

    Returns
    -------
    np.ndarray | None
        An array containing the HWP angles or `None`
    """
    if hwp is None:
        if obs.has_hwp:
            if hasattr(obs, "hwp_angle"):
                return obs.hwp_angle
            else:
                return obs.get_hwp_angle(pointings_dtype=pointing_dtype)
        else:
            if hasattr(obs, "mueller_hwp"):
                if not obs.no_mueller_hwp():
                    raise AssertionError(
                        "Detectors have been initialized with a mueller_hwp, "
                        "but no HWP is either passed or initialized in the pointing."
                    )
            return None
    else:
        # Compute HWP angle from HWP object
        start_time = obs.start_time_process - obs.start_time_simulation
        time_from_sim_start_s = (
            start_time.to("s").value
            if isinstance(start_time, astropy.time.TimeDelta)
            else start_time
        )

        hwp_angle = np.empty(obs.n_samples_process, dtype=pointing_dtype)
        hwp.get_hwp_angle(
            hwp_angle,
            time_from_sim_start_s,
            1.0 / obs.sampling_rate_hz,
        )

        obs.has_hwp = True
        return hwp_angle


@deprecated(
    version="0.19.0",
    reason="Kept only for BrahMap; use ObservationPointings",
)
def _get_pointings_array(
    detector_idx: int,
    pointings: npt.NDArray | Callable,
    hwp_angle: np.ndarray | None,
    output_coordinate_system: CoordinateSystem,
    nside_centering: int | None = None,
    pointings_dtype=np.float64,
    nthreads: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute the pointings (θ, φ) and HWP angle for a given detector.

    Parameters
    ----------
    detector_idx : int
        Index of the detector, local to an :class:`Observation`.
    pointings : npt.NDArray | Callable
        Pointing information, either a precomputed array or a callable returning
        (pointings, hwp_angle) for the specified detector.
    hwp_angle : np.ndarray | None
        Array of HWP angles. If None, the angle is assumed to be provided by the `pointings` callable.
    output_coordinate_system : CoordinateSystem
        Desired coordinate system for the output pointings.
    nside_centering : int | None, optional
        If provided, the pointings will be aligned to the center of the HEALPix pixel.
    pointings_dtype : np.dtype, optional
        Data type for computed pointings and angles. Default is `np.float64`.
    nthreads : int or None, optional
        Number of threads to use for HEALPix operations when centering pointings.
        If None, resolved via :func:`.resolve_nthreads`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        A tuple `(pointings, hwp_angle)`, where:
                    - `pointings` is an array of shape (n_samples, N_cols), where
                        N_cols is typically 2 ([θ, φ]) or 3 ([θ, φ, ψ]).
          - `hwp_angle` is either the provided array or the one computed by the callable.
    """
    nthreads = resolve_nthreads(nthreads)

    if isinstance(pointings, np.ndarray):
        # Precomputed pointings are in Ecliptic coordinates
        curr_pointings_det = pointings[detector_idx, :, :]
        computed_hwp_angle = None
        if output_coordinate_system == CoordinateSystem.Galactic:
            curr_pointings_det = rotate_coordinates_e2g(curr_pointings_det)
    else:
        # Do not recompute the HWP angle for each detector if the caller has it
        curr_pointings_det, computed_hwp_angle = pointings(
            detector_idx,
            pointings_dtype=pointings_dtype,
            compute_hwp_angle=hwp_angle is None,
            output_coordinate_system=output_coordinate_system,
        )

    if nside_centering is not None:
        hpx = Healpix_Base(nside_centering, "RING")
        output_pointings = np.empty_like(curr_pointings_det)

        # Apply centering on the first two columns (θ, φ)
        output_pointings[:, 0:2] = hpx.pix2ang(
            hpx.ang2pix(curr_pointings_det[:, 0:2], nthreads=nthreads),
            nthreads=nthreads,
        )

        # Copy any additional columns (e.g., polarization angles) without change
        if curr_pointings_det.shape[1] > 2:
            output_pointings[:, 2:] = curr_pointings_det[:, 2:]

        return output_pointings, hwp_angle if isinstance(
            hwp_angle, np.ndarray
        ) else computed_hwp_angle

    return curr_pointings_det, hwp_angle if isinstance(
        hwp_angle, np.ndarray
    ) else computed_hwp_angle


@deprecated(
    version="0.19.0",
    reason="Kept only for BrahMap; use ObservationPointings",
)
def _normalize_observations_and_pointings(
    observations: Observation | list[Observation],
    pointings: np.ndarray | list[np.ndarray] | None,
) -> tuple[list[Observation], list[npt.NDArray | Callable]]:
    """This function builds the tuple (`obs_list`, `ptg_list`) and returns it.

    - `obs_list` contains a list of the observations to be used by current MPI
      process. This is *always* a list, even if just one observation is
      provided in the input.
    - `ptg_list` contains a list of pointing matrices, either as numpy arrays
      or callable functions, one per each observation, each belonging to the
      current MPI process.

    Parameters
    ----------
    observations : Observation | list[Observation]
        An observation or a list of observations
    pointings : np.ndarray | list[np.ndarray] | None
        External pointing information, if not already included in the observation

    Returns
    -------
    tuple[list[Observation], list[npt.NDArray | Callable]]
        The tuple of the list of observations and list of pointings
    """

    obs_list: list[Observation]
    ptg_list: list[npt.NDArray | Callable]

    if pointings is None:
        if isinstance(observations, Observation):
            obs_list = [observations]
            if hasattr(observations, "pointing_matrix"):
                ptg_list = [observations.pointing_matrix]
            else:
                ptg_list = [observations.get_pointings]
        else:
            obs_list = list(observations)
            ptg_list = []
            for ob in observations:
                if hasattr(ob, "pointing_matrix"):
                    ptg_list.append(ob.pointing_matrix)
                else:
                    ptg_list.append(ob.get_pointings)
    else:
        if isinstance(observations, Observation):
            assert isinstance(pointings, np.ndarray), (
                "You must pass a list of observations *and* a list "
                + "of pointing matrices to scan_map_in_observations"
            )
            obs_list = [observations]
            ptg_list = [pointings]
        else:
            assert isinstance(pointings, list), (
                "When you pass a list of observations to scan_map_in_observations, "
                + "you must do the same for `pointings`"
            )
            assert len(observations) == len(pointings), (
                f"The list of observations has {len(observations)} elements, but "
                + f"the list of pointings has {len(pointings)} elements"
            )
            obs_list = list(observations)
            ptg_list = list(pointings)

    return obs_list, ptg_list
