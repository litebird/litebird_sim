from collections.abc import Callable, Iterator

import astropy.time
import numpy as np
import numpy.typing as npt
from deprecated import deprecated
from ducc0.healpix import Healpix_Base

from .coordinates import CoordinateSystem, rotate_coordinates_e2g
from .detectors import InstrumentInfo
from .hwp import HWP
from .observations import Observation
from .scanning import RotQuaternion
from .seeding import regenerate_or_check_detector_generators


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

    for cur_obs in obs_list:
        cur_obs.prepare_pointings(
            instrument=instrument, spin2ecliptic_quats=spin2ecliptic_quats, hwp=hwp
        )


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

    start_time = observations.start_time - observations.start_time_global
    if isinstance(start_time, astropy.time.TimeDelta):
        start_time_s = start_time.to("s").value
    else:
        start_time_s = start_time

    hwp.add_hwp_angle(
        pointing_matrix,
        start_time_s,
        1.0 / observations.sampling_rate_hz,
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
                if any(m is not None for m in obs.mueller_hwp):
                    raise AssertionError(
                        "Detectors have been initialized with a mueller_hwp, "
                        "but no HWP is either passed or initialized in the pointing."
                    )
            return None
    else:
        # Compute HWP angle from HWP object
        start_time = obs.start_time - obs.start_time_global
        start_time_s = (
            start_time.to("s").value
            if isinstance(start_time, astropy.time.TimeDelta)
            else start_time
        )

        hwp_angle = np.empty(obs.n_samples, dtype=pointing_dtype)
        hwp.get_hwp_angle(
            hwp_angle,
            start_time_s,
            1.0 / obs.sampling_rate_hz,
        )

        obs.has_hwp = True
        return hwp_angle


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


def _get_pointings_array(
    detector_idx: int,
    pointings: npt.NDArray | Callable,
    hwp_angle: np.ndarray | None,
    output_coordinate_system: CoordinateSystem,
    nside_centering: int | None = None,
    pointings_dtype=np.float64,
    nthreads: int = 0,
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
    nthreads : int, optional
        Number of threads to use for HEALPix operations when centering pointings.
        Default is 0 (use all available threads).

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        A tuple `(pointings, hwp_angle)`, where:
                    - `pointings` is an array of shape (n_samples, N_cols), where
                        N_cols is typically 2 ([θ, φ]) or 3 ([θ, φ, ψ]).
          - `hwp_angle` is either the provided array or the one computed by the callable.
    """
    if isinstance(pointings, np.ndarray):
        curr_pointings_det = pointings[detector_idx, :, :]
        computed_hwp_angle = None
    else:
        curr_pointings_det, computed_hwp_angle = pointings(
            detector_idx, pointings_dtype=pointings_dtype
        )

    if output_coordinate_system == CoordinateSystem.Galactic:
        curr_pointings_det = rotate_coordinates_e2g(curr_pointings_det)

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


def _get_pointings_and_pol_angles_det(
    obs: Observation,
    det_idx: int,
    hwp: HWP | None = None,
    pointings: np.ndarray | list[np.ndarray] | None = None,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    pointing_dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the pointings (θ and φ) and polarization angle of a detector

    Parameters
    ----------
    obs : Observation
        An instance of :class:`.Observation` class
    det_idx : int
        Detector index, local to an :class:`.Observation`
    hwp : HWP | None, optional
        An instance of the :class:`.HWP` class (optional)
    pointings : np.ndarray | list[np.ndarray] | None, optional
        An array of pointings or a list containing the array of pointings,
        by default `None`
    output_coordinate_system : CoordinateSystem, optional
        Coordinate system of the output pointings, by default
        `CoordinateSystem.Galactic`
    pointing_dtype : dtype, optional
        The dtype for the computed hwp angle, by default `np.float64`

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the pointings and polarization angle of the detector
    """

    hwp_angle = _get_hwp_angle(
        obs=obs,
        hwp=hwp,
        pointing_dtype=pointing_dtype,
    )

    __, pointings = _normalize_observations_and_pointings(  # type: ignore[assignment]
        observations=obs, pointings=pointings
    )

    assert pointings is not None, "Pointings should not be None after normalization."
    pointings_det, hwp_angle = _get_pointings_array(
        detector_idx=det_idx,
        pointings=pointings[0],
        hwp_angle=hwp_angle,
        output_coordinate_system=output_coordinate_system,
        pointings_dtype=pointing_dtype,
    )

    pol_angle = _get_pol_angle(
        curr_pointings_det=pointings_det,
        hwp_angle=hwp_angle,
        pol_angle_detectors=obs.pol_angle_rad[det_idx],
    )

    return pointings_det, pol_angle


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
        when *requires_rng* is ``True``.

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


def for_each_observation_with_pointings(
    observations: Observation | list[Observation],
    pointings: np.ndarray | list[np.ndarray] | None,
    component: str = "tod",
) -> Iterator[tuple[Observation, np.ndarray, np.ndarray]]:
    """Iterate over observations paired with their pointing matrices.

    The pointing-aware counterpart of :func:`for_each_observation`, for effects
    that need a pointing matrix per observation (e.g. the CMB dipole).  It uses
    :func:`_normalize_observations_and_pointings`, so a single observation may
    be paired with a single pointing array, or a list with a list.  When
    *pointings* is ``None`` the pointing matrix is taken from each observation.

    Yields
    ------
    tuple
        ``(cur_obs, tod, cur_ptg)`` for each observation, where ``cur_ptg`` is
        either a pointing array or the observation's ``get_pointings`` callable
        (matching the behaviour of the underlying normalizer).
    """
    obs_list, ptg_list = _normalize_observations_and_pointings(observations, pointings)
    for cur_obs, cur_ptg in zip(obs_list, ptg_list):
        yield cur_obs, _get_tod_component(cur_obs, component), cur_ptg
