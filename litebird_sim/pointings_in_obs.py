# -*- encoding: utf-8 -*-

from typing import List, Optional, Union, Tuple, Callable

import numpy as np
import numpy.typing as npt
import astropy.time

from deprecated import deprecated

from ducc0.healpix import Healpix_Base

from .detectors import InstrumentInfo
from .hwp import HWP
from .observations import Observation
from .scanning import RotQuaternion

from .coordinates import CoordinateSystem, rotate_coordinates_e2g


def prepare_pointings(
    observations: Union[Observation, List[Observation]],
    instrument: InstrumentInfo,
    spin2ecliptic_quats: RotQuaternion,
    hwp: Optional[HWP] = None,
) -> None:
    """Store the quaternions needed to compute pointings into a list of :class:`.Observation` objects

    This function computes the quaternions that convert the boresight direction
    of `instrument` into the Ecliptic reference frame. The `spin2ecliptic_quats`
    object must be an instance of the :class:`.RotQuaternion` class and can
    be created using the method :meth:`.ScanningStrategy.generate_spin2ecl_quaternions`.
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
    observations: Union[Observation, List[Observation]],
    pointings_dtype=np.float64,
) -> None:
    """Precompute all the pointings for a set of observations

    Compute the full pointing matrix and the HWP angle for each :class:`.Observation`
    object in `obs_list` and store them in the fields ``pointing_matrix`` and ``hwp_angle``.
    The datatype for the pointings is specified by `pointings_dtype`.
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
    hwp: Union[HWP, None] = None,
    pointing_dtype=np.float64,
) -> Union[np.ndarray, None]:
    """Obtains the hwp angle for an observation

    Parameters
    ----------
    obs : Observation
        An instance of the :class:`.Observation` class
    hwp : Union[HWP, None], optional
        An instance of the :class:`.HWP` class (optional)
    pointing_dtype : dtype, optional
        The dtype for the computed hwp angle, by default `np.float64`

    Returns
    -------
    Union[np.ndarray, None]
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
    hwp_angle: Union[np.ndarray, None],
    pol_angle_detectors: float,
) -> np.ndarray:
    """Computes the polarization angle of the detector

    Parameters
    ----------
    curr_pointings_det : np.ndarray
        Pointing information of the detector, here we take just the orientation
    hwp_angle : Union[np.ndarray, None]
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
    pointings: Union[npt.ArrayLike, Callable],
    hwp_angle: Union[np.ndarray, None],
    output_coordinate_system: CoordinateSystem,
    pointings_dtype=np.float64,
) -> Tuple[np.ndarray, Union[np.ndarray], None]:
    """
    Compute the pointings (θ, φ) and HWP angle for a given detector.

    Parameters
    ----------
    detector_idx : int
        Index of the detector, local to an :class:`Observation`.
    pointings : Union[npt.ArrayLike, Callable]
        Pointing information, either a precomputed array or a callable returning
        (pointings, hwp_angle) for the specified detector.
    hwp_angle : Optional[np.ndarray]
        Array of HWP angles. If None, the angle is assumed to be provided by the `pointings` callable.
    output_coordinate_system : CoordinateSystem
        Desired coordinate system for the output pointings.
    pointings_dtype : np.dtype, optional
        Data type for computed pointings and angles. Default is `np.float64`.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        A tuple `(pointings, hwp_angle)`, where:
          - `pointings` is an array of shape (n_samples, 2) with [θ, φ].
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

    return curr_pointings_det, hwp_angle if isinstance(
        hwp_angle, np.ndarray
    ) else computed_hwp_angle


def _get_centered_pointings(
    input_pointings: npt.ArrayLike,
    nside_centering: int,
) -> np.ndarray:
    """Returns a copy of the input pointings aligned to the center of the HEALPix
    pixel they belong to.

    Parameters
    ----------
    input_pointings : npt.ArrayLike
        Pointing information of the detector
    nside_centering : int
        HEALPix NSIDE parameter used to determine the pixel centers.

    Returns
    -------
    np.ndarray
        An array with the same dimensions of input_pointings alligned with the center of the
        belonging healpix pixel
    """
    hpx = Healpix_Base(nside_centering, "RING")
    output_pointings = np.empty_like(input_pointings)

    # Apply centering on the first two columns (θ, φ)
    output_pointings[:, 0:2] = hpx.pix2ang(hpx.ang2pix(input_pointings[:, 0:2]))

    # Copy any additional columns (e.g., polarization angles) without change
    if input_pointings.shape[1] > 2:
        output_pointings[:, 2:] = input_pointings[:, 2:]

    return output_pointings


def _normalize_observations_and_pointings(
    observations: Union[Observation, List[Observation]],
    pointings: Union[np.ndarray, List[np.ndarray], None],
) -> Tuple[List[Observation], List[npt.NDArray]]:
    """This function builds the tuple (`obs_list`, `ptg_list`) and returns it.

    - `obs_list` contains a list of the observations to be used by current MPI
      process. This is *always* a list, even if just one observation is
      provided in the input.
    - `ptg_list` contains a list of pointing matrices, either as numpy arrays
      or callable functions, one per each observation, each belonging to the
      current MPI process.

    Parameters
    ----------
    observations : Union[Observation, List[Observation]]
        An observation or a list of observations
    pointings : Union[np.ndarray, List[np.ndarray], None]
        External pointing information, if not already included in the observation

    Returns
    -------
    Tuple[List[Observation], List[npt.NDArray]]
        The tuple of the list of observations and list of pointings
    """

    if pointings is None:
        if isinstance(observations, Observation):
            obs_list = [observations]
            if hasattr(observations, "pointing_matrix"):
                ptg_list = [observations.pointing_matrix]
            else:
                ptg_list = [observations.get_pointings]
        else:
            obs_list = observations
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
            obs_list = observations
            ptg_list = pointings

    return obs_list, ptg_list


def _get_pointings_and_pol_angles_det(
    obs: Observation,
    det_idx: int,
    hwp: Optional[HWP] = None,
    pointings: Union[np.ndarray, List[np.ndarray], None] = None,
    output_coordinate_system: CoordinateSystem = CoordinateSystem.Galactic,
    pointing_dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the pointings (θ and φ) and polarization angle of a detector

    Parameters
    ----------
    obs : Observation
        An instance of :class:`.Observation` class
    det_idx : int
        Detector index, local to an :class:`.Observation`
    hwp : Union[HWP, None], optional
        An instance of the :class:`.HWP` class (optional)
    pointings : Union[np.ndarray, List[np.ndarray], None], optional
        An array of pointings or a list containing the array of pointings,
        by default `None`
    output_coordinate_system : CoordinateSystem, optional
        Coordinate system of the output pointings, by default
        `CoordinateSystem.Galactic`
    pointing_dtype : dtype, optional
        The dtype for the computed hwp angle, by default `np.float64`

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the pointings and polarization angle of the detector
    """

    hwp_angle = _get_hwp_angle(
        obs=obs,
        hwp=hwp,
        pointing_dtype=pointing_dtype,
    )

    __, pointings = _normalize_observations_and_pointings(
        observations=obs, pointings=pointings
    )

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
