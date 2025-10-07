import numpy as np
from numba import njit, prange

from numbers import Number

from .observations import Observation
from .hwp import HWP
from .pointings_in_obs import _get_hwp_angle


# We calculate the additive signal coming from hwp harmonics.
# here we calculate 2f directly.
@njit
def compute_2f_for_one_sample(angle_rad, amplitude_k):
    return amplitude_k * np.cos(2 * angle_rad)


@njit(parallel=True)
def add_2f_for_one_detector(tod_det, angle_det_rad, amplitude_k):
    for i in prange(len(tod_det)):
        tod_det[i] += compute_2f_for_one_sample(
            angle_rad=angle_det_rad[i], amplitude_k=amplitude_k
        )


def add_2f(
    tod,
    hwp_angle,
    pol_angle_rad: float,
    amplitude_2f_k: float,
):
    """Add the HWP differential emission to some time-ordered data

    This functions modifies the values in `tod` by adding the contribution of the HWP
    synchronous signal coming from differential emission. The `amplitude_2f_k` argument must be
    a N_dets array containing the amplitude of the HWPSS."""

    assert len(tod.shape) == 2
    num_of_dets = tod.shape[0]

    if isinstance(amplitude_2f_k, Number):
        amplitude_2f_k = np.array([amplitude_2f_k] * num_of_dets)

    assert len(amplitude_2f_k) == num_of_dets

    for detector_idx in range(tod.shape[0]):
        add_2f_for_one_detector(
            tod_det=tod[detector_idx],
            angle_det_rad=hwp_angle - pol_angle_rad[detector_idx],
            amplitude_k=amplitude_2f_k[detector_idx],
        )


def add_2f_to_observations(
    observations: Observation | list[Observation],
    hwp: HWP | None = None,
    component: str = "tod",
    amplitude_2f_k: float | None = None,
    pointings_dtype=np.float64,
):
    """Add the HWP differential emission to some time-ordered data

    This is a wrapper around the :func:`.add_2f` function that applies to the TOD
    stored in `observations`, which can either be one :class:`.Observation` instance
    or a list of observations.

    By default, the TOD is added to ``Observation.tod``. If you want to add it to some
    other field of the :class:`.Observation` class, use `component`::

    for cur_obs in sim.observations:
        # Allocate a new TOD for the 2f alone
        cur_obs.2f_tod = np.zeros_like(cur_obs.tod)

        # Ask `add_2f_to_observations` to store the 2f
        # in `observations.2f_tod`
        add_2f_to_observations(sim.observations, component="2f_tod")
    """
    if isinstance(observations, Observation):
        obs_list = [observations]
    else:
        obs_list = observations

    # iterate through each observation
    for cur_obs in obs_list:
        if amplitude_2f_k is None:
            amplitude_2f_k = cur_obs.amplitude_2f_k

        # Determine the HWP angle to use:
        # - If an external HWP object is provided, compute the angle from it
        # - If not, compute or retrieve the HWP angle from the observation, depending on availability
        hwp_angle = _get_hwp_angle(obs=cur_obs, hwp=hwp, pointing_dtype=pointings_dtype)

        add_2f(
            tod=getattr(cur_obs, component),
            hwp_angle=hwp_angle,
            pol_angle_rad=cur_obs.pol_angle_rad,
            amplitude_2f_k=amplitude_2f_k,
        )
