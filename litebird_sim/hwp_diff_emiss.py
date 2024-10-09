# -*- encoding: utf-8 -*-


from numba import njit, prange
import numpy as np

from typing import Union, List
from numbers import Number

from .observations import Observation
from .hwp import HWP


"""def convert_pW_to_K(power_pW, NET, NEP):
    temperature_K = power_pW * NET*1e-6/NEP*1e-18/sqrt(2) 
    return(temperature_K)"""  # we could use this if we prefer having IMo quantities in pW


# We calculate the additive signal coming from hwp harmonics.
# here we calculate 2f directly.
@njit
def compute_2f_for_one_sample(angle_rad, amplitude_k, monopole_k):
    return amplitude_k * np.cos(angle_rad) + monopole_k


@njit(parallel=True)
def add_2f_for_one_detector(tod_det, angle_det_rad, amplitude_k, monopole_k):
    for i in prange(len(tod_det)):
        tod_det[i] += compute_2f_for_one_sample(
            angle_rad=angle_det_rad[i], amplitude_k=amplitude_k, monopole_k=monopole_k
        )


def add_2f(
    tod,
    hwp: HWP,
    start_time_s,
    delta_time_s,
    amplitude_k: float,
    monopole_k: float,
):
    """Add the HWP differential emission to some time-ordered data

    This functions modifies the values in `tod` by adding the contribution of the HWP
    synchronous signal coming from differential emission. The `amplitude_k` argument must be
    a N_dets array containing the amplitude of the HWPSS. The `monopole_k` argument must have
    the same size and contain the value of the nominal optical power for the considered frequency channel."""

    assert len(tod.shape) == 2
    num_of_dets = tod.shape[0]

    if isinstance(amplitude_k, Number):
        amplitude_k = np.array([amplitude_k] * num_of_dets)

    if isinstance(monopole_k, Number):
        monopole_k = np.array([monopole_k] * num_of_dets)

    assert len(amplitude_k) == num_of_dets
    assert len(monopole_k) == num_of_dets

    angle_rad = np.empty(tod.shape[1])

    hwp.get_hwp_angle(
        output_buffer=angle_rad, start_time_s=start_time_s, delta_time_s=delta_time_s
    )  # fills angle_rad with 2*hwp angle

    for detector_idx in range(tod.shape[0]):
        add_2f_for_one_detector(
            tod_det=tod[detector_idx],
            angle_det_rad=angle_rad,
            amplitude_k=amplitude_k[detector_idx],
            monopole_k=monopole_k[detector_idx],
        )


def add_2f_to_observations(
    observations: Union[Observation, List[Observation]],
    hwp: HWP,
    component: str = "tod",
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
    for i, cur_obs in enumerate(obs_list):
        add_2f(
            tod=getattr(cur_obs, component),
            hwp=hwp,
            start_time_s=(cur_obs.start_time - cur_obs.start_time_global).to("s").value,
            delta_time_s=1 / cur_obs.sampling_rate_hz,
            amplitude_k=cur_obs.amplitude_k,
            monopole_k=cur_obs.monopole_k,
        )
