# -*- encoding: utf-8 -*-

import numpy as np
from numba import njit, prange

from typing import Union, List, Optional
from numbers import Number

from .observations import Observation
from .hwp import HWP
from .pointings import get_hwp_angle


"""def convert_pW_to_K(power_pW, NET, NEP):
    temperature_K = power_pW * NET*1e-6/NEP*1e-18/sqrt(2) 
    return(temperature_K)"""  # we could use this if we prefer having IMo quantities in pW


# We calculate the additive signal coming from hwp harmonics.
# here we calculate 2f directly.
@njit
def compute_2f_for_one_sample(angle_rad, amplitude_k, monopole_k):
    return amplitude_k * np.cos(2 * angle_rad) + monopole_k


@njit(parallel=True)
def add_2f_for_one_detector(tod_det, angle_det_rad, amplitude_k, monopole_k):
    for i in prange(len(tod_det)):
        tod_det[i] += compute_2f_for_one_sample(
            angle_rad=angle_det_rad[i], amplitude_k=amplitude_k, monopole_k=monopole_k
        )


def add_2f(
    tod,
    hwp_angle,
    amplitude_2f_k: float,
    optical_power_k: float,
):
    """Add the HWP differential emission to some time-ordered data

    This functions modifies the values in `tod` by adding the contribution of the HWP
    synchronous signal coming from differential emission. The `amplitude_2f_k` argument must be
    a N_dets array containing the amplitude of the HWPSS. The `optical_power_k` argument must have
    the same size and contain the value of the nominal optical power for the considered frequency channel."""

    assert len(tod.shape) == 2
    num_of_dets = tod.shape[0]

    if isinstance(amplitude_2f_k, Number):
        amplitude_2f_k = np.array([amplitude_2f_k] * num_of_dets)

    if isinstance(optical_power_k, Number):
        optical_power_k = np.array([optical_power_k] * num_of_dets)

    assert len(amplitude_2f_k) == num_of_dets
    assert len(optical_power_k) == num_of_dets

    for detector_idx in range(tod.shape[0]):
        add_2f_for_one_detector(
            tod_det=tod[detector_idx],
            angle_det_rad=hwp_angle,
            amplitude_k=amplitude_2f_k[detector_idx],
            monopole_k=optical_power_k[detector_idx],
        )


def add_2f_to_observations(
    observations: Union[Observation, List[Observation]],
    hwp: Optional[HWP] = None,
    component: str = "tod",
    amplitude_2f_k: Union[float, None] = None,
    optical_power_k: Union[float, None] = None,
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

        if optical_power_k is None:
            optical_power_k = cur_obs.optical_power_k

        if hwp is None:
            if hasattr(cur_obs, "hwp_angle"):
                hwp_angle = cur_obs.hwp_angle
            else:
                hwp_angle = None
        else:
            hwp_angle = get_hwp_angle(cur_obs, hwp)

        add_2f(
            tod=getattr(cur_obs, component),
            hwp_angle=hwp_angle,
            amplitude_2f_k=amplitude_2f_k,
            optical_power_k=optical_power_k,
        )
