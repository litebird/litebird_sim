# -*- encoding: utf-8 -*-

from typing import List, Union

from numba import njit, prange

from .observations import Observation


@njit
def apply_quadratic_nonlin_for_one_sample(data, g_one_over_k):
    return data + g_one_over_k * data**2


@njit(parallel=True)
def apply_quadratic_nonlin_for_one_detector(tod_det, g_one_over_k):
    for i in prange(len(tod_det)):
        tod_det[i] += g_one_over_k * tod_det[i] ** 2


def apply_quadratic_nonlin(tod, g_one_over_k: float):
    """Apply a quadratic nonlinearity to some time-ordered data

    This functions modifies the values in `tod` by adding a parabolic function of the TOD itself.
    The `g_one_over_k` argument must be a N_dets array containing the amplitude of the nonlinear gain to be applied."""

    assert len(tod.shape) == 2
    num_of_dets = tod.shape[0]
    assert len(g_one_over_k) == num_of_dets

    for detector_idx in range(tod.shape[0]):
        apply_quadratic_nonlin_for_one_detector(
            tod_det=tod[detector_idx],
            g_one_over_k=g_one_over_k[detector_idx],
        )


def apply_quadratic_nonlin_to_observations(
    observations: Union[Observation, List[Observation]],
    component: str = "tod",
    g_one_over_k: Union[float, None] = None,
):
    """Apply a quadratic nonlinearity to some time-ordered data

    This is a wrapper around the :func:`.apply_quadratic_nonlin` function that applies to the TOD
    stored in `observations`, which can either be one :class:`.Observation` instance
    or a list of observations.

    By default, the modified TOD is ``Observation.tod``. If you want to modify some
    other field of the :class:`.Observation` class, use `component`::

        for cur_obs in sim.observations:
        # Allocate a new TOD for the nonlinear TOD alone
        cur_obs.nl_tod = np.zeros_like(cur_obs.tod)

        # Ask `apply_quadratic_nonlin_to_observations` to store the nonlinear TOD
        # in `observations.nl_tod`
        apply_quadratic_nonlin_to_observations(sim.observations, component="nl_tod")
    """

    if isinstance(observations, Observation):
        obs_list = [observations]
    else:
        obs_list = observations

    # iterate through each observation
    for cur_obs in obs_list:
        if g_one_over_k is None:
            g_one_over_k = cur_obs.g_one_over_k

        apply_quadratic_nonlin(
            tod=getattr(cur_obs, component),
            g_one_over_k=g_one_over_k,
        )
