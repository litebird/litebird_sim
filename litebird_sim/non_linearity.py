# -*- encoding: utf-8 -*-

import numpy as np
from typing import Union, List
from dataclasses import dataclass

from .observations import Observation
from .seeding import regenerate_or_check_detector_generators


@dataclass
class NonLinParams:
    """
    A class to store the non-lineairty injection parameters.

        - ``sampling_gaussian_loc`` (`float`): Mean of the Gaussian distribution.

        - ``sampling_gaussian_scale`` (`float`): Standard deviation of the Gaussian
          distribution.

    """

    # Parameters for sampling distribution
    sampling_gaussian_loc: float = 0.0
    sampling_gaussian_scale: float = 0.1


def apply_quadratic_nonlin_for_one_sample(
    data,
    det_name: str = None,
    nl_params: NonLinParams = None,
    g_one_over_k: float = None,
    random: np.random.Generator = None,
):
    assert random is not None, (
        "You should pass a random number generator which implements the `normal` method."
    )

    if nl_params is None:
        nl_params = NonLinParams()

    # If g_one_over_k is not provided, we must use det_name to generate it
    # Having a non-random g_one_over_k is useful to avoid over-complications in the hwp_sys module
    if g_one_over_k is None:
        if det_name is None:
            raise ValueError("Either `g_one_over_k` or `det_name` must be provided.")

        if not isinstance(det_name, str):
            raise TypeError(
                "`det_name` must be a string when `g_one_over_k` is not provided."
            )

        g_one_over_k = random.normal(
            loc=nl_params.sampling_gaussian_loc,
            scale=nl_params.sampling_gaussian_scale,
        )

    return data + g_one_over_k * data**2


def apply_quadratic_nonlin_for_one_detector(
    tod_det,
    nl_params: NonLinParams = None,
    random: np.random.Generator = None,
):
    """This function applies the quadratic non-linearity on the TOD corresponding to
    only one detector.

    Args:

        det_tod (np.ndarray): The TOD array corresponding to only one
          detector.

        det_name (str): The name of the detector to which the TOD belongs.
          This name is used with ``user_seed`` to generate hash. This hash is used to
          set random slope in case of linear drift, and randomized detector mismatch
          in case of thermal gain drift.

        nl_params (:class:`.NonLinParams`, optional): The non-linearity
          injection parameters object. Defaults to None.

        user_seed (int, optional): A seed provided by the user. Defaults
          to None.

        random (np.random.Generator, optional): A random number generator.
          Defaults to None.
    """
    assert random is not None, (
        "You should pass a random number generator which implements the `normal` method."
    )
    if nl_params is None:
        nl_params = NonLinParams()

    g_one_over_k = random.normal(
        loc=nl_params.sampling_gaussian_loc,
        scale=nl_params.sampling_gaussian_scale,
    )

    for i in range(len(tod_det)):
        tod_det[i] += g_one_over_k * tod_det[i] ** 2


def apply_quadratic_nonlin(
    tod: np.ndarray,
    nl_params: NonLinParams = None,
    dets_random: Union[np.random.Generator, None] = None,
):
    """Apply a quadratic nonlinearity to some time-ordered data

    This functions modifies the values in `tod` by adding a parabolic function of the TOD itself.
    """

    assert len(tod.shape) == 2

    for detector_idx in range(tod.shape[0]):
        apply_quadratic_nonlin_for_one_detector(
            tod_det=tod[detector_idx],
            nl_params=nl_params,
            random=dets_random[detector_idx],
        )


def apply_quadratic_nonlin_to_observations(
    observations: Union[Observation, List[Observation]],
    nl_params: NonLinParams = None,
    component: str = "tod",
    user_seed: Union[int, None] = None,
    dets_random: Union[List[np.random.Generator]] = None,
):
    """
    Apply a quadratic nonlinearity to some time-ordered data

    This is a wrapper around the :func:`.apply_quadratic_nonlin` function that applies to the TOD
    stored in `observations`, which can either be one :class:`.Observation` instance
    or a list of observations. It ensures proper setup of per-detector
    random number generators using either a user-provided seed or a list of
    pre-initialized RNGs.

    By default, the modified TOD is ``Observation.tod``. If you want to modify some
    other field of the :class:`.Observation` class, use `component`::

        for cur_obs in sim.observations:
        # Allocate a new TOD for the nonlinear TOD alone
        cur_obs.nl_tod = np.zeros_like(cur_obs.tod)

        # Ask `apply_quadratic_nonlin_to_observations` to store the nonlinear TOD
        # in `observations.nl_tod`
        apply_quadratic_nonlin_to_observations(sim.observations, component="nl_tod")

    Parameters
    ----------
    observations : Observation or list of Observation
        A single `Observation` instance or a list of them.
    nl_params : NonLinParams, optional
        Parameters defining the quadratic non-linearity model. If not
        provided, a default configuration is used.
    component : str, optional
        Name of the TOD attribute to modify. Defaults to `"tod"`.
    user_seed : int, optional
        Base seed to build the RNG hierarchy and generate detector-level RNGs
        that overwrite any eventual `dets_random`. Required if `dets_random`
        is not provided.
    dets_random : list of np.random.Generator, optional
        List of per-detector random number generators. If not provided, and
        `user_seed` is given, generators are created internally. One of
        `user_seed` or `dets_random` must be provided.

    Raises
    ------
    TypeError
        If `observations` is neither an `Observation` nor a list of them.
    ValueError
        If neither `user_seed` nor `dets_random` is provided.
    AssertionError
        If the number of random generators does not match the number of detectors.
    """
    if nl_params is None:
        nl_params = NonLinParams()

    if isinstance(observations, Observation):
        obs_list = [observations]
    elif isinstance(observations, list):
        obs_list = observations
    else:
        raise TypeError(
            "The parameter `observations` must be an `Observation` or a list of `Observation`."
        )
    dets_random = regenerate_or_check_detector_generators(
        observations=obs_list,
        user_seed=user_seed,
        dets_random=dets_random,
    )

    # iterate through each observation
    for cur_obs in obs_list:
        tod = getattr(cur_obs, component)

        apply_quadratic_nonlin(
            tod=tod,
            nl_params=nl_params,
            dets_random=dets_random,
        )
