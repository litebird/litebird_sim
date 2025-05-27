# -*- encoding: utf-8 -*-

import numpy as np
import hashlib
from typing import Union, List
from dataclasses import dataclass

from .observations import Observation
from .seeding import RNGHierarchy


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


def _hash_function(
    input_str: str,
    user_seed: int = 12345,
    rank: int = 0,
    _seed_type: str = "non-linear",
) -> int:
    """This functions generates a unique and reproducible hash for a given pair of
    `input_str` and `user_seed`. This hash is used to generate the common noise time
    stream for a group of detectors, and to introduce randomness in the noise time
    streams.

    Args:

        input_str (str): A string, for example, the detector name.

        user_seed (int): A seed provided by the user. Defaults to 12345.

    Returns:

        int: An `md5` hash from generated from `input_str` and `user_seed`
    """
    assert isinstance(input_str, str), "The parameter `input_str` must be a string!"

    bytesobj = (input_str + str(user_seed) + _seed_type + str(rank)).encode("utf-8")

    hashobj = hashlib.md5()
    hashobj.update(bytesobj)
    digest = hashobj.digest()

    return int.from_bytes(bytes=digest, byteorder="little")


def apply_quadratic_nonlin_for_one_sample(
    data,
    det_name: str = None,
    nl_params: NonLinParams = None,
    user_seed: Union[int, None] = None,
    g_one_over_k: float = None,
    random: Union[np.random.Generator, None] = None,
):
    if user_seed is not None and random is not None:
        raise ValueError(
            "You should pass only one between 'user_seed' and 'random', not both."
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

        if random is None:
            assert user_seed is not None, (
                "You should either pass a random generator that that implements the 'normal' method, or an integer seed, none are provided."
            )
            random = np.random.default_rng(seed=_hash_function(det_name, user_seed))
        g_one_over_k = random.normal(
            loc=nl_params.sampling_gaussian_loc,
            scale=nl_params.sampling_gaussian_scale,
        )

    return data + g_one_over_k * data**2


def apply_quadratic_nonlin_for_one_detector(
    tod_det,
    det_name: str,
    nl_params: NonLinParams = None,
    user_seed: Union[int, None] = None,
    random: Union[np.random.Generator, None] = None,
    rank: int = 0,
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
    if user_seed is not None and random is not None:
        raise ValueError(
            "You should pass only one between 'user_seed' and 'random', not both."
        )

    if nl_params is None:
        nl_params = NonLinParams()

    if random is None:
        assert user_seed is not None, (
            "You should either pass a random generator that that implements the 'normal' method, or an integer seed, none are provided."
        )
        assert isinstance(det_name, str), "The parameter `det_name` must be a string"
        random = np.random.default_rng(seed=_hash_function(det_name, user_seed, rank))

    g_one_over_k = random.normal(
        loc=nl_params.sampling_gaussian_loc,
        scale=nl_params.sampling_gaussian_scale,
    )

    for i in range(len(tod_det)):
        tod_det[i] += g_one_over_k * tod_det[i] ** 2


def apply_quadratic_nonlin(
    tod: np.ndarray,
    det_name: Union[List, np.ndarray],
    nl_params: NonLinParams = None,
    user_seed: Union[int, None] = None,
    rank: int = 0,
    dets_random: Union[np.random.Generator, None] = None,
):
    """Apply a quadratic nonlinearity to some time-ordered data

    This functions modifies the values in `tod` by adding a parabolic function of the TOD itself.
    """

    assert len(tod.shape) == 2

    for detector_idx in range(tod.shape[0]):
        apply_quadratic_nonlin_for_one_detector(
            det_name=det_name[detector_idx],  # --> questo poi lo definisco nella obs
            tod_det=tod[detector_idx],
            nl_params=nl_params,  # --> questo dovrebbe essere una realiz diversa per ogni det automaticamente
            user_seed=user_seed,
            rank=rank,
            random=dets_random[detector_idx],
        )


def apply_quadratic_nonlin_to_observations(
    observations: Union[Observation, List[Observation]],
    nl_params: NonLinParams = None,
    component: str = "tod",
    user_seed: Union[int, None] = None,
    dets_random: Union[List[np.random.Generator]] = None,
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
    num_detectors = getattr(obs_list[0], component).shape[0]
    if user_seed is not None:
        RNG_hierarchy = RNGHierarchy(user_seed, num_detectors, len(obs_list[0].name))
        dets_random = RNG_hierarchy.get_detector_level_generators_on_rank(
            rank=obs_list[0].comm.rank
        )
    if user_seed is None and dets_random is None:
        raise ValueError("You should pass either `user_seed` or `dets_random`.")
    assert len(dets_random) == num_detectors, (
        "The number of random generators must match the number of detectors"
    )

    # iterate through each observation
    for cur_obs in obs_list:
        tod = getattr(cur_obs, component)
        det_name = cur_obs.name

        apply_quadratic_nonlin(
            tod=tod,
            det_name=det_name,
            nl_params=nl_params,
            user_seed=user_seed,
            rank=cur_obs.comm.rank,
            dets_random=dets_random,
        )
