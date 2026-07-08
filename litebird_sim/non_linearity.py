from dataclasses import dataclass

import numpy as np

from .observations import Observation
from .seeding import regenerate_or_check_detector_generators
from .units import Units, UnitUtils


@dataclass
class NonLinParams:
    """
    A class to store the non-lineairty injection parameters.

        - ``sampling_gaussian_loc`` (`float`): Mean of the Gaussian distribution.

        - ``sampling_gaussian_scale`` (`float`): Standard deviation of the Gaussian
          distribution.

        - ``user_seed`` (`int` or None) : Base seed to build the RNG hierarchy and
            generate samples of the detector non-linearity factor, independently of
            the MPI distribution across detectors and time. By setting it to
            `None`, the generation of random numbers will not be reproducible.

        - ``units`` (`Units`): units of the sampled non linearity value. Must be one
            of the pysm3/astropy units accepted by the lbs.Units module.

    """

    # Parameters for sampling distribution
    sampling_gaussian_loc: float = 0.0
    sampling_gaussian_scale: float = 0.1
    user_seed: int | None = None
    units: Units = Units.K_CMB


def apply_quadratic_nonlin_for_one_sample(
    data,
    g_nonlin: float | None = None,
    conv_factor_nl: float = 1.0,
):
    assert g_nonlin is not None, "Either `g_nonlin` or `det_name` must be provided."

    g_nonlin = (1 / conv_factor_nl) * g_nonlin

    return data + g_nonlin * data**2


def apply_quadratic_nonlin_for_one_detector(
    tod_det,
    tod_units: Units,
    bandcenter_ghz_det: float,
    nl_params: NonLinParams | None = None,
    random: np.random.Generator | None = None,
):
    """This function applies the quadratic non-linearity on the TOD corresponding to
    only one detector.

    Args:

        tod_det (np.ndarray): The TOD array corresponding to only one
          detector.

        nl_params (:class:`.NonLinParams`, optional): The non-linearity
          injection parameters object. Defaults to None.

        random (np.random.Generator, optional): A random number generator.
          Defaults to None.



    """
    assert random is not None, (
        "You should pass a random number generator which implements the `normal` method."
    )
    assert nl_params is not None, "You should pass a NonLinParams object."

    g_nonlin = random.normal(
        loc=nl_params.sampling_gaussian_loc,
        scale=nl_params.sampling_gaussian_scale,
    )

    if nl_params.units != tod_units:
        conv_factor_nl = UnitUtils.get_conversion_factor(
            nl_params.units,
            tod_units,
            bandcenter_ghz_det,
        )
    else:
        conv_factor_nl = 1

    g_nonlin = (1 / conv_factor_nl) * g_nonlin

    for i in range(len(tod_det)):
        tod_det[i] += g_nonlin * tod_det[i] ** 2


def apply_quadratic_nonlin(
    tod: np.ndarray,
    tod_units: Units,
    bandcenter_ghz: np.ndarray,
    nl_params: NonLinParams | None = None,
    dets_random: list[np.random.Generator] | None = None,
):
    """Apply a quadratic nonlinearity to some time-ordered data

    This functions modifies the values in `tod` by adding a parabolic function of the TOD itself.
    """

    assert len(tod.shape) == 2
    assert dets_random is not None, "dets_random is required"

    for detector_idx in range(tod.shape[0]):
        apply_quadratic_nonlin_for_one_detector(
            tod_det=tod[detector_idx],
            tod_units=tod_units,
            bandcenter_ghz_det=bandcenter_ghz[detector_idx],
            nl_params=nl_params,
            random=dets_random[detector_idx],
        )


def apply_quadratic_nonlin_to_observations(
    observations: Observation | list[Observation],
    nl_params: NonLinParams | None = None,
    component: str = "tod",
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

    Raises
    ------
    TypeError
        If `observations` is neither an `Observation` nor a list of them.
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

    user_seed = (
        nl_params.user_seed
        if nl_params.user_seed is not None
        else np.random.randint(1000)
    )

    dets_random = regenerate_or_check_detector_generators(
        observations=obs_list,
        comm=obs_list[0].comm_time_block,
        user_seed=user_seed,
    )

    # iterate through each observation
    for obs_idx, cur_obs in enumerate(obs_list):
        tod = getattr(cur_obs, component)
        tod_units = cur_obs.tod_list[obs_idx].units
        bandcenter_ghz = getattr(cur_obs, "bandcenter_ghz")

        apply_quadratic_nonlin(
            tod=tod,
            tod_units=tod_units,
            nl_params=nl_params,
            dets_random=dets_random,
            bandcenter_ghz=bandcenter_ghz,
        )
