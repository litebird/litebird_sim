from dataclasses import dataclass

import numpy as np

from .hwp_harmonics.hwp_harmonics import _dBodTth
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
    det_name: str | None = None,
    nl_params: NonLinParams | None = None,
    g_one_over_k: float | None = None,
    random: np.random.Generator | None = None,
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
    det_bandcenter_ghz: np.float64 | None = None,
    det_bandwidth_ghz: np.float64 | None = None,
    nl_params: NonLinParams | None = None,
    random: np.random.Generator | None = None,
    conv_K_to_SR: bool = False,
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

        conv_K_to_SR (bool, optional): Flag for temperature to spectral radiance
            units conversion. Defaults to False.

        det_freq_ghz (np.float64, optional): Detector central frequency in GHz.
            Used in the K to SR conversion. Defaults to None.

        det_bandwidth_ghz (np.float64, optional): Detector bandwidth in GHz.
            Used in the K to SR conversion. Defaults to None.

    """
    assert random is not None, (
        "You should pass a random number generator which implements the `normal` method."
    )
    if nl_params is None:
        nl_params = NonLinParams()

    g_nonlin = random.normal(
        loc=nl_params.sampling_gaussian_loc,
        scale=nl_params.sampling_gaussian_scale,
    )

    if conv_K_to_SR:
        import pysm3.units as u

        assert det_bandcenter_ghz is not None and det_bandwidth_ghz is not None, (
            "You should pass det_bandcenter_ghz and det_bandwidth_ghz when conv_K_to_SR is set to True."
        )

        g_inv = (1 / g_nonlin) * u.K_CMB
        gn2 = (
            g_inv.to(
                u.MJy / u.sr,
                equivalencies=u.cmb_equivalencies(det_bandcenter_ghz * u.GHz),
            )
            * det_bandwidth_ghz
            * 1e9
        )

    for i in range(len(tod_det)):
        tod_det[i] += (1 / gn2) * tod_det[i] ** 2


def apply_quadratic_nonlin(
    tod: np.ndarray,
    bandcenter_ghz: np.ndarray,
    bandwidth_ghz: np.ndarray,
    nl_params: NonLinParams | None = None,
    dets_random: list[np.random.Generator] | None = None,
    conv_K_to_SR: bool = False,
):
    """Apply a quadratic nonlinearity to some time-ordered data

    This functions modifies the values in `tod` by adding a parabolic function of the TOD itself.
    """

    assert len(tod.shape) == 2
    assert dets_random is not None, "dets_random is required"

    for detector_idx in range(tod.shape[0]):
        apply_quadratic_nonlin_for_one_detector(
            tod_det=tod[detector_idx],
            nl_params=nl_params,
            random=dets_random[detector_idx],
            conv_K_to_SR=conv_K_to_SR,
            det_bandcenter_ghz=bandcenter_ghz[detector_idx],
            det_bandwidth_ghz=bandwidth_ghz[detector_idx],
        )


def apply_quadratic_nonlin_to_observations(
    observations: Observation | list[Observation],
    nl_params: NonLinParams | None = None,
    component: str = "tod",
    user_seed: int | None = None,
    conv_K_to_SR: bool = False,
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
        apply_quadratic_nonlin_to_observations(sim.observations, component="nl_tod", user_seed=1234)

    Parameters
    ----------
    observations : Observation or list of Observation
        A single `Observation` instance or a list of them.
    nl_params : NonLinParams, optional
        Parameters defining the quadratic non-linearity model. If not
        provided, a default configuration is used.
    component : str, optional
        Name of the TOD attribute to modify. Defaults to `"tod"`.
    user_seed : int or None
            Base seed to build the RNG hierarchy and generate samples of the detector non-linearity factor,
            independently of the MPI distribution across detectors and time.
            The user is required to set this parameter.
    conv_K_to_SR (bool, optional): Flag for temperature to spectral radiance
        units conversion. Defaults to False.

    Raises
    ------
    TypeError
        If `observations` is neither an `Observation` nor a list of them.
    ValueError
        If `user_seed` is not provided.
    AssertionError
        If the number of random generators does not match the number of detectors.
    """

    assert user_seed is not None, (
        "user_seed must be given in apply_quadratic_nonlin_to_observations."
    )

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
        comm=obs_list[0].comm_time_block,
        user_seed=user_seed,
    )

    # iterate through each observation
    for cur_obs in obs_list:
        tod = getattr(cur_obs, component)
        bandcenter_ghz = getattr(cur_obs, "bandcenter_ghz")
        bandwidth_ghz = getattr(cur_obs, "bandwidth_ghz")

        apply_quadratic_nonlin(
            tod=tod,
            nl_params=nl_params,
            dets_random=dets_random,
            conv_K_to_SR=conv_K_to_SR,
            bandcenter_ghz=bandcenter_ghz,
            bandwidth_ghz=bandwidth_ghz,
        )
