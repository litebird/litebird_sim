# -*- encoding: utf-8 -*-

from numbers import Number
from typing import List, Union

import numpy as np
import scipy as sp
from numba import njit

from .observations import Observation


def nearest_pow2(data):
    """returns the next largest power of 2 that will encompass the full data set

    data: 1-D numpy array
    """
    return int(2 ** np.ceil(np.log2(len(data))))


def add_white_noise(data, sigma: float, random):
    """Adds white noise with the given sigma to the array data.

    To be called from add_noise_to_observations.

    Args:

        `data` : 1-D numpy array

        `sigma` : white noise level

        `random` : a random number generator that implements the ``normal`` method.
                   You should typically use the `random` field of a :class:`.Simulation`
                   object for this. It must be specified
    """
    data += random.normal(0, sigma, data.shape)


@njit
def build_one_over_f_model(ft, freqs, fknee_mhz, fmin_hz, alpha, sigma):
    fknee_hz_alpha = pow(fknee_mhz / 1000, alpha)
    fmin_hz_alpha = pow(fmin_hz, alpha)

    # Skip the first element, as it is the constant offset
    for i in range(1, len(ft)):
        f_hz_alpha = pow(abs(freqs[i]), alpha)
        ft[i] *= (
            np.sqrt((f_hz_alpha + fknee_hz_alpha) / (f_hz_alpha + fmin_hz_alpha))
            * sigma
        )
    ft[0] = 0


def add_one_over_f_noise(
    data,
    fknee_mhz: float,
    fmin_hz: float,
    alpha: float,
    sigma: float,
    sampling_rate_hz: float,
    random,
):
    """Adds a 1/f noise timestream with the given f knee and alpha to data
    To be called from add_noise_to_observations

    Args:

        `data` : 1-D numpy array

        `fknee_mhz` : knee frequency in mHz

        `fmin_hz` : kmin frequency for high pass in Hz

        `alpha` : low frequency spectral tilt

        `sigma` : white noise level

        `sampling_rate_hz` : the sampling frequency of the data

        `random` : a random number generator that implements the ``normal`` method.
                   You should typically use the `random` field of a :class:`.Simulation`
                   object for this. It must be specified
    """

    noiselen = nearest_pow2(data)

    # makes a white noise timestream with unit variance
    noise = random.normal(0, 1, noiselen)

    noise = sp.fft.rfft(noise, overwrite_x=True)
    freqs = sp.fft.rfftfreq(noiselen, d=1 / (2 * sampling_rate_hz))

    # filters the white noise in the frequency domain with the 1/f filter
    build_one_over_f_model(noise, freqs, fknee_mhz, fmin_hz, alpha, sigma)

    # transforms the data back to the time domain
    noise = sp.fft.irfft(noise, overwrite_x=True)

    data += noise[: len(data)]


def rescale_noise(net_ukrts: float, sampling_rate_hz: float, scale: float):
    return net_ukrts * np.sqrt(sampling_rate_hz) * scale / 1e6


def add_noise(
    tod,
    noise_type: str,
    sampling_rate_hz: float,
    net_ukrts,
    fknee_mhz,
    fmin_hz,
    alpha,
    random,
    scale=1.0,
):
    """
    Add noise (white or 1/f) to a 2D array of floating-point values

    This function sums an array of random number following a white noise model with
    an optional 1/f component to `data`, which is assumed to be a D×N array containing
    the TOD for D detectors, each containing N samples.

    The parameter `noisetype` must either be ``white`` or ``one_over_f``; in the latter
    case, the noise will contain a 1/f part and a white noise part.

    The parameter `scale` can be used to introduce measurement unit conversions when
    appropriate. Default units: [K].

    The parameter `random` must be specified and must be a random number generator that
    implements the ``normal`` method. You should typically use the `random` field
    of a :class:`.Simulation` object for this.

    The parameters `net_ukrts`, `fknee_mhz`, `fmin_hz`, `alpha`, and `scale` can
    either be scalars or arrays; in the latter case, their size must be the same as
    ``tod.shape[0]``, which is the number of detectors in the TOD.
    """
    assert len(tod.shape) == 2
    num_of_dets = tod.shape[0]

    if isinstance(net_ukrts, Number):
        net_ukrts = np.array([net_ukrts] * num_of_dets)

    if isinstance(fknee_mhz, Number):
        fknee_mhz = np.array([fknee_mhz] * num_of_dets)

    if isinstance(fmin_hz, Number):
        fmin_hz = np.array([fmin_hz] * num_of_dets)

    if isinstance(alpha, Number):
        alpha = np.array([alpha] * num_of_dets)

    if isinstance(scale, Number):
        scale = np.array([scale] * num_of_dets)

    assert len(net_ukrts) == num_of_dets
    assert len(fknee_mhz) == num_of_dets
    assert len(fmin_hz) == num_of_dets
    assert len(alpha) == num_of_dets
    assert len(scale) == num_of_dets

    for i in range(num_of_dets):
        if noise_type == "white":
            add_white_noise(
                data=tod[i][:],
                sigma=rescale_noise(
                    net_ukrts=net_ukrts[i],
                    sampling_rate_hz=sampling_rate_hz,
                    scale=scale[i],
                ),
                random=random,
            )
        elif noise_type == "one_over_f":
            add_one_over_f_noise(
                data=tod[i][:],
                fknee_mhz=fknee_mhz[i],
                fmin_hz=fmin_hz[i],
                alpha=alpha[i],
                sigma=rescale_noise(
                    net_ukrts=net_ukrts[i],
                    sampling_rate_hz=sampling_rate_hz,
                    scale=scale[i],
                ),
                sampling_rate_hz=sampling_rate_hz,
                random=random,
            )


def add_noise_to_observations(
    obs: Union[Observation, List[Observation]],
    noise_type: str,
    random: np.random.Generator,
    scale: float = 1.0,
    component: str = "tod",
):
    """Add noise of the defined type to the observations in obs

    This class provides an interface to the low-level function :func:`.add_noise`.
    The parameter `obs` can either be one :class:`.Observation` instance or a list
    of observations, which are typically taken from the field `observations` of a
    :class:`.Simulation` object. Unlike :func:`.add_noise`, it is not needed to
    pass the noise parameters here, as they are taken from the characteristics of
    the detectors saved in `obs`. The parameter `random` must be specified and must
    be a random number generator that implements the ``normal`` method. You should
    typically use the `random` field of a :class:`.Simulation` object for this.

    By default, the noise is added to ``Observation.tod``. If you want to add it to some
    other field of the :class:`.Observation` class, use `component`:

        for cur_obs in sim.observations:
            # Allocate a new TOD for the noise alone
            cur_obs.noise_tod = np.zeros_like(cur_obs.tod)

        # Ask `add_noise_to_observations` to store the noise
        # in `obs.noise_tod`
        add_noise_to_observations(sim.observations, …, component="noise_tod")

    See :func:`.add_noise` for more information.
    """
    if noise_type not in ["white", "one_over_f"]:
        raise ValueError("Unknown noise type " + noise_type)

    if isinstance(obs, Observation):
        obs_list = [obs]
    else:
        obs_list = obs

    # iterate through each observation
    for i, cur_obs in enumerate(obs_list):
        add_noise(
            tod=getattr(cur_obs, component),
            noise_type=noise_type,
            sampling_rate_hz=cur_obs.sampling_rate_hz,
            net_ukrts=cur_obs.net_ukrts,
            fknee_mhz=cur_obs.fknee_mhz,
            fmin_hz=cur_obs.fmin_hz,
            alpha=cur_obs.alpha,
            scale=scale,
            random=random,
        )
