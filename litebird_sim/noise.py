# -*- encoding: utf-8 -*-

from numbers import Number
from typing import List, Union

import numpy as np
import scipy as sp
from numba import njit

from litebird_sim import Observation


def nearest_pow2(data):
    """returns the next largest power of 2 that will encompass the full data set

    data: 1-D numpy array
    """
    return int(2 ** np.ceil(np.log2(len(data))))


def add_white_noise(data, sigma: float, random=None):
    """Adds white noise with the given sigma to the array data.

    To be called from add_noise_to_observations.

    Args:

        `data` : 1-D numpy array

        `sigma` : white noise level

        `random` : a random number generator if you want reproducible randomness
    """
    if random is None:
        random = np.random.default_rng()

    data += random.normal(0, sigma, data.shape)


@njit
def build_one_over_f_model(ft, freqs, fknee_mhz, alpha, sigma):
    fknee_hz = fknee_mhz / 1000

    # Skip the first element, as it is the constant offset
    for i in range(1, len(ft)):
        ft[i] *= np.sqrt((1 + pow(abs(freqs[i]) / fknee_hz, -alpha))) * sigma
    ft[0] = 0


def add_one_over_f_noise(
    data,
    fknee_mhz: float,
    alpha: float,
    sigma: float,
    sampling_rate_hz: float,
    random=None,
):
    """Adds a 1/f noise timestream with the given f knee and alpha to data
    To be called from add_noise_to_observations

    Args:

        `data` : 1-D numpy array

        `fknee_mhz` : knee frequency in mHz

        `alpha` : low frequency spectral tilt

        `sigma` : white noise level

        `sampling_rate_hz` : the sampling frequency of the data

        `random` : a random number generator if you want reproducible randomness
    """

    if random is None:
        random = np.random.default_rng()

    noiselen = nearest_pow2(data)

    # makes a white noise timestream with unit variance
    noise = random.normal(0, 1, noiselen)

    ft = sp.fft.fft(noise, n=noiselen)
    freqs = sp.fft.fftfreq(noiselen, d=1 / (2 * sampling_rate_hz))

    # filters the white noise in the frequency domain with the 1/f filter
    build_one_over_f_model(ft, freqs, fknee_mhz, alpha, sigma)

    # transforms the data back to the time domain
    ifft = sp.fft.ifft(ft)

    data += np.real(ifft[: len(data)])


def rescale_noise(net_ukrts: float, sampling_rate_hz: float, scale: float):
    return net_ukrts * np.sqrt(sampling_rate_hz) * scale / 1e6


def add_noise(
    tod,
    noise_type: str,
    sampling_rate_hz: float,
    net_ukrts,
    fknee_mhz,
    alpha,
    scale=1.0,
    random=None,
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

    The parameter `random`, if specified, must be a random number generator that
    implements the ``normal`` method. You should typically use the `random` field
    of a :class:`.Simulation` object for this.

    The parameters `net_ukrts`, `fknee_mhz`, `alpha`, and `scale` can either be scalars
    or arrays; in the latter case, their size must be the same as ``tod.shape[0]``,
    which is the number of detectors in the TOD.
    """
    assert len(tod.shape) == 2
    num_of_dets = tod.shape[0]

    if isinstance(net_ukrts, Number):
        net_ukrts = np.array([net_ukrts] * num_of_dets)

    if isinstance(fknee_mhz, Number):
        fknee_mhz = np.array([fknee_mhz] * num_of_dets)

    if isinstance(alpha, Number):
        alpha = np.array([alpha] * num_of_dets)

    if isinstance(scale, Number):
        scale = np.array([scale] * num_of_dets)

    assert len(net_ukrts) == num_of_dets
    assert len(fknee_mhz) == num_of_dets
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
    obs: List[Observation],
    noise_type: str,
    scale: float = 1.0,
    random: Union[np.random.Generator, None] = None,
):
    """Add noise of the defined type to the observations in obs

    This class provides an interface to the low-level function :func:`.add_noise`.
    The parameter `obs` is a list of :class:`.Observation` objects, which are
    typically taken from the field `observations` of a :class:`.Simulation` object.
    Unlike :func:`.add_noise`, it is not needed to pass the noise parameters here,
    as they are taken from the characteristics of the detectors saved in `obs`.

    See :func:`.add_noise` for more information.
    """
    if noise_type not in ["white", "one_over_f"]:
        raise ValueError("Unknown noise type " + noise_type)

    # iterate through each observation
    for i, ob in enumerate(obs):
        add_noise(
            tod=ob.tod,
            noise_type=noise_type,
            sampling_rate_hz=ob.sampling_rate_hz,
            net_ukrts=ob.net_ukrts,
            fknee_mhz=ob.fknee_mhz,
            alpha=ob.alpha,
            scale=scale,
            random=random,
        )
