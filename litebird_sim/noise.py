# -*- encoding: utf-8 -*-

import numpy as np
import scipy as sp


def add_noise(obs, noisetype, scale=1, random=None):
    """adds noise of the defined type to the observations in obs

    Args:
        obs (:class:`Observation`): an Observation object

        noisetype (str): 'white' or 'one_over_f'

        scale (float): multiplicative factor used to rescale the noise timeline.
                       The default produces noise in K

        random: a random number generator (default is None, which will use the default
                generator)
    """
    if noisetype not in ["white", "one_over_f"]:
        raise ValueError("Unknown noise type " + noisetype)

    # iterate through each observation
    for ob in obs:
        assert len(ob.tod.shape) == 2
        for i in range(ob.tod.shape[0]):
            if noisetype == "white":
                generate_white_noise(
                    ob.tod[i][:],
                    ob.net_ukrts[i] * np.sqrt(ob.sampling_rate_hz) * scale / 1e6,
                    random=random,
                )
            elif noisetype == "one_over_f":
                generate_one_over_f_noise(
                    ob.tod[i][:],
                    ob.fknee_mhz[i],
                    ob.alpha[i],
                    ob.net_ukrts[i] * np.sqrt(ob.sampling_rate_hz) * scale / 1e6,
                    ob.sampling_rate_hz,
                    random=random,
                )


def generate_white_noise(data, sigma, random=None):
    """Adds white noise with the given sigma to the array data
    To be called from add_noise.

    data: 1-D numpy array
    sigma_uk: white noise level
    random: a random number generator if you want reproducible randomness
    """
    if random is None:
        random = np.random.default_rng()

    data += random.normal(0, sigma, data.shape)


def generate_one_over_f_noise(data, fknee_mhz, alpha, sigma, freq_hz, random=None):
    """Adds a 1/f noise timestream with the given f knee and alpha to data
    To be called from add_noise

    data: 1-D numpy array
    fknee: knee frequency
    alpha: low frequency spectral tilt
    sigma0_uk: white noise level
    freq: the sampling frequency of the data
    random: a random number generator if you want reproducible randomness
    """

    if random is None:
        random = np.random.default_rng()

    noiselen = nearest_pow2(data)

    # makes a white noise timestream with unit varience
    noise = random.normal(0, 1, noiselen)

    ft = sp.fft.fft(noise, n=noiselen)
    freqs = sp.fft.fftfreq(noiselen, d=1 / (2 * freq_hz))

    # filters the white noise in the frequency domain with the 1/f filter

    model = freqs
    # This is what the style checker wants but it looks rediculous to me
    model[freqs != 0] = (
        np.sqrt((1 + pow(abs(freqs[freqs != 0]) / (fknee_mhz / 1000), -1 * alpha)))
        * sigma
    )

    model[freqs == 0] = 0

    # transforms the data back to the time domain
    ifft = sp.fft.ifft(ft * model)

    data += np.real(ifft[: len(data)])


def nearest_pow2(data):
    """returns the next largest power of 2 that will encompass the full data set

    data: 1-D numpy array
    """
    return int(2 ** np.ceil(np.log2(len(data))))
