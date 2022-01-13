# -*- encoding: utf-8 -*-

from numbers import Number
import numpy as np
import scipy as sp


def nearest_pow2(data):
    """returns the next largest power of 2 that will encompass the full data set

    data: 1-D numpy array
    """
    return int(2 ** np.ceil(np.log2(len(data))))


def add_white_noise(data, sigma, random=None):
    """Adds white noise with the given sigma to the array data
    To be called from add_noise_to_observations.

    data: 1-D numpy array
    sigma_uk: white noise level
    random: a random number generator if you want reproducible randomness
    """
    if random is None:
        random = np.random.default_rng()

    data += random.normal(0, sigma, data.shape)


def add_one_over_f_noise(data, fknee_mhz, alpha, sigma, freq_hz, random=None):
    """Adds a 1/f noise timestream with the given f knee and alpha to data
    To be called from add_noise_to_observations

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

    # makes a white noise timestream with unit variance
    noise = random.normal(0, 1, noiselen)

    ft = sp.fft.fft(noise, n=noiselen)
    freqs = sp.fft.fftfreq(noiselen, d=1 / (2 * freq_hz))

    # filters the white noise in the frequency domain with the 1/f filter

    mask = freqs != 0

    # We are soon no longer using "freqs", so we reuse the memory allocated for it to
    # build the 1/f model profile
    model = freqs

    model[~mask] = 0
    model[mask] = (
        np.sqrt((1 + pow(abs(freqs[mask]) / (fknee_mhz / 1000), -1 * alpha))) * sigma
    )

    # transforms the data back to the time domain
    ifft = sp.fft.ifft(ft * model)

    data += np.real(ifft[: len(data)])


def rescale_noise(net_ukrts, sampling_rate_hz, scale):
    return net_ukrts * np.sqrt(sampling_rate_hz) * scale / 1e6


def add_noise(
    tod, sampling_rate_hz, net_ukrts, fknee_mhz, alpha, noisetype, scale=1, random=None
):
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
        if noisetype == "white":
            add_white_noise(
                tod[i][:],
                rescale_noise(
                    net_ukrts=net_ukrts[i],
                    sampling_rate_hz=sampling_rate_hz,
                    scale=scale[i],
                ),
                random=random,
            )
        elif noisetype == "one_over_f":
            add_one_over_f_noise(
                tod[i][:],
                fknee_mhz[i],
                alpha[i],
                rescale_noise(
                    net_ukrts=net_ukrts[i],
                    sampling_rate_hz=sampling_rate_hz,
                    scale=scale[i],
                ),
                sampling_rate_hz,
                random=random,
            )


def add_noise_to_observations(obs, noisetype, scale=1, random=None):
    """adds noise of the defined type to the observations in obs

    Args:
        obs (:class:`Observation`): an Observation object

        noisetype (str): 'white' or 'one_over_f'. In the latter case, 1/f *and*
                       white noise will be present

        scale (float): multiplicative factor used to rescale the noise timeline.
                       The default produces noise in K

        random: a random number generator (default is None, which will use the default
                generator)
    """
    if noisetype not in ["white", "one_over_f"]:
        raise ValueError("Unknown noise type " + noisetype)

    # iterate through each observation
    for i, ob in enumerate(obs):
        add_noise(
            tod=ob.tod,
            noisetype=noisetype,
            sampling_rate_hz=ob.sampling_rate_hz,
            net_ukrts=ob.net_ukrts,
            fknee_mhz=ob.fknee_mhz,
            alpha=ob.alpha,
            scale=scale,
            random=random,
        )
