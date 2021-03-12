import numpy as np
import math
import scipy as sp


def add_noise(obs, noisetype, random=None):
    ''' adds noise of the defined type to the observations in obs

    obs: an Observation object
    noisetype: 'white' or 'one_over_f'
    random: a random number generator if you want reproducible randomness
    '''
    if noisetype not in ['white', 'one_over_f']:
        raise ValueError("Unknown noise type " + noisetype)

    # iterate through each observation
    for ob in obs:
        if len(ob.tod.shape) == 1:
            # single detector data
            # I'm not sure if this mode can ever be called because it's possible
            # that the array is always 2-D

            if noisetype == 'white':
                generate_white_noise(
                    ob.tod,
                    ob.net_ukrts / np.sqrt(ob.sampling_rate_hz),
                    random=random)
            elif noisetype == 'one_over_f':
                generate_one_over_f_noise(
                    ob.tod,
                    ob.fknee_mhz,
                    ob.alpha,
                    ob.net_ukrts / np.sqrt(ob.sampling_rate_hz),
                    ob.sampling_rate_hz,
                    random=random)

        elif len(ob.tod.shape) == 2:
            for i in range(ob.tod.shape[0]):
                if noisetype == 'white':
                    generate_white_noise(
                        ob.tod[i][:],
                        ob.net_ukrts[i] / np.sqrt(ob.sampling_rate_hz),
                        random=random)
                elif noisetype == 'one_over_f':
                    generate_one_over_f_noise(
                        ob.tod[i][:],
                        ob.fknee_mhz[i],
                        ob.alpha[i],
                        ob.net_ukrts[i] / np.sqrt(ob.sampling_rate_hz),
                        ob.sampling_rate_hz,
                        random=random)

        else:
            raise TypeError("Array with shape " + ob.tod.shape
                            + " not supported in add_noise."
                            + " Try a 1 or 2 D array instead.")


def generate_white_noise(data, sigma_uk, random=None):
    '''Adds white noise with the given sigma to the array data
    To be called from add_noise.

    data: 1-D numpy array
    sigma: the varience of the desired white noise
    random: a random number generator if you want reproducible randomness
    '''
    if random is None:
        random = np.random.default_rng()

    data += random.normal(0, sigma_uk / 1000000, data.shape)


def generate_one_over_f_noise(data, fknee_mhz, alpha, sigma0_uk, freq_hz, random=None):
    '''Adds a 1/f noise timestream with the given f knee and alpha to data
    To be called from add_noise

    data: 1-D numpy array
    fknee: knee frequency
    alpha: low frequency spectral tilt
    sigma0: white noise level
    freq: the sampling frequency of the data
    random: a random number generator if you want reproducible randomness
    '''

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
    model[freqs != 0] = np.sqrt((1 + pow(abs(freqs[freqs != 0])
                                         / (fknee_mhz / 1000), -1 * alpha))) \
        * (sigma0_uk / 1000000)

    model[freqs == 0] = 0

    # transforms the data back to the time domain
    ifft = sp.fft.ifft(ft * model)

    data += np.real(ifft[:len(data)])


def nearest_pow2(data):
    '''returns the next largest power of 2 that will encompass the full data set

    data: 1-D numpy array
    '''
    return pow(2, math.ceil(math.log(len(data), 2)))
