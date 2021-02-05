import numpy as np
from numba import njit

import numpy as np
import math
import scipy as sp

def add_noise(obs, noisetype):
    ''' adds noise of the defined type to the observations in obs

    obs: an Observation object
    noisetype: 'white' or 'one_over_f'
    '''
    if noisetype not in ['white', 'one_over_f']:
        raise ValueError("Unknown noise type " + noisetype)



    #iterate through each observation
    for ob in obs:
        if len(ob.tod.shape) == 1:
            #single detector data

            if noisetype == 'white':
                generate_white_noise(ob.tod, ob.net_ukrthrz)
            elif noisetype == 'one_over_f':
                generate_one_over_f_noise(ob.tod, ob.fknee_mhz, ob.alpha, ob.net_ukrthz, ob.sampling_rate_hz)

        elif len(ob.tod.shape) == 2:
            for i in range(ob.tod.shape[0]):
                if noisetype == 'white':
                    generate_white_noise(ob.tod[i][:], ob.net_ukrthrz)
                elif noisetype == 'one_over_f':
                    generate_one_over_f_noise(ob.tod[i][:], ob.fknee_mhz, ob.alpha, ob.net_ukrthrz, ob.sampling_rate_hz)

        else:
            raise TypeError("Array with shape " + ob.tod.shape + " not supported in add_noise. Try a 1 or 2 D array instead.")

def generate_white_noise(data, sigma):
    '''Adds white noise with the given sigma to the array data
    To be called from add_noise.

    data: 1-D numpy array
    sigma: the varience of the desired white noise

    '''
    data += numpy.random.normal(0, sigma, size=len(data))

def generate_one_over_f_noise(data, fknee, alpha, sigma0, freq):
    '''Adds a 1/f noise timestream with the given f knee and alpha to data
    To be called from add_noise

    data: 1-D numpy array
    fknee: knee frequency
    alpha: low frequency spectral tilt
    sigma0: white noise level
    freq: the sampling frequency of the data
    '''

    noiselen = nearest_pow2(data)

    noise = numpy.random.normal(0, 1, size=noiselen)

    ft = sp.fftpack.fft(noise, n=noiselen)
    freqs = sp.fftpack.fftfreq(noiselen, d=1/freq)

    model = sigma0*sigma0*(1 + pow(freqs/fknee, alpha))

    ifft = sp.fftpack.ifft(ft*model)

    data += ifft[:len(data)]


def nearest_pow2(data):
    '''returns the next largest power of 2 that will encompass the full data set
    
    data: 1-D numpy array
    '''
    return pow(2, math.ceil(math.log(len(data), 2)))
