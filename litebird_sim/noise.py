import numpy as np
from numba import njit

def generate_white_noise(nsamp, sigma):
    return numpy.random.normal(0, sigma, size=nsamp)

def generate_one_over_f_noise(nsamp, fknee, alpha):
    return np.zeros(nsamp)
