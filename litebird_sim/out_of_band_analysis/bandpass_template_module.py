import litebird_sim as lbs
import numpy as np
from typing import Union, List
from scipy import signal

# Top-hat Bandpass
def top_hat_bandpass(freqs, f0, f1):
    """Define a top-hat bandpass
    freqs: frequency in GHz
    f0: low-frequency edge of the top-hat in GHz
    f1: high-frequency edge of the top-hat in GHz
    """
    transmission = np.zeros_like(freqs)

    for i in range(len(freqs)):

        if freqs[i] >= f0 and freqs[i] <= f1:

            transmission[i] = 1.0

        else:

            transmission[i] = 0.0

    return transmission

# Top-hat bandpass with exponential wings
def top_hat_bandpass_with_exp_wings(freqs, f0, f1, alpha = 1, beta = 1):
    """Define a bandpass with exponential tails and unit transmission in band
    freqs: frequency in GHz
    f0: low-frequency edge of the band in GHz
    f1: high-frequency edge of the band in GHz
    alpha: out-of-band exponential decay index for low freq edge
    beta: out-of-band exponential decay index for high freq edge
    
    If alpha and beta are not specified a value of 1 is used for both.
    
    """
    transmission = np.zeros_like(freqs)

    for i in range(len(freqs)):

        if freqs[i] >= f0 and freqs[i] <= f1:

            transmission[i] = 1.0

        elif freqs[i] > f1:

            transmission[i] = np.exp(-beta * (freqs[i] - f1))

        elif freqs[i] < f0:

            transmission[i] = np.exp(alpha * (freqs[i] - f0))

    return transmission

# Chebyshev profile bandpass
def bandpass_chebyshev(freqs, f0, f1, order = 3, ripple_dB = 3):
    """Define a bandpass with chebyshev prototype
    freqs: frequency in GHz
    f0: low-frequency edge of the band in GHz
    f1: high-frequency edge of the band in GHz
    order: chebyshev filter order
    ripple_dB: maximum ripple amplitude in decibels
    
    If order and ripple_dB are not specified a value of 3 is used for both.
    
    """
    b, a = signal.cheby1(order, ripple_dB, [2.*np.pi*f0*1e9, 2.*np.pi*f1*1e9], 'bandpass', analog=True)
    w, h = signal.freqs(b, a, worN=freqs*2*np.pi*1e9)
    
    transmission = abs(h)

    return transmission

# Chebyshev profile lowpass
def lowpass_chebyshev(freqs, f0, order = 1, ripple_dB = 1):
    """Define a lowpass with chebyshev prototype
    freqs: frequency in GHz
    f0: low-frequency edge of the band in GHz
    order: chebyshev filter order
    ripple_dB: maximum ripple amplitude in decibels
    
    If order and ripple_dB are not specified a value of 3 is used for both.
    
    """
    b, a = signal.cheby1(order, ripple_dB, 2.*np.pi*f0*1e9, 'lowpass', analog=True)
    w, h = signal.freqs(b, a, worN=freqs*2*np.pi*1e9)
    
    transmission = abs(h)

    return transmission

# Find effective central frequency of a bandpass profile
def find_central_frequency(freqs, bandpass):
    """Find the effective central frequency of 
    a bandpass profile as defined in https://arxiv.org/abs/1303.5070
    freqs: frequency in GHz
    bandpass: transmission profile
    """
    df = freqs[1]-freqs[0]
    
    fc = sum(freqs*bandpass*df)/sum(bandpass*df)
    
    return fc

# Add high frequency leakage to a bandpass profile  
def add_high_frequency_transmission(freqs, bandpass, location = 3, transmission = 0.5):
    """Add high frequency leakage
    freqs: frequency in GHz
    bandpass: transmission profile
    location: multiple of the central frequency of the bandpass profile where add the leakage
    transmission: relative amplitude of the high frequency leakage with respect to the nominal band
    
    If location and transmission are not specified a value of 3 and 0.5 are set by default.
    
    """
    
    df = freqs[1]-freqs[0]
    fc = find_central_frequency(freqs, bandpass)
    
    diff_freq = abs(freqs-fc)
    i_fc = np.where(diff_freq == min(diff_freq))[0][0]
    delta_fc = abs(freqs[-1] - freqs[i_fc])
    
    high_freq_fc = location*fc
    
    new_freqs_min = freqs[0]
    new_freqs_max = high_freq_fc + delta_fc

        
    freqs_new = np.linspace(freqs[0], new_freqs_max, int((new_freqs_max-new_freqs_min)/df + 1))
    bandpass_new = np.zeros_like(freqs_new)
    #finding the shifted position of freqs[0] i.e. the position in freqs_new from which the band is transmitted at a higher frequency 
    i_nf0 = np.where(np.round(freqs_new - (high_freq_fc-delta_fc)) == 0)[0][0] 
    
    for i in range(len(freqs_new)):
        
        if i < len(freqs):
            
            bandpass_new[i] = bandpass[i]
            
        elif i >= int(i_nf0):
            bandpass_new[i] = transmission*bandpass[i-int(i_nf0)]

    return freqs_new, bandpass_new

# Beam throughput
def beam_throughtput(freqs):
    """Beam throughtput factor
    freqs: frequency in GHz
    """
    return 1.0 / freqs / freqs / 1.0e9 / 1.0e9

# Define bandpass profile
def bandpass_profile(
 freqs: Union[np.array, None] = None,
 bandpass: Union[dict, None] = None,
 include_beam_throughput: Union[bool, None] = None,
):
    
    profile = np.ones_like(freqs)
    
    if 'bandpass_file' in bandpass.keys():
        
        profile = np.ones_like(freqs)
        print('We can not process bandpass files yet. This will be implemented in the future. For now we define a top-hat between ', freqs[0], ' and ', freqs[-1], ' GHz.')
               
    elif 'band_type' in bandpass.keys():
    
        if bandpass['band_type'] == 'top_hat':
        
            profile = top_hat_bandpass(freqs,
                                       bandpass['band_low_edge'],
                                       bandpass['band_high_edge'],
                                       )
                                           
        elif bandpass['band_type'] == 'top_hat_with_exp_wings':
        
            if 'band_alpha' not in bandpass.keys() and 'band_beta' not in bandpass.keys():
        
                profile = top_hat_bandpass_with_exp_wings(freqs,
                                                          bandpass['band_low_edge'],
                                                          bandpass['band_high_edge'],
                                                          )
                                           
            else:
            
                profile = top_hat_bandpass_with_exp_wings(freqs,
                                                              bandpass['band_low_edge'],
                                                              bandpass['band_high_edge'],
                                                              bandpass['band_alpha'],
                                                              bandpass['band_beta']
                                                              )
                                           
        elif bandpass['band_type'] == 'chebyshev':
        
            if 'band_order' not in bandpass.keys() and 'band_ripple_dB' not in bandpass.keys():
        
                profile = bandpass_chebyshev(freqs,
                                        bandpass['band_low_edge'],
                                        bandpass['band_high_edge'],
                                        )
                                        
            else:
            
                profile = bandpass_chebyshev(freqs,
                                        bandpass['band_low_edge'],
                                        bandpass['band_high_edge'],
                                        bandpass['band_order'],
                                        bandpass['band_ripple_dB']
                                        )
                                        
    if 'band_high_freq_leak' in bandpass.keys() and bandpass['band_high_freq_leak'] == True:
    
        if 'band_high_freq_loc' not in bandpass.keys() and 'band_high_freq_trans' not in bandpass.keys():
    
            freqs, profile = add_high_frequency_transmission(freqs, profile)
            
        else:
        
            freqs, profile = add_high_frequency_transmission(freqs, 
                                                             profile, 
                                                             bandpass['band_high_freq_loc'],
                                                             bandpass['band_high_freq_trans'])
                                                             
    if include_beam_throughput == True:
    
        profile = profile * beam_throughtput(freqs)
                                                             
    return freqs, profile
