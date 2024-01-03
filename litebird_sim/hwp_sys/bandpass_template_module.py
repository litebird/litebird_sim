import numpy as np
from typing import Union
from scipy import signal
from litebird_sim import BandPassInfo
import logging


# Chebyshev profile lowpass
def lowpass_chebyshev(freqs, f0, order=3, ripple_dB=0.2):
    """Define a lowpass with chebyshev prototype

    Args:
      freqs (np.array): frequency in GHz
      f0 (float): low-frequency edge of the band in GHz
      order (float): chebyshev filter order (default: 3)
      ripple_dB (float): maximum ripple amplitude in decibels (default: 0.2)

    Returns:
      The bandpass transmission
    """
    b, a = signal.cheby1(
        order, ripple_dB, 2.0 * np.pi * f0 * 1e9, "lowpass", analog=True
    )
    w, h = signal.freqs(b, a, worN=freqs * 2 * np.pi * 1e9)

    transmission = abs(h)

    return transmission


# Find effective central frequency of a bandpass profile
def find_central_frequency(freqs, bandpass):
    """Find the effective central frequency of
     a bandpass profile as defined in https://arxiv.org/abs/1303.5070

    Args:
      freqs (np.array): frequency in GHz
      bandpass (np.array): transmission profile
    Returns:
      central frequency
    """
    df = freqs[1] - freqs[0]

    fc = sum(freqs * bandpass * df) / sum(bandpass * df)

    return fc


# Add high frequency leakage to a bandpass profile
def add_high_frequency_transmission(freqs, bandpass, location=3, transmission=0.5):
    """Add high frequency leakage

    Args:
      freqs (np.array): frequency in GHz
      bandpass (np.array): transmission profile
      location (float): multiple of the central frequency of the bandpass profile
        where add the leakage (default: 3)
      transmission (float): relative amplitude of the high frequency leakage
        with respect to the nominal band (default: 0.5)

    Returns:
      Frequency range and bandpass transmission
    """

    df = freqs[1] - freqs[0]
    fc = find_central_frequency(freqs, bandpass)

    diff_freq = abs(freqs - fc)
    i_fc = np.where(diff_freq == min(diff_freq))[0][0]
    delta_fc = abs(freqs[-1] - freqs[i_fc])

    high_freq_fc = location * fc

    new_freqs_min = freqs[0]
    new_freqs_max = high_freq_fc + delta_fc

    freqs_new = np.linspace(
        freqs[0], new_freqs_max, int((new_freqs_max - new_freqs_min) / df + 1)
    )
    bandpass_new = np.zeros_like(freqs_new)
    # finding the shifted position of freqs[0] i.e. the position
    # in freqs_new from which the band is transmitted at a higher frequency
    i_nf0 = np.where(np.round(freqs_new - (high_freq_fc - delta_fc)) == 0)[0][0]

    for i in range(len(freqs_new)):
        if i < len(freqs):
            bandpass_new[i] = bandpass[i]

        elif i >= int(i_nf0):
            bandpass_new[i] = transmission * bandpass[i - int(i_nf0)]

    return freqs_new, bandpass_new


# Beam throughput
def beam_throughtput(freqs):
    r"""Beam throughtput factor

    Args:
      freqs (np.array): frequency in GHz

    Returns:
      :math:`1/\nu^2`
    """
    return 1.0 / freqs / freqs / 1.0e9 / 1.0e9


# Define bandpass profile
def bandpass_profile(
    freqs: Union[np.array, None] = None,
    bandpass: Union[dict, None] = None,
    include_beam_throughput: Union[bool, None] = None,
):
    r"""
    Returns the bandpass profile either reading it from a file or
    computing it through the class `BandPassInfo`

    Args:
      freqs (np.array): the array of frequencies considered
      bandpass (dict): dictionary with either the key "bandpass_file"
       pointing to a txt file with a column of freqs and a column of
       bandpass profile (attention: the frequencies read from the file
       have to coincide with the input array `freqs`) or with a key
       "band_type" (same naming convention as `bandtype` in `BandPassInfo`).
       In the second case, you should also define the edges of the bandpass,
       `band_high_edge` and `band_low_edge`, that would otherwise be set to
       the lowest and highest value of `freqs`.
      include_beam_throughput (bool): if True, divides by :math:`\nu^2`

    Returns:
      Array of frequency range (same as `freqs`) and bandpass profile
    """

    profile = np.ones_like(freqs)

    if "bandpass_file" in bandpass.keys():
        try:
            f, profile = np.loadtxt(
                bandpass["bandpass_file"], unpack=True, comments="#"
            )
        except Exception:
            print("missing bandpass file or wrong number of columns")

        if not np.allclose(freqs, f, atol=1e-5):
            raise ValueError("wrong frequencies in bandpass file")

    elif "band_type" in bandpass.keys():
        if "band_alpha" not in bandpass.keys():
            bandpass["band_alpha"] = 1
        if "band_beta" not in bandpass.keys():
            bandpass["band_beta"] = 1
        if "cosine_apo_length" not in bandpass.keys():
            bandpass["cosine_apo_length"] = 5
        if "band_order" not in bandpass.keys():
            bandpass["band_order"] = 3
        if "band_ripple_dB" not in bandpass.keys():
            bandpass["band_ripple_dB"] = 0.2
        if "normalize" not in bandpass.keys():
            bandpass["normalize"] = False
        if "band_high_edge" not in bandpass.keys():
            bandpass["band_high_edge"] = freqs[-1]
            logging.warning(
                "high band edge not defined,\
                          assigned to highest frequency"
            )
        if "band_low_edge" not in bandpass.keys():
            bandpass["band_low_edge"] = freqs[0]
            logging.warning(
                "low band edge not defined,\
                          assigned to lowest frequency"
            )

        bandclass = BandPassInfo(
            bandcenter_ghz=bandpass["bandcenter_ghz"],
            bandwidth_ghz=bandpass["band_high_edge"] - bandpass["band_low_edge"],
            bandlow_ghz=freqs[0],
            bandhigh_ghz=freqs[-1],
            nsamples_inband=freqs.size,
            alpha_exp=bandpass["band_alpha"],
            beta_exp=bandpass["band_beta"],
            cosine_apo_length=bandpass["cosine_apo_length"],
            cheby_poly_order=bandpass["band_order"],
            cheby_ripple_dB=bandpass["band_ripple_dB"],
            normalize=bandpass["normalize"],
            bandtype=bandpass["band_type"],
        )

        freqs = bandclass.freqs_ghz
        profile = bandclass.weights
    else:
        raise ValueError(
            "bandpass not defined, \
                         assign bandpass_file or \
                         band_type in bandpass dict"
        )

    if (
        "band_high_freq_leak" in bandpass.keys()
        and bandpass["band_high_freq_leak"] is True
    ):
        if (
            "band_high_freq_loc" not in bandpass.keys()
            and "band_high_freq_trans" not in bandpass.keys()
        ):
            freqs, profile = add_high_frequency_transmission(freqs, profile)

        else:
            freqs, profile = add_high_frequency_transmission(
                freqs,
                profile,
                bandpass["band_high_freq_loc"],
                bandpass["band_high_freq_trans"],
            )

    if include_beam_throughput is True:
        profile = profile * beam_throughtput(freqs)

    return freqs, profile
