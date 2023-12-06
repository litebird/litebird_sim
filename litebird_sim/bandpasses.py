# -*- encoding: utf-8 -*-

from dataclasses import dataclass

import numpy as np
import scipy as sp
import logging
from typing import Optional, Union
from uuid import UUID

from .imo import Imo


def _apod(x, a, b):
    return (1 + np.cos((x - a) / (b - a) * np.pi)) / 2


@dataclass
class BandPassInfo:
    """A class wrapping the basic information about a detector bandpass.

    This class encodes the basic properties of a frequency band.

    It can be initialized in three ways:

    - The default constructor will generate top-hat bandpasses,
      assuming that band centroid and width are provided.

    - Through the class method :meth:`.from_imo`, which reads the
      detector bandpass from the LiteBIRD Instrument Model (see
      the :class:`.Imo` class).

    Args:

        - bandcenter_ghz (float):  center frequency in GHz

        - bandwidth_ghz (float):  width of the band (default=0.)

        - bandlow_ghz (float): lowest frequency to sample the band,
          i.e. the first element of the frequency array considered
          (default: bandcenter_ghz - bandwidth_ghz)

        - bandhigh_ghz (float): highest frequency to sample the band,
          i.e. the last element of frequency array considered
          (default: bandcenter_ghz + bandwidth_ghz)

        - nsamples_inband (int): number of elements to sample the band
          (default=128)

        - name (str) : ID of the band

        - normalize(bool) : If set to true bandpass weights will be
          normalized to 1

        - bandtype (str): a string describing the band profile. It can be
          one of the following:

            - ``top-hat`` (default)

            - ``top-hat-exp``: the edges of the band are apodized with an
              exponential profile

            - ``top-hat-cosine``: the edges of the band are apodized with
              a cosine profile

            - ``cheby``: the bandpass encodes a Chebyshev profile

        - alpha_exp (float): out-of-band exponential decay index for
          low freq edge.

        - beta_exp (float) : out-of-band exponential decay index for
          high freq edge

        - cosine_apo_length (int): the numerical factor related to
          the cosine apodization length

        - cheby_poly_order (int): chebyshev filter order.

        - cheby_ripple_dB (int): maximum ripple amplitude in decibels.
    """

    bandcenter_ghz: float = 0.0
    bandwidth_ghz: float = 0.0
    bandlow_ghz: Union[float, None] = None
    bandhigh_ghz: Union[float, None] = None
    nsamples_inband: int = 128
    name: str = ""
    bandtype: str = "top-hat"
    normalize: bool = False
    model: str = None

    alpha_exp: float = 1
    beta_exp: float = 1
    cosine_apo_length: float = 5
    cheby_poly_order: int = 3
    cheby_ripple_dB: int = 3

    def __post_init__(self):
        """
        Constructor of the class

        """
        if self.bandcenter_ghz <= 0.0:
            raise ValueError(
                "The band center must be strictly positive, \
                assign a value to bandcenter_ghz > 0.0"
            )
        self.f0, self.f1 = self.get_edges()
        # we extend the wings out-of-band of the top-hat bandpass
        # before and after the edges
        if not self.bandlow_ghz:
            self.bandlow_ghz = self.bandcenter_ghz - self.bandwidth_ghz
        if not self.bandhigh_ghz:
            self.bandhigh_ghz = self.bandcenter_ghz + self.bandwidth_ghz
        self.freqs_ghz = np.linspace(
            self.bandlow_ghz, self.bandhigh_ghz, self.nsamples_inband
        )
        # checking that the bandpass edges lie within the freq range
        if self.bandlow_ghz > self.f0 or self.bandhigh_ghz < self.f1 - 1:
            raise ValueError("The bandpass is out of the frequency range")

        self.isnormalized = False
        if self.bandtype == "top-hat":
            self._get_top_hat_bandpass(normalize=self.normalize)
        elif self.bandtype == "top-hat-cosine":
            self._get_top_hat_bandpass(apodization="cosine", normalize=self.normalize)
        elif self.bandtype == "top-hat-exp":
            self._get_top_hat_bandpass(apodization="exp", normalize=self.normalize)
        elif self.bandtype == "cheby":
            self._get_chebyshev_bandpass(normalize=self.normalize)
        else:
            logging.warning(f"{self.bandtype} profile not implemented. Assume top-hat")
            self._get_top_hat_bandpass(normalize=self.normalize)

    def get_edges(self):
        """
        get the edges of the tophat band
        """
        return (
            self.bandcenter_ghz - self.bandwidth_ghz / 2,
            self.bandcenter_ghz + self.bandwidth_ghz / 2,
        )

    def _get_top_hat_bandpass(self, normalize=False, apodization: Optional[str] = None):
        """
        Sample a top-hat bandpass, given the centroid and the bandwidth.
        If the `normalize` flag is set to ``True``, the transmission
        coefficients are
        normalized so that its integral is 1 over the frequency band.
        The parameter `apodization` must be either ``"cosine"``, ``"exp"``,
        or ``None``: in the latter case, no apodization of the beamshape
        is performed.
        """

        self.weights = np.zeros_like(self.freqs_ghz)
        mask = np.ma.masked_inside(self.freqs_ghz, self.f0, self.f1).mask
        self.weights[mask] = 1.0

        if apodization == "cosine":
            self._cosine_apodize_bandpass()
        elif apodization == "exp":
            self._exp_apodize_bandpass()

        if normalize:
            self.normalize_band()

    def normalize_band(self):
        """
        Normalize the band transmission coefficients

        """
        A = np.trapz(self.weights, self.freqs_ghz)
        self.weights /= A
        self.isnormalized = True

    def _exp_apodize_bandpass(self):
        """Define a bandpass with exponential tails and unit transmission
        in band freqs: frequency in GHz
        """
        mask_beta = np.ma.masked_greater(self.freqs_ghz, self.f1).mask
        self.weights[mask_beta] = np.exp(
            -self.beta_exp * (self.freqs_ghz[mask_beta] - self.f1)
        )
        mask_alpha = np.ma.masked_less(self.freqs_ghz, self.f0).mask
        self.weights[mask_alpha] = np.exp(
            self.alpha_exp * (self.freqs_ghz[mask_alpha] - self.f0)
        )

    def _cosine_apodize_bandpass(self):
        """
        Define a bandpass with cosine tails and unit transmission in band
        """

        apolength = self.bandwidth_ghz / self.cosine_apo_length
        f_above = self.bandcenter_ghz * (1 + self.bandwidth_ghz / 2 + apolength)
        f_below = self.bandcenter_ghz * (1 - self.bandwidth_ghz / 2 - apolength)
        mask_above = np.ma.masked_inside(self.freqs_ghz, self.f1, f_above).mask

        x_ab = np.linspace(self.f1, f_above, self.freqs_ghz[mask_above].size)

        self.weights[mask_above] = _apod(x_ab, self.f1, f_above)

        mask_below = np.ma.masked_inside(self.freqs_ghz, f_below, self.f0).mask
        x_bel = np.linspace(f_below, self.f0, self.freqs_ghz[mask_below].size)
        self.weights[mask_below] = _apod(x_bel, self.f0, f_below)

    # Chebyshev profile bandpass
    def _get_chebyshev_bandpass(self, normalize=False):
        """
        Define a bandpass with chebyshev prototype.
        """
        b, a = sp.signal.cheby1(
            self.cheby_poly_order,
            self.cheby_ripple_dB,
            [2.0 * np.pi * self.f0 * 1e9, 2.0 * np.pi * self.f1 * 1e9],
            "bandpass",
            analog=True,
        )
        w, h = sp.signal.freqs(b, a, worN=self.freqs_ghz * 2 * np.pi * 1e9)

        self.weights = abs(h)
        if normalize:
            A = self.get_normalization()
            self.weights /= A
            self.isnormalized = True

    def get_normalization(self):
        """
        Estimate the integral over the frequency band
        """
        return np.trapz(self.weights, self.freqs_ghz)

    # Find effective central frequency of a bandpass profile
    def find_central_frequency(self):
        """Find the effective central frequency of
        a bandpass profile as defined in https://arxiv.org/abs/1303.5070
        """
        if self.isnormalized:
            return np.trapz(self.freqs_ghz * self.weights, self.freqs_ghz)
        else:
            return (
                np.trapz(self.freqs_ghz * self.weights, self.freqs_ghz)
                / self.get_normalization()
            )

    @staticmethod
    def from_imo(imo: Imo, url: Union[UUID, str]):
        """Create a `BandPassInfo` object from a definition in the IMO
        The `url` must either specify a UUID or a full URL to the
        object.

        """
        obj = imo.query(url)
        return BandPassInfo.from_dict(obj.metadata)

    def _interpolate_band(self):
        """
        This function aims at building the sampler in order to
        generate random samples
        statistically equivalent to the model bandpass
        """
        # normalize band

        if not self.isnormalized:
            self.normalize_band()
        # Interpolate the band
        b = sp.interpolate.interp1d(x=self.freqs_ghz, y=self.weights)
        # estimate the CDF
        Pnu = np.array(
            [
                sp.integrate.quad(b, a=self.freqs_ghz.min(), b=inu)[0]
                for inu in self.freqs_ghz[1:]
            ]
        )
        # interpolate the inverse CDF
        self.Sampler = sp.interpolate.interp1d(
            Pnu,
            self.freqs_ghz[:-1] + np.diff(self.freqs_ghz),
            bounds_error=False,
            fill_value="extrapolate",
        )

    def bandpass_resampling(self, bstrap_size=1000, nresample=54, model=None):
        """
        Resample a  bandpass with bootstrap resampling.

        Note that the user can provide any sampler built with the
        `interpolate_band` method; otherwise, an error will be raised!

        This function should be used when the user wants to generate
        many realizations of bandpasses, e.g. per detector bands.
        There is no need to initialize many
        instances of the class :class:`.BandPassInfo`, just rerun this
        functions multiple times issuing the same bandpass model instance.

        Args :

        - `bstrap_size` (int): encodes the size of the random dataset
           to be generated from the Sampler

        - `nresample` (int): define how fine is the grid for the
           resampled bandpass

        - `model` (BandPassInfo.model): We can resample from a model previously
           constructed with this function. The default value is set to ``None``:
           in this case, it initializes the bandpass sampler with the model set
           in the class instance (recommended use).
        """

        if model is not None:
            logging.info(f"Bandpass sampler from {model.name }")
            sampler = model.Sampler
        else:
            try:
                sampler = self.Sampler
            except AttributeError:
                print(
                    "Can't resample if no sampler is built and/or provided, "
                    "initializing the sampler and interpolating the band"
                )
                self._interpolate_band()
                sampler = self.Sampler
            except AttributeError:
                logging.warning(
                    "Can't resample if no sampler is built and/or provided, "
                    "initializing the sampler and interpolating the band"
                )
                self._interpolate_band()
                sampler = self.Sampler

        X = np.random.uniform(size=bstrap_size)
        bins_nu = np.linspace(self.freqs_ghz.min(), self.freqs_ghz.max(), nresample)
        h, xb = np.histogram(sampler(X), density=True, bins=bins_nu)

        nu_b = xb[:-1] + np.diff(xb)
        resampled_bpass = abs(
            sp.interpolate.interp1d(
                nu_b, h, kind="cubic", bounds_error=False, fill_value=0.0
            )(self.freqs_ghz)
        )
        if self.isnormalized:
            return resampled_bpass / np.trapz(resampled_bpass, self.freqs_ghz)
        else:
            return resampled_bpass
