import numpy as np
import scipy as sp
from numba import njit
from ducc0.misc import OofaNoise

from .observations import Observation
from .seeding import regenerate_or_check_detector_generators


# --- TRANSFER FUNCTIONS (MODELS) ---


@njit
def _transfer_toast(f_sq, fknee_hz, fmin_hz, alpha):
    """
    Toast Model: PSD ~ (f^alpha + fknee^alpha) / (f^alpha + fmin^alpha)
    Used commonly in simple 1/f simulations.
    """
    # Note: f_sq is f^2, so we need sqrt(f_sq) to get f
    f = np.sqrt(f_sq)
    f_alpha = np.power(f, alpha)
    fknee_alpha = np.power(fknee_hz, alpha)
    fmin_alpha = np.power(fmin_hz, alpha)

    # Amplitude is sqrt(PSD)
    return np.sqrt((f_alpha + fknee_alpha) / (f_alpha + fmin_alpha))


@njit
def _transfer_keshner(f_sq, fknee_hz, fmin_hz, alpha):
    """
    Keshner Model (DUCC-style): PSD ~ ((f^2 + fknee^2) / (f^2 + fmin^2))^(alpha/2)
    This corresponds to a sum of superposition of relaxation processes.
    """
    fknee_sq = fknee_hz * fknee_hz
    fmin_sq = fmin_hz * fmin_hz

    ratio = (f_sq + fknee_sq) / (f_sq + fmin_sq)

    # Amplitude is sqrt(PSD), so exponent is alpha/4
    return np.power(ratio, alpha / 4.0)


# --- CORE LOGIC ---


@njit
def apply_transfer_function(ft, freqs, fknee_mhz, fmin_hz, alpha, sigma, model_id):
    """
    Applies the selected transfer function model to the Fourier coefficients.

    Parameters
    ----------
    ft : array-like
        The Fourier transform of the white noise (modified in-place).
    freqs : array-like
        The frequency bins corresponding to `ft`.
    fknee_mhz : float
        The knee frequency in mHz.
    fmin_hz : float
        The minimum frequency in Hz.
    alpha : float
        The spectral index (slope).
    sigma : float
        The white noise level (standard deviation).
    model_id : int
        The identifier for the model to use:
        0 = 'toast'
        1 = 'keshner'
    """
    fknee_hz = fknee_mhz / 1000.0

    # Iterate over frequencies (skip DC)
    for i in range(1, len(freqs)):
        f_sq = freqs[i] * freqs[i]

        if model_id == 1:  # Keshner
            tf = _transfer_keshner(f_sq, fknee_hz, fmin_hz, alpha)
        else:  # Toast model (Default)
            tf = _transfer_toast(f_sq, fknee_hz, fmin_hz, alpha)

        ft[i] *= sigma * tf

    ft[0] = 0.0


def nearest_pow2(data):
    """
    Returns the next largest power of 2 that will encompass the full data set.

    Parameters
    ----------
    data : 1-D numpy array
        The input data array.
    """
    return int(2 ** np.ceil(np.log2(len(data))))


def add_white_noise(data, sigma: float, random):
    """
    Adds white noise with the given sigma to the array data.

    Parameters
    ----------
    data : 1-D numpy array
        The input data array (modified in-place).
    sigma : float
        The white noise level per sample. Be sure *not* to include cosmic ray
        loss, repointing maneuvers, etc., as these affect the integration time
        but **not** the white noise per sample.
    random : numpy.random.Generator
        A random number generator that implements the ``normal`` method.
        This is typically obtained from the RNGHierarchy of the ``Simulation`` class.
    """
    data += random.normal(0, sigma, data.shape)


def add_one_over_f_noise(
    data,
    fknee_mhz: float,
    fmin_hz: float,
    alpha: float,
    sigma: float,
    sampling_rate_hz: float,
    random,
    engine: str = "fft",
    model: str = "toast",
):
    """
    Adds 1/f noise to the data array using a specific engine and physical model.

    This function supports multiple noise generation engines (FFT-based synthesis
    and DUCC0's time-domain filtering) and multiple physical models for the
    noise power spectral density (PSD).

    Parameters
    ----------
    data : 1-D numpy array
        The input time-ordered data (TOD) array (modified in-place).
    fknee_mhz : float
        The knee frequency in mHz.
    fmin_hz : float
        The minimum frequency in Hz below which the spectrum flattens.
    alpha : float
        The spectral slope (e.g., 1.0 for pink noise).
    sigma : float
        The white noise standard deviation (RMS) per sample.
    sampling_rate_hz : float
        The sampling rate of the data in Hz.
    random : numpy.random.Generator
        The random number generator instance.
    engine : str, optional
        The computational method used to generate the noise. Defaults to ``"fft"``.

        * ``"fft"``: Generates noise in the Fourier domain. Supports all `model` types.
        * ``"ducc"``: Uses ``ducc0.misc.OofaNoise`` (time-domain filtering).
          Supports **only** the ``"keshner"`` model. Very efficient for long streams.
    model : str, optional
        The physical model for the Power Spectral Density (PSD). Defaults to ``"toast"``.

        * ``"toast"``: The classic power-law ratio model.
          :math:`P(f) \\propto (f^\\alpha + f_{knee}^\\alpha) / (f^\\alpha + f_{min}^\\alpha)`
        * ``"keshner"``: The model implemented by DUCC0 (sum of relaxation processes).
          :math:`P(f) \\propto ((f^2 + f_{knee}^2) / (f^2 + f_{min}^2))^{\\alpha/2}`

    Raises
    ------
    ValueError
        If the ``engine`` is unknown or if the selected ``engine`` does not support
        the requested ``model``.
    """

    if engine == "fft":
        # 1. Generate unit variance white noise
        noiselen = nearest_pow2(data)
        noise = random.normal(0, 1, noiselen)

        # 2. FFT
        noise_ft = sp.fft.rfft(noise, overwrite_x=True)
        freqs = sp.fft.rfftfreq(noiselen, d=1 / sampling_rate_hz)

        # 3. Apply Model
        # Map string to integer ID for Numba
        model_id = 1 if model == "keshner" else 0
        apply_transfer_function(
            noise_ft, freqs, fknee_mhz, fmin_hz, alpha, sigma, model_id
        )

        # 4. IFFT
        noise_final = sp.fft.irfft(noise_ft, overwrite_x=True)
        data += noise_final[: len(data)]

    elif engine == "ducc":
        if model != "keshner":
            raise ValueError(
                f"DUCC engine only supports 'keshner' model. Got '{model}'. "
                "Use engine='fft' for other models, or set model='keshner' to use DUCC."
            )

        fknee_hz = fknee_mhz / 1000.0

        # DUCC0 Logic (Slope must be negative)
        oofa = OofaNoise(
            sigmawhite=sigma,  # Correct normalization (Time RMS)
            f_knee=fknee_hz,
            f_min=fmin_hz,
            f_samp=sampling_rate_hz,
            slope=-alpha,  # DUCC expects negative slope
        )

        gauss_input = random.normal(0, 1, data.shape[-1])
        data += oofa.filterGaussian(gauss_input)

    elif engine == "random_walk":
        # Legacy alias for DUCC
        add_one_over_f_noise(
            data,
            fknee_mhz,
            fmin_hz,
            alpha,
            sigma,
            sampling_rate_hz,
            random,
            engine="ducc",
            model="keshner",
        )
    else:
        raise ValueError(f"Unknown engine '{engine}'.")


def rescale_noise(net_ukrts: float, sampling_rate_hz: float, scale: float):
    """
    Converts NET [uK*sqrt(s)] to sigma per sample [K].

    Parameters
    ----------
    net_ukrts : float or array-like
        Noise Equivalent Temperature in micro-Kelvin sqrt(seconds).
    sampling_rate_hz : float
        The sampling rate in Hz.
    scale : float
        A multiplicative scaling factor applied to the NET.

    Returns
    -------
    float or array-like
        The standard deviation (sigma) of the white noise per sample in Kelvin.
    """
    return net_ukrts * np.sqrt(sampling_rate_hz) * scale / 1e6


def add_noise(
    tod,
    noise_type,
    sampling_rate_hz,
    net_ukrts,
    fknee_mhz,
    fmin_hz,
    alpha,
    dets_random,
    scale=1.0,
    engine="fft",
    model="toast",
):
    """
    Adds noise (white or 1/f) to a TOD array for a specific detector.

    This function handles the correct broadcasting if `net_ukrts`, `fknee_mhz`, etc.,
    are arrays (indicating multiple detectors) while the `tod` is processed
    one detector at a time.

    Parameters
    ----------
    tod : ndarray
        The Time-Ordered Data array of shape (n_detectors, n_samples).
    noise_type : str
        The type of noise to add: ``"white"`` or ``"one_over_f"``.
    sampling_rate_hz : float
        Sampling rate in Hz.
    net_ukrts : float or array-like
        NET in uK*sqrt(s). Can be a scalar or an array of length n_detectors.
    fknee_mhz : float or array-like
        Knee frequency in mHz. Can be a scalar or an array of length n_detectors.
    fmin_hz : float or array-like
        Minimum frequency in Hz. Can be a scalar or an array of length n_detectors.
    alpha : float or array-like
        Spectral slope. Can be a scalar or an array of length n_detectors.
    dets_random : list of numpy.random.Generator
        List of random number generators (one per detector).
    scale : float, optional
        A multiplicative scaling factor applied to the NET. Defaults to 1.0.
    engine : str, optional
        Computation engine (``"fft"`` or ``"ducc"``). Defaults to ``"fft"``.
    model : str, optional
        Physical noise model (``"toast"`` or ``"keshner"``). Defaults to ``"toast"``.
    """
    sigma = rescale_noise(net_ukrts, sampling_rate_hz, scale)

    # Helper function to extract scalar parameters for the i-th detector
    # if the parameter is an array (representing multiple detectors).
    def _get_val(param, idx):
        if np.ndim(param) > 0:
            return param[idx]
        return param

    for idet in range(tod.shape[0]):
        # Extract scalar values for this detector
        sigma_i = _get_val(sigma, idet)
        fknee_i = _get_val(fknee_mhz, idet)
        fmin_i = _get_val(fmin_hz, idet)
        alpha_i = _get_val(alpha, idet)

        if noise_type == "white":
            add_white_noise(tod[idet], sigma_i, dets_random[idet])
        elif noise_type == "one_over_f":
            add_one_over_f_noise(
                tod[idet],
                fknee_i,
                fmin_i,
                alpha_i,
                sigma_i,
                sampling_rate_hz,
                dets_random[idet],
                engine=engine,
                model=model,
            )


def add_noise_to_observations(
    observations,
    noise_type,
    user_seed=None,
    dets_random=None,
    scale=1.0,
    component="tod",
    engine="fft",
    model="toast",
):
    """
    Adds instrumental noise (white or 1/f) to a list of Observations.

    This is the high-level interface for noise simulation. It iterates over
    detectors and observations, handling random number generator initialization
    and parameter broadcasting. It modifies the observations in-place.

    Parameters
    ----------
    observations : Observation or list of Observation
        The observation(s) to which noise will be added. Can be a single
        :class:`.Observation` instance or a list of them.
    noise_type : str
        The type of noise to generate. Options are:

        * ``"white"``: Uncorrelated Gaussian noise based on NET.
        * ``"one_over_f"``: Correlated noise characterized by
          knee frequency and spectral index.
    user_seed : int, optional
        A master integer seed used to initialize the random number generators
        if ``dets_random`` is not provided. If ``None`` and ``dets_random``
        is also ``None``, the generators will be initialized unpredictably
        (usually from the OS entropy source).
    dets_random : list of numpy.random.Generator, optional
        A list of pre-initialized random number generators, one per detector.
        If provided, ``user_seed`` is ignored. This allows for precise control
        over the RNG state for reproducibility across different calls.
    scale : float, optional
        A multiplicative scaling factor applied to the noise level (NET).
        Defaults to 1.0. Useful for simulating different noise realizations
        or scaling noise down for debugging.
    component : str, optional
        The name of the attribute in the :class:`.Observation` objects where
        the noise should be added. Defaults to ``"tod"``. Can be changed
        (e.g., to ``"noise_tod"``) to store noise separately from the signal.
    engine : str, optional
        The computational backend for 1/f noise generation. Defaults to ``"fft"``.

        * ``"fft"``: Uses Fourier synthesis. Supports all models.
        * ``"ducc"``: Uses the `ducc0` library's IIR filter. Supports only the
          ``"keshner"`` model.
    model : str, optional
        The physical model for the 1/f noise Power Spectral Density (PSD).
        Defaults to ``"toast"``.

        * ``"toast"``: :math:`P(f) \\propto (f^\\alpha + f_{knee}^\\alpha) / (f^\\alpha + f_{min}^\\alpha)`
        * ``"keshner"``: :math:`P(f) \\propto ((f^2 + f_{knee}^2) / (f^2 + f_{min}^2))^{\\alpha/2}`

    Raises
    ------
    ValueError
        If ``noise_type`` is not recognized.
    ValueError
        If the ``ducc`` engine is requested with the ``toast`` model.
    """
    if noise_type not in ["white", "one_over_f"]:
        raise ValueError("Unknown noise type " + noise_type)

    if isinstance(observations, Observation):
        obs_list = [observations]
    else:
        obs_list = observations

    dets_random = regenerate_or_check_detector_generators(
        observations=obs_list,
        user_seed=user_seed,
        dets_random=dets_random,
    )

    for cur_obs in obs_list:
        add_noise(
            tod=getattr(cur_obs, component),
            noise_type=noise_type,
            sampling_rate_hz=cur_obs.sampling_rate_hz,
            net_ukrts=cur_obs.net_ukrts * scale,
            fknee_mhz=getattr(cur_obs, "fknee_mhz"),
            fmin_hz=getattr(cur_obs, "fmin_hz"),
            alpha=getattr(cur_obs, "alpha"),
            dets_random=dets_random,
            engine=engine,
            model=model,
        )
