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


def _synthesize_one_over_f_stream(
    n_samples: int,
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
    Generate and return a 1D 1/f noise array with the requested PSD.

    This is an internal helper used by :func:`add_correlated_noise` to produce
    a noise stream that can be scaled and combined before being added to the TOD.
    It delegates to :func:`add_one_over_f_noise` on a temporary buffer.

    Parameters
    ----------
    n_samples : int
        Number of time samples to generate.
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
        Computation engine (``"fft"`` or ``"ducc"``). Defaults to ``"fft"``.
    model : str, optional
        Physical PSD model (``"toast"`` or ``"keshner"``). Defaults to ``"toast"``.

    Returns
    -------
    numpy.ndarray
        1D noise array of length ``n_samples``.
    """
    stream = np.zeros(n_samples)
    add_one_over_f_noise(
        stream, fknee_mhz, fmin_hz, alpha, sigma, sampling_rate_hz, random,
        engine=engine, model=model,
    )
    return stream


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


def _get_param_value(param, idx):
    """Extract a detector-specific scalar from a scalar or a per-detector array."""
    if np.ndim(param) > 0:
        return param[idx]
    return param


# --- CORRELATED NOISE ---


def _normalize_rho(rho, n_detectors):
    """
    Validate and broadcast the correlation coefficient *rho*.

    Parameters
    ----------
    rho : float or array-like
        Fraction of variance in the common mode.  Must be in [0, 1].
        A scalar is broadcast to all detectors; an array must have length
        ``n_detectors``.

    n_detectors : int
        Number of detectors.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``n_detectors`` with per-detector rho values.

    Raises
    ------
    ValueError
        If any value is outside [0, 1] or if the array length is wrong.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim == 0:
        rho = np.full(n_detectors, float(rho))
    if rho.shape != (n_detectors,):
        raise ValueError(
            f"rho must be a scalar or a 1-D array of length {n_detectors}, "
            f"got shape {rho.shape}."
        )
    if np.any((rho < 0) | (rho > 1)):
        raise ValueError("All rho values must be in [0, 1].")
    return rho


def _validate_grouping(n_detectors, groups):
    """
    Validate the group label array and return it as a numpy array of ints.

    Parameters
    ----------
    n_detectors : int
        Expected number of detectors.
    groups : array-like
        1-D array of integer group labels with length ``n_detectors``.

    Returns
    -------
    numpy.ndarray
        Integer array of length ``n_detectors``.

    Raises
    ------
    ValueError
        If the array has the wrong length.
    """
    groups = np.asarray(groups, dtype=int)
    if groups.shape != (n_detectors,):
        raise ValueError(
            f"groups must be a 1-D array of length {n_detectors}, "
            f"got shape {groups.shape}."
        )
    return groups


def _spawn_group_rngs_from_detectors(dets_random, groups):
    """
    Derive one deterministic RNG per group from the detector RNGs.

    The seed for group *g* is obtained by XOR-ing the bit-state integers of all
    detectors belonging to that group.  This is reproducible regardless of the
    order in which detector generators were initialised.

    Parameters
    ----------
    dets_random : list of numpy.random.Generator
        One RNG per detector.
    groups : numpy.ndarray
        Integer group-label array (length = number of detectors).

    Returns
    -------
    dict
        Mapping ``{group_label: numpy.random.Generator}``.
    """
    seeds = {}
    for idet, grp in enumerate(groups):
        # Extract the 128-bit state of the PCG generator as an integer
        state_int = int(dets_random[idet].bit_generator.state["state"]["state"])
        seeds[grp] = seeds.get(grp, 0) ^ state_int

    return {grp: np.random.default_rng(seed) for grp, seed in seeds.items()}


def _build_detector_groups(obs, grouping):
    """
    Build an integer group-label array from a grouping specification.

    Parameters
    ----------
    obs : :class:`.Observation`
        The observation whose detectors are to be grouped.
    grouping : None, str, or array-like
        Grouping specification:

        * ``None``: all detectors belong to group 0 (single shared stream).
        * ``str``: name of a per-detector attribute on *obs* (e.g.
          ``"wafer"``).  The unique values are mapped to consecutive
          integers starting from 0.
        * array-like: explicit integer group labels, one per detector.

    Returns
    -------
    numpy.ndarray
        Integer group-label array of length ``n_detectors``.
    """
    n_det = obs.n_detectors
    if grouping is None:
        return np.zeros(n_det, dtype=int)
    if isinstance(grouping, str):
        labels = getattr(obs, grouping)
        unique = {v: i for i, v in enumerate(dict.fromkeys(labels))}
        return np.array([unique[v] for v in labels], dtype=int)
    return _validate_grouping(n_det, grouping)


def add_correlated_noise(
    tod,
    sampling_rate_hz,
    net_ukrts,
    fknee_mhz,
    fmin_hz,
    alpha,
    dets_random,
    groups=None,
    rho=0.25,
    scale=1.0,
    engine="fft",
    model="toast",
    common_mode_type="one_over_f",
    corr_matrix=None,
):
    """
    Add correlated noise to a TOD.

    Two correlation models are supported, selected by which parameter is
    provided:

    **Cholesky model** (``corr_matrix`` is given)
        A full :math:`n_{det} \\times n_{det}` correlation matrix
        :math:`\\mathbf{R}` is factored as :math:`\\mathbf{R} = \\mathbf{L}
        \\mathbf{L}^T` via Cholesky decomposition.  Then :math:`n_{det}`
        independent unit-variance noise streams :math:`z_j(t)` are generated
        (each with the PSD of detector *j*) and mixed:

        .. math::

           n_i(t) = \\sigma_i \\sum_j L_{ij}\\,z_j(t)

        This supports arbitrary positive-semi-definite correlation structures.

    **Common-mode model** (``groups`` is given, default)
        Each detector *i* in group *g* receives:

        .. math::

           n_i(t) = \\sqrt{\\rho_i}\\,c_g(t) + \\sqrt{1 - \\rho_i}\\,u_i(t)

        where :math:`c_g(t)` is a shared stream for the whole group and
        :math:`u_i(t)` is a detector-unique stream.  This is a rank-1
        approximation within each group.

    Parameters
    ----------
    tod : ndarray, shape (n_detectors, n_samples)
        Time-ordered data array, modified **in-place**.
    sampling_rate_hz : float
        Sampling rate in Hz.
    net_ukrts : float or array-like
        Noise Equivalent Temperature in μK√s.  Scalar or per-detector array.
    fknee_mhz : float or array-like
        Knee frequency in mHz.  Scalar or per-detector array.
    fmin_hz : float or array-like
        Minimum frequency in Hz.  Scalar or per-detector array.
    alpha : float or array-like
        Spectral slope.  Scalar or per-detector array.
    dets_random : list of numpy.random.Generator
        One RNG per detector.
    groups : array-like of int or None, optional
        Integer group labels, length ``n_detectors``.  Used only by the
        common-mode model.  Required when ``corr_matrix`` is ``None``.
    rho : float or array-like, optional
        Fraction of variance in the common mode.  Must be in [0, 1].
        Used only by the common-mode model.  Defaults to 0.25.
    scale : float, optional
        Multiplicative factor applied to NET before unit conversion.
        Defaults to 1.0.
    engine : str, optional
        Noise-generation engine (``"fft"`` or ``"ducc"``).  Defaults to
        ``"fft"``.
    model : str, optional
        PSD model (``"toast"`` or ``"keshner"``).  Defaults to ``"toast"``.
    common_mode_type : str, optional
        PSD shape for the common-mode stream: ``"one_over_f"`` (default) or
        ``"white"``.  Used only by the common-mode model.
    corr_matrix : array-like of shape (n_detectors, n_detectors) or None, optional
        Symmetric positive-semi-definite correlation matrix.  When provided,
        the Cholesky model is used and ``groups`` / ``rho`` /
        ``common_mode_type`` are ignored.  The diagonal should be 1 (unit
        variance before per-detector sigma scaling).

    Raises
    ------
    ValueError
        If ``corr_matrix`` is provided but is not square, not the right size,
        or not positive-semi-definite.
    ValueError
        If ``corr_matrix`` is ``None`` and ``groups`` is also ``None``.
    ValueError
        If ``common_mode_type`` is not recognised (common-mode path).
    """
    n_det, n_samp = tod.shape
    sigma = rescale_noise(net_ukrts, sampling_rate_hz, scale)

    # --- Cholesky path ---
    if corr_matrix is not None:
        R = np.asarray(corr_matrix, dtype=float)
        if R.shape != (n_det, n_det):
            raise ValueError(
                f"corr_matrix must be a ({n_det}, {n_det}) matrix, "
                f"got shape {R.shape}."
            )
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            raise ValueError(
                "corr_matrix is not positive-definite. "
                "Ensure it is a valid correlation matrix (symmetric, PSD, "
                "unit diagonal). A small diagonal regularisation may help: "
                "R + eps * np.eye(n) where eps ~ 1e-10."
            )
        # Generate n_det independent unit-variance noise streams
        unit_streams = np.zeros((n_det, n_samp))
        for j in range(n_det):
            fknee_j = _get_param_value(fknee_mhz, j)
            fmin_j = _get_param_value(fmin_hz, j)
            alpha_j = _get_param_value(alpha, j)
            unit_streams[j] = _synthesize_one_over_f_stream(
                n_samp, fknee_j, fmin_j, alpha_j, 1.0,
                sampling_rate_hz, dets_random[j], engine=engine, model=model,
            )
        # Mix streams: n_i = sigma_i * sum_j L[i,j] * z_j
        mixed = L @ unit_streams  # shape (n_det, n_samp)
        for i in range(n_det):
            sigma_i = _get_param_value(sigma, i)
            tod[i] += sigma_i * mixed[i]
        return

    # --- Common-mode path ---
    if groups is None:
        raise ValueError(
            "Either 'corr_matrix' or 'groups' must be provided to "
            "add_correlated_noise."
        )
    groups = _validate_grouping(n_det, groups)
    rho_arr = _normalize_rho(rho, n_det)

    # One deterministic RNG per group (for the common-mode streams)
    group_rngs = _spawn_group_rngs_from_detectors(dets_random, groups)

    # Pre-generate common-mode streams (one per unique group)
    common_streams = {}
    for grp, grp_rng in group_rngs.items():
        # Use the NET/spectral parameters of the *first* detector in this group
        idet0 = int(np.where(groups == grp)[0][0])
        sigma_g = _get_param_value(sigma, idet0)
        fknee_g = _get_param_value(fknee_mhz, idet0)
        fmin_g = _get_param_value(fmin_hz, idet0)
        alpha_g = _get_param_value(alpha, idet0)

        if common_mode_type == "one_over_f":
            common_streams[grp] = _synthesize_one_over_f_stream(
                n_samp, fknee_g, fmin_g, alpha_g, sigma_g,
                sampling_rate_hz, grp_rng, engine=engine, model=model,
            )
        elif common_mode_type == "white":
            stream = np.zeros(n_samp)
            add_white_noise(stream, sigma_g, grp_rng)
            common_streams[grp] = stream
        else:
            raise ValueError(
                f"Unknown common_mode_type '{common_mode_type}'. "
                "Choose 'one_over_f' or 'white'."
            )

    # Add common + unique streams to each detector
    for idet in range(n_det):
        rho_i = rho_arr[idet]
        sigma_i = _get_param_value(sigma, idet)
        fknee_i = _get_param_value(fknee_mhz, idet)
        fmin_i = _get_param_value(fmin_hz, idet)
        alpha_i = _get_param_value(alpha, idet)
        grp = groups[idet]

        # Unique stream
        unique_stream = _synthesize_one_over_f_stream(
            n_samp, fknee_i, fmin_i, alpha_i, sigma_i,
            sampling_rate_hz, dets_random[idet], engine=engine, model=model,
        )

        tod[idet] += (
            np.sqrt(rho_i) * common_streams[grp]
            + np.sqrt(1.0 - rho_i) * unique_stream
        )


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
    correlation=None,
):
    """
    Adds noise to a TOD array for all detectors.

    This function handles the correct broadcasting if `net_ukrts`, `fknee_mhz`, etc.,
    are arrays (indicating multiple detectors) while the `tod` is processed
    one detector at a time.

    Parameters
    ----------
    tod : ndarray
        The Time-Ordered Data array of shape (n_detectors, n_samples).
    noise_type : str
        The type of noise to add: ``"white"``, ``"one_over_f"``, or
        ``"correlated"``.
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
    correlation : dict or None, optional
        Required when ``noise_type="correlated"``.  Supported keys:

        * ``"corr_matrix"`` *(ndarray)*: full :math:`(n_{det}, n_{det})`
          correlation matrix.  When present, the Cholesky model is used and
          ``"groups"`` / ``"rho"`` / ``"common_mode_type"`` are ignored.
        * ``"groups"`` *(array-like)*: integer group labels (one per detector).
          Required when ``"corr_matrix"`` is absent.
        * ``"rho"`` *(float or array-like)*: fraction of variance in the
          common mode, in [0, 1].  Defaults to 0.25.
        * ``"common_mode_type"`` *(str)*: PSD shape of the common-mode stream:
          ``"one_over_f"`` (default) or ``"white"``.
    """
    if noise_type == "correlated":
        if correlation is None:
            raise ValueError(
                "noise_type='correlated' requires a 'correlation' dict."
            )
        add_correlated_noise(
            tod=tod,
            sampling_rate_hz=sampling_rate_hz,
            net_ukrts=net_ukrts,
            fknee_mhz=fknee_mhz,
            fmin_hz=fmin_hz,
            alpha=alpha,
            dets_random=dets_random,
            groups=correlation.get("groups"),
            rho=correlation.get("rho", 0.25),
            scale=scale,
            engine=engine,
            model=model,
            common_mode_type=correlation.get("common_mode_type", "one_over_f"),
            corr_matrix=correlation.get("corr_matrix"),
        )
        return

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
        else:
            raise ValueError(f"Unknown noise type '{noise_type}'.")


def add_noise_to_observations(
    observations,
    noise_type,
    user_seed=None,
    dets_random=None,
    scale=1.0,
    component="tod",
    engine="fft",
    model="toast",
    correlation=None,
):
    """
    Adds instrumental noise (white, 1/f, or correlated) to a list of Observations.

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
        * ``"one_over_f"``: 1/f noise characterised by knee frequency and
          spectral index.
        * ``"correlated"``: Common-mode + detector-unique 1/f noise.  Requires
          the ``correlation`` argument.
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
    correlation : dict or None, optional
        Required when ``noise_type="correlated"``.  Supported keys:

        * ``"group_by"`` *(str or None)*: name of a per-detector attribute on
          each :class:`.Observation` (e.g. ``"wafer"``).  Detectors with the
          same value share a common-mode stream.  ``None`` (default) puts all
          detectors in one group.
        * ``"groups"`` *(array-like)*: explicit integer group-label array.
          Takes precedence over ``"group_by"`` when present.
        * ``"corr_matrix"`` *(ndarray)*: full :math:`(n_{det}, n_{det})`
          correlation matrix.  When present, the Cholesky model is used and
          ``"group_by"`` / ``"groups"`` / ``"rho"`` / ``"common_mode_type"``
          are ignored.
        * ``"group_by"`` *(str or None)*: name of a per-detector attribute on
          each :class:`.Observation` (e.g. ``"wafer"``).  Detectors with the
          same value share a common-mode stream.  ``None`` (default) puts all
          detectors in one group.  Used only when ``"corr_matrix"`` is absent.
        * ``"groups"`` *(array-like)*: explicit integer group-label array.
          Takes precedence over ``"group_by"`` when present.
        * ``"rho"`` *(float or array-like)*: fraction of variance in the
          common mode.  Must be in [0, 1]. Defaults to 0.25.
        * ``"common_mode_type"`` *(str)*: PSD shape of the common-mode stream:
          ``"one_over_f"`` (default) or ``"white"``.
    """
    if noise_type not in ["white", "one_over_f", "correlated"]:
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

    if noise_type == "correlated":
        if correlation is None:
            raise ValueError(
                "noise_type='correlated' requires the 'correlation' argument."
            )
        corr_matrix = correlation.get("corr_matrix")
        rho = correlation.get("rho", 0.25)
        common_mode_type = correlation.get("common_mode_type", "one_over_f")
        group_by = correlation.get("group_by", None)

        for cur_obs in obs_list:
            if corr_matrix is not None:
                groups = None
            elif "groups" in correlation:
                groups = _validate_grouping(cur_obs.n_detectors, correlation["groups"])
            else:
                groups = _build_detector_groups(cur_obs, group_by)
            add_correlated_noise(
                tod=getattr(cur_obs, component),
                sampling_rate_hz=cur_obs.sampling_rate_hz,
                net_ukrts=cur_obs.net_ukrts,
                fknee_mhz=getattr(cur_obs, "fknee_mhz"),
                fmin_hz=getattr(cur_obs, "fmin_hz"),
                alpha=getattr(cur_obs, "alpha"),
                dets_random=dets_random,
                groups=groups,
                rho=rho,
                scale=scale,
                engine=engine,
                model=model,
                common_mode_type=common_mode_type,
                corr_matrix=corr_matrix,
            )
        return

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
