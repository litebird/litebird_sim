import numpy as np
from scipy.ndimage import binary_dilation

from .flags import FLAG_DTYPE, FLAG_TIME_CONSTANT, get_local_flags
from .observations import Observation
from .observation_utilities import for_each_observation

# The length of the impulse response of the filter is the number of samples
# above this fraction of its peak, as in TOAST (toast.fft.convolve)
_IMPULSE_THRESHOLD = 0.02

# ponytail: maximum number of samples used to measure the impulse response;
# longer responses (τ ≳ 1000 / sampling rate) are truncated to this length
_MAX_IMPULSE_SAMPLES = 2**16


def _filter_time_constant(
    tod,
    time_constant_ms: float | np.ndarray,
    sampling_rate_hz: float,
    deconvolve: bool,
):
    """Multiply each detector timeline by the transfer function of a one-pole filter

    The transfer function of a detector with time constant τ is
    H(f) = 1 / (1 + 2πi f τ); with `deconvolve` the timeline is multiplied by
    1 / H(f) instead. The timeline is extended with its mirror image before the
    FFT, so that it is continuous at the edges, and then cut back.
    """

    assert len(tod.shape) == 2
    num_of_dets, num_of_samples = tod.shape

    # A number applies to all the detectors; an array must have one value per detector
    time_constant_ms = np.broadcast_to(
        np.asarray(time_constant_ms, dtype=np.float64), (num_of_dets,)
    )

    # ponytail: the filter is applied to the samples, which is exact only for
    # signals with no power above the Nyquist frequency; to go beyond, simulate
    # at a higher sampling rate and decimate. With τ shorter than the sampling
    # period the filter is a sub-sample delay, whose response decays only as
    # 1/n: the mirror image limits the error at the edges, which still affects
    # the first and last ~100 samples (~5e-3 of a 3 Hz signal in the last 10
    # samples, with τ = 5 ms at 19 Hz). Flagged samples are filtered like the
    # others: the functions on observations flag the edges and the samples
    # around the flagged ones (see _flag_filter_edges)
    freqs_hz = np.fft.rfftfreq(2 * num_of_samples, d=1.0 / sampling_rate_hz)

    for det_idx in range(num_of_dets):
        tau_s = time_constant_ms[det_idx] * 1e-3
        if tau_s == 0.0:
            continue

        transfer = 1.0 + 2j * np.pi * freqs_hz * tau_s
        if not deconvolve:
            transfer = 1.0 / transfer

        extended = np.concatenate((tod[det_idx], tod[det_idx][::-1]))
        filtered = np.fft.irfft(np.fft.rfft(extended) * transfer, n=len(extended))
        tod[det_idx] = filtered[:num_of_samples]


def _impulse_response_length(
    time_constant_ms: np.ndarray,
    sampling_rate_hz: float,
    num_of_samples: int,
    deconvolve: bool,
) -> np.ndarray:
    """Return, for each detector, the length in samples of the impulse response

    The length is the distance between the first and the last sample whose
    absolute value is larger than ``_IMPULSE_THRESHOLD`` times the peak; it is
    zero for τ = 0, when the filter does nothing.
    """
    num_of_impulse_samples = min(num_of_samples, _MAX_IMPULSE_SAMPLES)
    lengths = np.zeros(len(time_constant_ms), dtype=int)
    # The response depends only on τ: compute it once for each value
    for tau_ms in np.unique(time_constant_ms):
        if tau_ms == 0.0:
            continue
        impulse = np.zeros((1, num_of_impulse_samples))
        impulse[0, num_of_impulse_samples // 2] = 1.0
        _filter_time_constant(impulse, tau_ms, sampling_rate_hz, deconvolve)
        response = np.abs(impulse[0])
        above = np.flatnonzero(response > _IMPULSE_THRESHOLD * response.max())
        lengths[time_constant_ms == tau_ms] = above[-1] - above[0] + 1
    return lengths


def _flag_filter_edges(cur_obs: Observation, lengths: np.ndarray, bit: int):
    """Set `bit` in the samples affected by the edges of the filter

    For each detector, the first and last ``lengths[i]`` samples are flagged,
    as well as those closer than ``lengths[i]`` to a sample that was already
    flagged (in ``local_flags`` or ``global_flags``) with a bit other than `bit`.
    """
    flags = get_local_flags(cur_obs)
    other_bits = np.invert(FLAG_DTYPE(bit))
    flagged = (flags & other_bits) != 0
    if cur_obs.global_flags is not None:
        flagged |= (cur_obs.global_flags & other_bits) != 0

    for det_idx, length in enumerate(lengths):
        if length == 0:
            continue
        near = binary_dilation(
            flagged[det_idx], structure=np.ones(2 * length + 1, dtype=bool)
        )
        near[:length] = True
        near[-length:] = True
        flags[det_idx, near] |= bit


def convolve_time_constant(
    tod,
    time_constant_ms: float | np.ndarray,
    sampling_rate_hz: float,
):
    """Convolve some time-ordered data with the time constant of the detectors

    The detectors are modeled as one-pole low-pass filters, with transfer function
    H(f) = 1 / (1 + 2πi f τ): the signal is delayed and its amplitude is reduced
    at frequencies f ≳ 1 / (2π τ).

    The `tod` is a 2D array with shape ``(N_dets, N_samples)`` and is modified in
    place. The time constant `time_constant_ms` is in ms, and it can be either a
    number (the same for all the detectors) or an array with one value per detector.
    """
    _filter_time_constant(
        tod=tod,
        time_constant_ms=time_constant_ms,
        sampling_rate_hz=sampling_rate_hz,
        deconvolve=False,
    )


def deconvolve_time_constant(
    tod,
    time_constant_ms: float | np.ndarray,
    sampling_rate_hz: float,
):
    """Deconvolve the time constant of the detectors from some time-ordered data

    This is the inverse of :func:`.convolve_time_constant`: the timelines are
    multiplied by 1 / H(f) = 1 + 2πi f τ. The parameters are the same.

    Deconvolving with the same time constant used by :func:`.convolve_time_constant`
    gives back the original timelines; using a different one leaves a residual,
    which can be used to simulate an error in the estimate of τ. Noise that was
    not convolved (e.g., readout noise) is amplified at high frequencies by
    the factor ``|1 + 2πi f τ|``.
    """
    _filter_time_constant(
        tod=tod,
        time_constant_ms=time_constant_ms,
        sampling_rate_hz=sampling_rate_hz,
        deconvolve=True,
    )


def _filter_time_constant_in_observations(
    observations: Observation | list[Observation],
    time_constant_ms: float | np.ndarray | None,
    component: str,
    bit: int,
    deconvolve: bool,
):
    # Without τ, it is read from the detectors: check it before modifying any TOD
    required_fields = ["time_constant_ms"] if time_constant_ms is None else None
    for cur_obs, tod, _ in for_each_observation(
        observations, component, required_fields=required_fields
    ):
        if time_constant_ms is None:
            cur_time_constant_ms = cur_obs.time_constant_ms
        elif np.ndim(time_constant_ms) == 0:
            cur_time_constant_ms = time_constant_ms
        else:
            # One value per detector of the whole observation: with MPI, each
            # process takes the values of its own detectors
            cur_time_constant_ms = np.asarray(time_constant_ms)
            if cur_time_constant_ms.shape != (cur_obs.n_detectors_global,):
                raise ValueError(
                    f"time_constant_ms has shape {cur_time_constant_ms.shape}, "
                    f"but it must be a number or have one value for each of the "
                    f"{cur_obs.n_detectors_global} detectors of the observation"
                )
            cur_time_constant_ms = cur_time_constant_ms[cur_obs.det_idx]

        cur_time_constant_ms = np.broadcast_to(
            np.asarray(cur_time_constant_ms, dtype=np.float64), (tod.shape[0],)
        )
        _filter_time_constant(
            tod=tod,
            time_constant_ms=cur_time_constant_ms,
            sampling_rate_hz=cur_obs.sampling_rate_hz,
            deconvolve=deconvolve,
        )

        if bit == 0 or tod.shape[1] == 0:
            continue
        lengths = _impulse_response_length(
            time_constant_ms=cur_time_constant_ms,
            sampling_rate_hz=cur_obs.sampling_rate_hz,
            num_of_samples=tod.shape[1],
            deconvolve=deconvolve,
        )
        if lengths.any():
            _flag_filter_edges(cur_obs, lengths, bit)


def convolve_time_constant_in_observations(
    observations: Observation | list[Observation],
    time_constant_ms: float | np.ndarray | None = None,
    component: str = "tod",
    bit: int = FLAG_TIME_CONSTANT,
):
    """Convolve the TOD of some observations with the time constant of the detectors

    This is a wrapper around :func:`.convolve_time_constant` that applies to the TOD
    stored in `observations`, which can either be one :class:`.Observation` instance
    or a list of observations. By default the TOD is ``Observation.tod``; use
    `component` to filter another one.

    If `time_constant_ms` is ``None``, the time constant of each detector is taken
    from the ``time_constant_ms`` field of :class:`.DetectorInfo`, and a
    ``ValueError`` is raised if it is missing (``None`` or ``NaN``) for some
    detector. Otherwise it can be a number (the same for all the detectors) or an
    array with one value for each detector of the observation, in the order of the
    detectors passed to :meth:`.Simulation.create_observations`: with MPI, every
    process passes the whole array and uses the values of its own detectors.

    Since the convolution is linear, the order of the calls determines which
    components are filtered: e.g., add the sky signal, convolve it, then add the
    noise that is not filtered by the detector (like readout noise).

    The samples where the filter is inaccurate are flagged with `bit` in
    ``Observation.local_flags``: the first and last ``L`` samples of each
    detector, and those closer than ``L`` to samples already flagged with
    other bits (in ``local_flags`` or ``global_flags``), where ``L`` is the
    number of samples in which the impulse response of the filter is above 2%
    of its peak (as in TOAST). Detectors with τ = 0 are not flagged; with
    ``bit=0`` no flag is set.
    """
    _filter_time_constant_in_observations(
        observations=observations,
        time_constant_ms=time_constant_ms,
        component=component,
        bit=bit,
        deconvolve=False,
    )


def deconvolve_time_constant_in_observations(
    observations: Observation | list[Observation],
    time_constant_ms: float | np.ndarray | None = None,
    component: str = "tod",
    bit: int = FLAG_TIME_CONSTANT,
):
    """Deconvolve the time constant of the detectors from the TOD of some observations

    This is the inverse of :func:`.convolve_time_constant_in_observations`, and it
    takes the same parameters, and flags the samples in the same way. To simulate an error in the estimate of the time
    constant, pass to `time_constant_ms` a value different from the one used in
    the convolution.
    """
    _filter_time_constant_in_observations(
        observations=observations,
        time_constant_ms=time_constant_ms,
        component=component,
        bit=bit,
        deconvolve=True,
    )
