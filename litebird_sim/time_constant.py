import numpy as np

from .observations import Observation
from .observation_utilities import for_each_observation


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
    # samples, with τ = 5 ms at 19 Hz). Flags are ignored, so the same happens
    # around gaps; mask them with the flags if needed
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
    deconvolve: bool,
):
    for cur_obs, tod, _ in for_each_observation(observations, component):
        if time_constant_ms is None:
            cur_time_constant_ms = cur_obs.time_constant_ms
        else:
            cur_time_constant_ms = time_constant_ms

        _filter_time_constant(
            tod=tod,
            time_constant_ms=cur_time_constant_ms,
            sampling_rate_hz=cur_obs.sampling_rate_hz,
            deconvolve=deconvolve,
        )


def convolve_time_constant_in_observations(
    observations: Observation | list[Observation],
    time_constant_ms: float | np.ndarray | None = None,
    component: str = "tod",
):
    """Convolve the TOD of some observations with the time constant of the detectors

    This is a wrapper around :func:`.convolve_time_constant` that applies to the TOD
    stored in `observations`, which can either be one :class:`.Observation` instance
    or a list of observations. By default the TOD is ``Observation.tod``; use
    `component` to filter another one.

    If `time_constant_ms` is ``None``, the time constant of each detector is taken
    from the ``time_constant_ms`` field of :class:`.DetectorInfo`. Otherwise it can
    be a number (the same for all the detectors) or an array with one value for each
    detector of the observations.

    Since the convolution is linear, the order of the calls determines which
    components are filtered: e.g., add the sky signal, convolve it, then add the
    noise that is not filtered by the detector (like readout noise).
    """
    _filter_time_constant_in_observations(
        observations=observations,
        time_constant_ms=time_constant_ms,
        component=component,
        deconvolve=False,
    )


def deconvolve_time_constant_in_observations(
    observations: Observation | list[Observation],
    time_constant_ms: float | np.ndarray | None = None,
    component: str = "tod",
):
    """Deconvolve the time constant of the detectors from the TOD of some observations

    This is the inverse of :func:`.convolve_time_constant_in_observations`, and it
    takes the same parameters. To simulate an error in the estimate of the time
    constant, pass to `time_constant_ms` a value different from the one used in
    the convolution.
    """
    _filter_time_constant_in_observations(
        observations=observations,
        time_constant_ms=time_constant_ms,
        component=component,
        deconvolve=True,
    )
