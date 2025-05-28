import numpy as np

from enum import IntEnum
from typing import Union, List
from dataclasses import dataclass

from .observations import Observation
from .seeding import regenerate_or_check_detector_generators


class GainDriftType(IntEnum):
    """An enumeration class to specify the type of gain drift injection.

    The gain drift type can be:

    - ``LINEAR_GAIN``: To inject linear gain drift in time with the
      possibility to calibrate the detectors at periodic interval

    - ``THERMAL_GAIN``: To inject a gain drift with :math:`1/f` psd
      mimicking the fluctuations in the focalplane temperature

    """

    LINEAR_GAIN = 0
    THERMAL_GAIN = 1
    # SLOW_GAIN = 2 # Remains to be implemented


class SamplingDist(IntEnum):
    """An enumeration class to specify the distribution for the random
    scaling factor applied on the gain drift. For linear gain drift, it
    specifies the distribution of the slope of the gain drift. In case of
    thermal gain drift, it specifies the distribution of the detector
    mismatch.

    The implemented distributions are:

    - ``UNIFORM``: Uniform distribution. The lower and upper bound of the
      uniform distribution can be specified by the attributes
      :attr:`.GainDriftParams.sampling_uniform_low` and
      :attr:`.GainDriftParams.sampling_uniform_high`.

    - ``GAUSSIAN``: Normal (Gaussian) distribution. The mean and standard
      deviation of the Gaussian distribution can be specified by the
      attributes :attr:`.GainDriftParams.sampling_gaussian_loc` and
      :attr:`.GainDriftParams.sampling_gaussian_scale`.

    """

    UNIFORM = 0
    GAUSSIAN = 1


@dataclass
class GainDriftParams:
    """
    A class to store the gain drift injection parameters.

    The gain drift type can be one of the following:

    - Linear: It simulates gain drift that increases linearly in time. The
      gain factor resets to one periodically with time interval specified by the
      attribute :attr:`.GainDriftParams.calibration_period_sec`.

    - Thermal: It simulates the gain drift as the fluctuation in the
      focalplane temperature. It offers the possibility to inject common mode drift
      to the TODs of detectors belonging to the same group of detectors identified
      by the attribute :attr:`.GainDriftParams.focalplane_group`. This is
      enabled by setting the attribute :attr:`.GainDriftParams.detector_mismatch` to 0.


    The complete list of parameters is provided here:

    - Parameters common for the simulation of all types of gain drifts:

        - ``drift_type`` (:class:`.GainDriftType`):
          Enumeration to determine the type of gain drift to be simulated.
          See :class:`.GainDriftType`.

        - ``sigma_drift`` (`float`): A dimensionless parameter that
          determines the slope of gain drift in case of linear gain drift, and
          amplitude of thermal fluctuation in case of thermal gain drift.

        - ``sampling_dist`` (:class:`.SamplingDist`): Enumeration
          to specify the distribution of the random scaling/mismatch
          factor applied on the gain drift. See :class:`.SamplingDist`.

    - Parameters that are specific to the simulation of linear gain drift:

        - ``calibration_period_sec`` (`int`): This is the time
          period in seconds after which the linear gain drift resets periodically.

    - Parameters that are specific to the simulation of thermal gain drift:

        - ``focalplane_group`` (`str`): Detector attribute to
          group the detectors. It is used to simulate same noise timestream
          for all the detectors belonging to a given group. It can be any of the
          detector attributes like `"wafer"`, `"pixtype"` or `"channel"`.

        - ``oversample`` (`int`): The factor by which to oversample
          thermal noise FFT beyond the TOD size.

        - ``fknee_drift_mHz`` (`float`): :math:`f_{knee}` of the thermal drift
          power spectral density given in mHz.

        - ``alpha_drift`` (`float`): The spectral index of thermal
          drift power spectral density.

        - ``detector_mismatch`` (`float`): The factor that determines
          the degree of mismatch in thermal fluctuation of detectors belonging
          to same focalplane group. A value other than 0 implies no common
          gain. Whereas a value 0 sets the thermal gain to be same for all
          detectors in a focalplane group.

        - ``thermal_fluctuation_amplitude_K`` (`float`): Amplitude of
          thermal gain fluctuation in Kelvin.

        - ``focalplane_Tbath_K`` (`float`): Temperature of the
          focalplane in Kelvin.

    - Parameters for the sampling distributions:

        - ``sampling_uniform_low`` (`float`): Lower boundary of the output for uniform
          distribution.

        - ``sampling_uniform_high`` (`float`): Upper boundary of the output for uniform
          distribution.

        - ``sampling_gaussian_loc`` (`float`): Mean of the Gaussian distribution.

        - ``sampling_gaussian_scale`` (`float`): Standard deviation of the Gaussian
          distribution.

    """

    # Parameters for sampling distribution
    sampling_uniform_low: float = 0.0
    sampling_uniform_high: float = 1.0
    sampling_gaussian_loc: float = 0.7
    sampling_gaussian_scale: float = 0.5

    # Common parameters
    drift_type: GainDriftType = GainDriftType.LINEAR_GAIN
    sigma_drift: float = 1.0e-2
    sampling_dist: SamplingDist = SamplingDist.UNIFORM

    # Linear gain parameters
    calibration_period_sec: int = 86400

    # Thermal gain parameters
    focalplane_group: str = "wafer"
    oversample: int = 2
    fknee_drift_mHz: float = 20.0
    alpha_drift: float = 1.0
    detector_mismatch: float = 1.0
    thermal_fluctuation_amplitude_K: float = 1.0
    focalplane_Tbath_K: float = 0.1

    # Slow gain parameters
    # To be added


def _responsivity_function(dT):
    """A function to specify the response of the detector electronics to the
    temperature"""

    # Appropriate function to be implemented later
    return dT


def _get_psd(
    freq: np.ndarray,
    sigma_drift: float = GainDriftParams.sigma_drift,
    fknee_drift_mHz: float = GainDriftParams.fknee_drift_mHz,
    alpha_drift: float = GainDriftParams.alpha_drift,
) -> np.ndarray:
    """The function to generate the :math:`1/f` noise power spectral density for the
    thermal fluctuation.

    Args:

        freq (np.ndarray): The frequency array

        sigma_drift (float, optional): A dimensionless parameter that
          determines the amplitude of thermal fluctuation. Defaults to
          :attr:`GainDriftParams.sigma_drift`.

        fknee_drift_mHz (float, optional): f_knee of the thermal drift
          power spectral density given in mHz. Defaults to
          :attr:`GainDriftParams.fknee_drift_mHz`.

        alpha_drift (float, optional): The spectral index of thermal drift
          power spectral density. Defaults to :attr:`GainDriftParams.alpha_drift`.

    Returns:

        np.ndarray: :math:`1/f` noise power spectral density
    """

    return (sigma_drift**2) * (fknee_drift_mHz * 1.0e-3 / freq) ** alpha_drift


def _noise_timestream(
    tod_size: int,
    sampling_freq_hz: float,
    drift_params: GainDriftParams = None,
    random: np.random.Generator = None,
) -> np.ndarray:
    """The function to generate the thermal noise time stream with
    :math:`1/f` power spectral density.

    Args:

        tod_size (int): The length of time ordered data array.

        sampling_freq_hz (float): The sampling frequency of the detector in Hz.

        focalplane_attr (str): The name of the focalplane attribute
          corresponding the focalplane group attribute.
          See :attr:`.GainDriftParams.focalplane_group`.

        drift_params (GainDriftParams, optional): The class object for
          gain drift simulation parameters. Defaults to None.

        user_seed (int, optional): A seed provided by the user. Defaults
          to None.

        random (np.random.Generator, optional): A random number generator.
          Defaults to None.

    Returns:

        np.ndarray: Thermal noise time stream with :math:`1/f` PSD.
    """
    assert random is not None, (
        "You should pass a random number generator which implements the `standard_normal` method."
    )
    if drift_params is None:
        drift_params = GainDriftParams()

    fftlen = 2
    while fftlen <= (drift_params.oversample * tod_size):
        fftlen *= 2

    npsd = fftlen // 2 + 1
    norm = sampling_freq_hz * fftlen / 2.0

    freq = np.fft.rfftfreq(fftlen, 1.0 / sampling_freq_hz)
    assert freq.size == npsd, (
        f"The size of frequency array is {freq.size} that is not same as the expected"
    )
    " value {npsd}"

    psd = np.zeros_like(freq)

    # Starting from 1st element to keep the dc term zero
    psd[1:] = _get_psd(
        freq[1:],
        drift_params.sigma_drift,
        drift_params.fknee_drift_mHz,
        drift_params.alpha_drift,
    )

    randarr = random.standard_normal(size=fftlen)

    fnoise_stream = np.zeros(npsd, dtype=np.complex128)
    fnoise_stream[1:-1] = randarr[1 : npsd - 1] + 1j * randarr[-1 : npsd - 1 : -1]
    fnoise_stream[0] = randarr[0] + 1j * 0.0
    fnoise_stream[-1] = randarr[npsd - 1] + 1j * 0.0

    fnoise_stream *= np.sqrt(psd * norm)
    noise_stream = np.fft.irfft(fnoise_stream)

    offset = (fftlen - tod_size) // 2
    noise_avg = np.mean(noise_stream[offset : offset + tod_size])

    return noise_stream[offset : offset + tod_size] - noise_avg


def apply_gaindrift_for_one_detector(
    det_tod: np.ndarray,
    sampling_freq_hz: float,
    drift_params: GainDriftParams = None,
    noise_timestream: np.ndarray = None,
    random: np.random.Generator = None,
):
    """This function applies the gain drift on the TOD corresponding to only one
    detector.

    The linear drift is applied on the TODs in a periodic way with the period size
    specified in seconds by ``drift_params.callibration_period_sec``. This is by
    assuming that the detectors are calibrated for linear gain drift periodically.
    The slope of the linear gain is determined randomly based on the detector name
    and the user-provided seed.

    The thermal gain drift, on the other hand, is based on the fluctuation of the
    focalplane temperature modeled after :math:`1/f` power spectral
    density (PSD). This :math:`1/f`
    PSD is common to all the detectors belonging to the focalplane group identified
    by ``drift_params.focalplane_group``. The function provides an option to introduce a
    mismatch between the individual detectors within a focalplane group with the
    parameter ``drift_params.detector_mismatch``. This mismatch parameter along with a
    random number determines the extent of the mismatch of the thermal fluctuation
    within the focalplane group. Finally the thermal fluctuation is applied to the TODs
    according to the responsivity function of the detectors.

    Args:

        det_tod (np.ndarray): The TOD array corresponding to only one
          detector.

        sampling_freq_hz (float): The sampling frequency of the detector in Hz.

        det_name (str): The name of the detector to which the TOD belongs.
          This name is used with ``user_seed`` to generate hash. This hash is used to
          set random slope in case of linear drift, and randomized detector mismatch
          in case of thermal gain drift.

        drift_params (:class:`.GainDriftParams`, optional): The gain drift
          injection parameters object. Defaults to None.

        focalplane_attr (str, optional): This is the parameter
          corresponding to the ``drift_params.focalplane_group`` attribute.
          For example, if ``drift_params.focalplane_group = 'wafer'``, the
          ``focalplane_attr`` will be the name of the detector wafer. Defaults to None.

        noise_timestream (np.ndarray, optional): The thermal noise time
          stream. Defaults to None.

        user_seed (int, optional): A seed provided by the user. Defaults
          to None.

        random (np.random.Generator, optional): A random number generator.
          Defaults to None.
    """
    assert random is not None, (
        "You should pass a random number generator which implements the `uniform` and `normal` methods."
    )
    if drift_params is None:
        drift_params = GainDriftParams()

    tod_size = len(
        det_tod
    )  # must be equal to sampling_freq_hz * mission_duration_seconds

    if drift_params.sampling_dist == SamplingDist.UNIFORM:
        rand = random.uniform(
            low=drift_params.sampling_uniform_low,
            high=drift_params.sampling_uniform_high,
        )
    elif drift_params.sampling_dist == SamplingDist.GAUSSIAN:
        rand = random.normal(
            loc=drift_params.sampling_gaussian_loc,
            scale=drift_params.sampling_gaussian_scale,
        )

    gain_arr_size = int(sampling_freq_hz * drift_params.calibration_period_sec)
    if drift_params.drift_type == GainDriftType.LINEAR_GAIN:
        gain_arr = 1.0 + rand * drift_params.sigma_drift * np.linspace(
            0, 1, gain_arr_size
        )

        div, mod = (
            tod_size // gain_arr_size,
            tod_size % gain_arr_size,
        )

        for i in np.arange(div):
            det_tod[i * gain_arr_size : (i + 1) * gain_arr_size] *= gain_arr

        det_tod[div * gain_arr_size :] *= gain_arr[:mod]

    elif drift_params.drift_type == GainDriftType.THERMAL_GAIN:
        if noise_timestream is None:
            noise_timestream = _noise_timestream(
                tod_size=tod_size,
                sampling_freq_hz=sampling_freq_hz,
                drift_params=drift_params,
                random=random,
            )

        thermal_factor = drift_params.thermal_fluctuation_amplitude_K
        if drift_params.detector_mismatch != 0:
            thermal_factor *= 1.0 + rand * drift_params.detector_mismatch

        Tdrift = thermal_factor * noise_timestream  # Thermal factor has kelvin unit

        dT = 1.0 + Tdrift / drift_params.focalplane_Tbath_K  # dT is scaler (no units)

        det_tod *= _responsivity_function(dT)

    elif drift_params.drift_type == GainDriftType.SLOW_GAIN:
        # !!! Remains to be implemented
        pass
    else:
        raise ValueError(
            "`drift_params.drift_type` can only be one of GainDriftType.LINEAR_GAIN,"
            " GainDriftType.THERMAL_GAIN or GainDriftType.SLOW_GAIN."
        )


def apply_gaindrift_to_tod(
    tod: np.ndarray,
    sampling_freq_hz: float,
    drift_params: GainDriftParams = None,
    focalplane_attr: Union[List, np.ndarray] = None,
    dets_random: Union[np.random.Generator, None] = None,
):
    """The function to apply the gain drift to all the detectors of a given TOD object.

    This function is a wrapper around :func:`.apply_gaindrift_for_one_detector()`
    that applies the gain drift on each detector TODs of the TOD object. In case of
    thermal gain drift injection, this function computes the thermal noise
    fluctuations at once for all the detectors belonging to the focalplane group
    specified by ``drift_params.focalplane_group`` and passes them to
    :func:`.apply_gaindrift_for_one_detector()` with individual TOD arrays to inject
    thermal gain drift.

    Args:

        tod (np.ndarray): The TOD object consisting TOD arrays for
          multiple detectors.

        sampling_freq_hz (float): The sampling frequency of the detector in Hz.

        det_name (Union[List, np.ndarray]): The list of the name of the
          detectors to which the TOD arrays correspond. The detector names
          are used to generate unique and reproducible random numbers for
          each detector.

        drift_params (:class:`.GainDriftParams`, optional): The gain drift
          injection parameters object. Defaults to None.

        focalplane_attr (Union[List, np.ndarray], optional): This is the
          parameter corresponding to the ``drift_params.focalplane_group``
          attribute. For example, if
          ``drift_params.focalplane_group = 'wafer'``, the
          ``focalplane_attr`` will be the list of the names of detector
          wafer. Defaults to None.

        user_seed (int, optional): A seed provided by the user. Defaults
          to None.

        dets_random : list of np.random.Generator
          List of per-detector random number generators. Must match the number
          of detectors. Typically obtained from an `RNGHierarchy`.
    """

    if drift_params is None:
        drift_params = GainDriftParams()

    tod_size = len(tod[0])

    if drift_params.drift_type == GainDriftType.LINEAR_GAIN:
        for detidx in np.arange(tod.shape[0]):
            apply_gaindrift_for_one_detector(
                det_tod=tod[detidx],
                sampling_freq_hz=sampling_freq_hz,
                drift_params=drift_params,
                noise_timestream=None,
                random=dets_random[detidx],
            )

    elif drift_params.drift_type == GainDriftType.THERMAL_GAIN:
        if focalplane_attr is None:
            raise ValueError(
                "The argument `focalplane_attr` is required to simulate thermal"
                " gaindrift."
            )

        if tod.shape[0] != len(focalplane_attr):
            raise AssertionError(
                "The number of elements in `focalplane_attr` must be same as the"
                " number of detectors included in tod object"
            )

        det_group = np.unique(focalplane_attr)

        noise_timestream = np.zeros((len(det_group), tod_size))

        for detidx, _ in enumerate(det_group):
            noise_timestream[detidx][:] = _noise_timestream(
                tod_size=tod_size,
                sampling_freq_hz=sampling_freq_hz,
                drift_params=drift_params,
                random=dets_random[detidx],
            )

        for detidx in np.arange(tod.shape[0]):
            det_mask = focalplane_attr[detidx] == det_group
            apply_gaindrift_for_one_detector(
                det_tod=tod[detidx],
                sampling_freq_hz=sampling_freq_hz,
                drift_params=drift_params,
                noise_timestream=noise_timestream[det_mask][
                    0
                ],  # array[mask] returns an array of shape (1, len(array)).
                # Therefore [0] indexing is necessary
                random=dets_random[detidx],
            )


def apply_gaindrift_to_observations(
    observations: Union[Observation, List[Observation]],
    drift_params: GainDriftParams = None,
    user_seed: Union[int, None] = None,
    component: str = "tod",
    dets_random: Union[List[np.random.Generator]] = None,
):
    """
    Apply gain drift to one or more observations.

    This function injects gain drift into the time-ordered data (TOD) of one
    or more `Observation` instances. It wraps
    :func:`apply_gaindrift_to_tod`, and ensures proper setup of per-detector
    random number generators using either a user-provided seed or a list of
    pre-initialized RNGs.

    Parameters
    ----------
    observations : Observation or list of Observation
        A single `Observation` instance or a list of them.
    drift_params : GainDriftParams, optional
        Parameters defining the gain drift injection (e.g., linear or thermal).
        If not provided, a default configuration is used.
    user_seed : int, optional
        Base seed to build the RNG hierarchy and generate detector-level RNGs that overwrite any eventual `dets_random`.
        Required if `dets_random` is not provided.
    component : str, optional
        Name of the TOD attribute to modify. Defaults to `"tod"`.
    dets_random : list of np.random.Generator, optional
        List of per-detector random number generators. If not provided, and
        `user_seed` is given, generators are created internally. One of
        `user_seed` or `dets_random` must be provided.

    Raises
    ------
    TypeError
        If `observations` is neither an `Observation` nor a list of them.
    ValueError
        If neither `user_seed` nor `dets_random` is provided.
    AssertionError
        If the number of random generators does not match the number of detectors.
    """

    if drift_params is None:
        drift_params = GainDriftParams()

    if isinstance(observations, Observation):
        obs_list = [observations]
    elif isinstance(observations, list):
        obs_list = observations
    else:
        raise TypeError(
            "The parameter `observations` must be an `Observation` or a list of `Observation`."
        )
    dets_random = regenerate_or_check_detector_generators(
        observations=obs_list,
        user_seed=user_seed,
        dets_random=dets_random,
    )

    for cur_obs in obs_list:
        tod = getattr(cur_obs, component)
        sampling_freq_hz = cur_obs.sampling_rate_hz
        focalplane_attr = getattr(cur_obs, drift_params.focalplane_group)

        apply_gaindrift_to_tod(
            tod=tod,
            sampling_freq_hz=sampling_freq_hz,
            drift_params=drift_params,
            focalplane_attr=focalplane_attr,
            dets_random=dets_random,
        )
