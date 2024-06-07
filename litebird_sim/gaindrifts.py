import numpy as np

import hashlib
from enum import IntEnum
from typing import Union, List
from dataclasses import dataclass

from .observations import Observation


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


def _hash_function(
    input_str: str,
    user_seed: int = 12345,
) -> int:
    """This functions generates a unique and reproducible hash for a given pair of
    `input_str` and `user_seed`. This hash is used to generate the common noise time
    stream for a group of detectors, and to introduce randomness in the noise time
    streams.

    Args:

        input_str (str): A string, for example, the detector name.

        user_seed (int, optional): A seed provided by the user. Defaults to 12345.

    Returns:

        int: An `md5` hash from generated from `input_str` and `user_seed`
    """

    bytesobj = (str(input_str) + str(user_seed)).encode("utf-8")

    hashobj = hashlib.md5()
    hashobj.update(bytesobj)
    digest = hashobj.digest()

    return int.from_bytes(bytes=digest, byteorder="little")


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
    focalplane_attr: str,
    drift_params: GainDriftParams = None,
    user_seed: int = 12345,
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

        user_seed (int, optional): The user provided seed for random number
          generation. Defaults to 12345.

    Returns:

        np.ndarray: Thermal noise time stream with :math:`1/f` PSD.
    """

    if drift_params is None:
        drift_params = GainDriftParams()

    fftlen = 2
    while fftlen <= (drift_params.oversample * tod_size):
        fftlen *= 2

    npsd = fftlen // 2 + 1
    norm = sampling_freq_hz * fftlen / 2.0

    freq = np.fft.rfftfreq(fftlen, 1.0 / sampling_freq_hz)
    assert (
        freq.size == npsd
    ), f"The size of frequency array is {freq.size} that is not same as the expected"
    " value {npsd}"

    psd = np.zeros_like(freq)

    # Starting from 1st element to keep the dc term zero
    psd[1:] = _get_psd(
        freq[1:],
        drift_params.sigma_drift,
        drift_params.fknee_drift_mHz,
        drift_params.alpha_drift,
    )

    rng = np.random.default_rng(seed=_hash_function(focalplane_attr, user_seed))

    randarr = rng.standard_normal(size=fftlen)

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
    det_name: str,
    drift_params: GainDriftParams = None,
    focalplane_attr: str = None,
    noise_timestream: np.ndarray = None,
    user_seed: int = 12345,
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
          to 12345.
    """

    if drift_params is None:
        drift_params = GainDriftParams()

    tod_size = len(
        det_tod
    )  # must be equal to sampling_freq_hz * mission_duration_seconds

    assert isinstance(det_name, str), "The parameter `det_name` must be a string"
    rng = np.random.default_rng(seed=_hash_function(det_name, user_seed))

    if drift_params.sampling_dist == SamplingDist.UNIFORM:
        rand = rng.uniform(
            low=drift_params.sampling_uniform_low,
            high=drift_params.sampling_uniform_high,
        )
    elif drift_params.sampling_dist == SamplingDist.GAUSSIAN:
        rand = rng.normal(
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
        if focalplane_attr is not None and noise_timestream is not None:
            raise ValueError(
                "`focalplane_attr` and `noise_timestream` cannot be used at the same"
                " time. Internally, `focalplane_attr` is hashed, and it is used to"
                " generate the `noise_timestream`."
            )

        if noise_timestream is None:
            assert isinstance(
                focalplane_attr, str
            ), "The parameter `focalplane_attr` must be a string"

            noise_timestream = _noise_timestream(
                tod_size=tod_size,
                sampling_freq_hz=sampling_freq_hz,
                focalplane_attr=focalplane_attr,
                drift_params=drift_params,
                user_seed=user_seed,
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
    det_name: Union[List, np.ndarray],
    drift_params: GainDriftParams = None,
    focalplane_attr: Union[List, np.ndarray] = None,
    user_seed: int = 12345,
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
          to 12345.
    """

    if drift_params is None:
        drift_params = GainDriftParams()

    if tod.shape[0] != len(det_name):
        raise AssertionError(
            "The number of elements in `det_name` must be same as the number of"
            " detectors included in tod object"
        )

    tod_size = len(tod[0])

    if drift_params.drift_type == GainDriftType.LINEAR_GAIN:
        for detidx in np.arange(tod.shape[0]):
            apply_gaindrift_for_one_detector(
                det_tod=tod[detidx],
                sampling_freq_hz=sampling_freq_hz,
                det_name=det_name[detidx],
                drift_params=drift_params,
                noise_timestream=None,
                user_seed=user_seed,
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

        for detidx, det_elem in enumerate(det_group):
            noise_timestream[detidx][:] = _noise_timestream(
                tod_size=tod_size,
                sampling_freq_hz=sampling_freq_hz,
                focalplane_attr=det_elem,
                drift_params=drift_params,
                user_seed=user_seed,
            )

        for detidx in np.arange(tod.shape[0]):
            det_mask = focalplane_attr[detidx] == det_group
            apply_gaindrift_for_one_detector(
                det_tod=tod[detidx],
                sampling_freq_hz=sampling_freq_hz,
                det_name=det_name[detidx],
                drift_params=drift_params,
                noise_timestream=noise_timestream[det_mask][
                    0
                ],  # array[mask] returns an array of shape (1, len(array)).
                # Therefore [0] indexing is necessary
                user_seed=user_seed,
            )


def apply_gaindrift_to_observations(
    observations: Union[Observation, List[Observation]],
    drift_params: GainDriftParams = None,
    user_seed: int = 12345,
    component: str = "tod",
):
    """The function to apply gain drift to the TOD of a :class:`.Observation`
    instance or a list of observations.

    This function is a wrapper around :func:`.apply_gaindrift_to_tod()`
    that injects gain drift to the TOD object.

    Args:

        observations (Union[Observation, List[Observation]]): An instance or a list
          of instances of :class:`.Observation`.

        drift_params (:class:`.GainDriftParams`, optional): The gain drift
          injection parameters object. Defaults to None.

        user_seed (int, optional): A seed provided by the user. Defaults
          to 12345.

        component (str, optional): The name of the TOD on which the gain
          drift has to be injected. Defaults to "tod".
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

    for cur_obs in obs_list:
        tod = getattr(cur_obs, component)
        det_name = cur_obs.name
        sampling_freq_hz = cur_obs.sampling_rate_hz
        focalplane_attr = getattr(cur_obs, drift_params.focalplane_group)

        apply_gaindrift_to_tod(
            tod=tod,
            sampling_freq_hz=sampling_freq_hz,
            det_name=det_name,
            drift_params=drift_params,
            focalplane_attr=focalplane_attr,
            user_seed=user_seed,
        )
