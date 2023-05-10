import numpy as np

import hashlib
from enum import IntEnum
from typing import Union, List
from dataclasses import dataclass, fields

from astropy import units as u

from .observations import Observation


class GainDriftType(IntEnum):
    LINEAR_GAIN = 0
    THERMAL_GAIN = 1
    SLOW_GAIN = 2


class SamplingDist(IntEnum):
    UNIFORM = 0
    GAUSSIAN = 1


@dataclass
class GainDriftParams:
    # Common parameters
    sigma_drift_K = 1.0
    drift_type: GainDriftType = GainDriftType.LINEAR_GAIN
    sampling_dist: SamplingDist = SamplingDist.UNIFORM

    # Linear gain parameters
    calibration_period = 86400  # in seconds

    # Thermal gain parameters
    focalplane_group: str = "wafer"  # or "pixtype"
    oversample: int = 2
    fknee_drift_mHz: float = 20.0
    alpha_drift: float = 1.0
    sampling_freq_Hz: float = 19.0
    detector_mismatch: float = 1.0
    thermal_fluctuation_amplitude_K: float = 1.0
    focalplane_Tbath_mK: float = 100.0

    # Slow gain parameters
    # To be added


def responsivity_function(dT):
    # Appropriate function to be implemented later
    return dT


def _hash_function(
    input_str: str,
    user_seed: int = 12345,
):
    bytesobj = (input_str + str(user_seed)).encode("utf-8")

    hashobj = hashlib.md5()
    hashobj.update(bytesobj)
    digest = hashobj.digest()

    return int.from_bytes(bytes=digest, byteorder="little")


def _get_psd(
    freq,
    sigma_drift_K=GainDriftParams.sigma_drift_K,
    fknee_drift_mHz=GainDriftParams.fknee_drift_mHz,
    alpha_drift=GainDriftParams.alpha_drift,
):
    return (sigma_drift_K**2) * (fknee_drift_mHz * 1.0e-3 / freq) ** alpha_drift


def _noise_timestream(
    tod_size,
    focalplane_attr,
    drift_params: GainDriftParams = GainDriftParams(),
    user_seed: int = 12345,
):
    fftlen = 2
    while fftlen <= (drift_params.oversample * tod_size):
        fftlen *= 2

    npsd = fftlen // 2 + 1
    norm = drift_params.sampling_freq_Hz * fftlen / 2.0

    freq = np.fft.rfftfreq(fftlen, 1.0 / drift_params.sampling_freq_Hz)
    assert (
        freq.size == npsd
    ), f"The size of frequency array is {freq.size} that is not same as the expected"
    " value {npsd}"

    psd = np.zeros_like(freq)

    # Starting from 1st element to keep the dc term zero
    psd[1:] = _get_psd(
        freq[1:],
        drift_params.sigma_drift_K,
        drift_params.fknee_drift_mHz,
        drift_params.alpha_drift,
    )

    rng = np.random.default_rng(seed=_hash_function(focalplane_attr, user_seed))

    randarr = rng.normal(loc=0.7, scale=0.5, size=fftlen)

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
    det_tod,
    det_name: str,
    drift_params: GainDriftParams = GainDriftParams(),
    focalplane_attr: str = None,
    noise_timestream=None,
    user_seed: int = 12345,
):

    tod_size = len(det_tod)

    rng = np.random.Generator(
        np.random.default_rng(seed=_hash_function(det_name, user_seed))
    )
    if drift_params.sampling_dist == SamplingDist.UNIFORM:
        rand = rng.uniform()
    elif drift_params.sampling_dist == SamplingDist.GAUSSIAN:
        rand = rng.normal(loc=0.7, scale=0.5)

    if drift_params.drift_type == GainDriftType.LINEAR_GAIN:
        gain_arr = 1.0 + rand * drift_params.sigma_drift_K * np.linspace(
            0, 1, drift_params.calibration_period
        )

        div, mod = (
            tod_size // drift_params.calibration_period,
            tod_size % drift_params.calibration_period,
        )

        for i in np.arange(div):
            det_tod[
                i
                * drift_params.calibration_period : (i + 1)
                * drift_params.calibration_period
            ] *= gain_arr

        det_tod[div * drift_params.calibration_period :] *= gain_arr[:mod]

    elif drift_params.drift_type == GainDriftType.THERMAL_GAIN:
        if focalplane_attr is not None and noise_timestream is not None:
            raise ValueError(
                "`focalplane_attr` and `noise_timestream` cannot be used at the same"
                " time. Internally, `focalplane_attr` is hashed, and it is used to"
                " generate the `noise_timestream`."
            )

        if noise_timestream is None:
            noise_timestream = _noise_timestream(
                tod_size=tod_size,
                focalplane_attr=focalplane_attr,
                drift_params=drift_params,
                user_seed=user_seed,
            )

        thermal_factor = drift_params.thermal_fluctuation_amplitude_K
        if drift_params.detector_mismatch != 0:
            thermal_factor *= 1.0 + rand * drift_params.detector_mismatch

        Tdrift = (
            thermal_factor * noise_timestream * 1.0e-3
        )  # Given that `thermal_factor` is in K unit, multiplying 1.e-3 to get
        # `Tdrift` in mK unit
        dT = 1.0 + Tdrift / drift_params.focalplane_Tbath_mK  # dT is scaler (no units)

        det_tod *= responsivity_function(dT)

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
    det_name: str,
    drift_params: GainDriftParams = GainDriftParams(),
    focalplane_attr: str = None,
    user_seed: int = 12345,
):

    tod_size = len(tod[0])

    if drift_params.drift_type == GainDriftType.LINEAR_GAIN:

        for detidx in np.arange(tod.shape[0]):
            apply_gaindrift_for_one_detector(
                det_tod=tod[detidx],
                det_name=det_name,
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

        det_group = np.unique(focalplane_attr)

        noise_timestream = np.zeros((len(det_group), tod_size))

        for detidx, det_elem in enumerate(det_group):
            noise_timestream[detidx][:] = _noise_timestream(
                tod_size=tod_size,
                focalplane_attr=det_elem,
                drift_params=drift_params,
                user_seed=user_seed,
            )

        for detidx in np.arange(tod.shape[0]):
            det_mask = focalplane_attr[detidx] == det_group
            apply_gaindrift_for_one_detector(
                det_tod=tod[detidx],
                det_name=det_name,
                drift_params=drift_params,
                noise_timestream=noise_timestream[det_mask],
                user_seed=user_seed,
            )


def apply_gaindrift_to_observations(
    obs: Union[Observation, List[Observation]],
    drift_params: GainDriftParams = GainDriftParams(),
    user_seed: int = 12345,
    component: str = "tod",
):

    if isinstance(obs, Observation):
        obs_list = [obs]
    elif isinstance(obs, list):
        obs_list = obs
    else:
        raise TypeError(
            "The parameter `obs` must be an `Observation` or a list of `Observation`."
        )

    for cur_obs in obs_list:
        tod = getattr(cur_obs, component)
        det_name = cur_obs.name
        focalplane_attr = getattr(cur_obs, drift_params.focalplane_group)

        apply_gaindrift_to_tod(
            tod=tod,
            det_name=det_name,
            drift_params=drift_params,
            focalplane_attr=focalplane_attr,
            user_seed=user_seed,
        )
