# -*- encoding: utf-8 -*-

import astropy.time as astrotime
import numpy as np

from .scanning import (
    ScanningStrategy,
    all_compute_pointing_and_polangle,
    calculate_sun_earth_angles_rad,
)
from ducc0.pointingprovider import PointingProvider


class Observation:
    """An observation made by a detector over some time window

    This class encodes the data acquired by a detector over a finite
    amount of time, and it is the fundamental block of a
    simulation. The characteristics of the detector are assumed to be
    stationary over the time span covered by a simulation; these include:

    - Noise parameters
    - Gain

    To access the TOD, use one of the following methods:

    - :py:meth:`.get_times` returns the array of time values (one per
      each sample in the TOD)
    - :py:meth:`.get_tod` returns the array of samples

    Args:
        detector (str): Name of the detector

        start_time: Start time of the observation. It can either be a
            `astropy.time.Time` type or a floating-point number. In
            the latter case, it must be expressed in seconds.

        sampling_rate_hz (float): The sampling frequency, in Hertz.

        nsamples (int): The number of samples in this observation.

    """

    def __init__(self, detector, start_time, sampling_rate_hz, nsamples):
        self.detector = detector

        self.start_time = start_time

        self.sampling_rate_hz = sampling_rate_hz
        self.nsamples = int(nsamples)

        self.pointing_freq_hz = None
        self.bore2ecliptic_quats = None

        self.tod = None

    def get_times(self, normalize=False, astropy_times=False):
        """Return a vector containing the time of each sample in the observation

        The measure unit of the result depends on the value of
        `astropy_times`: if it's true, times are returned as
        `astropy.time.Time` objects, which can be converted to several
        units (MJD, seconds, etc.); if `astropy_times` is false (the
        default), times are expressed in seconds.

        If `normalize=True`, then the first time is zero. Setting
        this flag requires that `astropy_times=False`.

        This can be a costly operation, particularly if
        `astropy_times=True`; you should cache this result if you plan
        to use it in your code, instead of calling this method over
        and over again.

        See also :py:meth:`get_tod`.

        """
        if normalize:
            assert (
                not astropy_times
            ), "you cannot pass astropy_times=True *and* normalize=True"

            return np.arange(self.nsamples) / self.sampling_rate_hz

        if astropy_times:
            assert isinstance(self.start_time, astrotime.Time), (
                "to use astropy_times=True you must specify an astropy.time.Time "
                "object in Observation.__init__"
            )
            delta = astrotime.TimeDelta(
                1.0 / self.sampling_rate_hz, format="sec", scale="tdb"
            )
            return self.start_time + np.arange(self.nsamples) * delta
        else:
            if isinstance(self.start_time, astrotime.Time):
                # We use "cxcsec" because of the following features:
                #
                # 1. It's one of the astropy.time.Time formats that
                #    measures time in seconds (alongside with "unix"
                #    and "gps")
                #
                # 2. Of the three choices, "cxcsec" uses the most
                #    recent date as reference (1998-01-01, vs.
                #    1990-01-01 for "gps" and 1980-01-06 for "unix").
                t0 = self.start_time.cxcsec
            else:
                t0 = self.start_time

            return t0 + np.arange(self.nsamples) / self.sampling_rate_hz

    def get_tod(self):
        """Return the array of samples measured during this observation

        If no values have been recorded yet, the function returns ``None``.

        Any modification to the array returned by this function will
        apply to the array kept in the `Observation` object, so that
        you can change the samples in the observation easily::

            tod = obs.get_tod()
            # Modify the TOD
            tod[:] *= 10.0


        See also :py:meth:`get_times`.

        """
        return self.tod

    def generate_pointing_information(
        self, scanning_strategy: ScanningStrategy, delta_time_s=60.0
    ):
        self.pointing_freq_hz = 1.0 / delta_time_s

        time_span_s = self.nsamples / self.sampling_rate_hz
        num_of_quaternions = int(time_span_s / delta_time_s) + 1
        if delta_time_s * (num_of_quaternions - 1) < time_span_s:
            num_of_quaternions += 1

        self.bore2ecliptic_quats = np.empty((num_of_quaternions, 4))

        if isinstance(self.start_time, astrotime.Time):
            delta_time = astrotime.TimeDelta(
                np.arange(num_of_quaternions) * delta_time_s, format="sec", scale="tdb"
            )
            time_s = delta_time.sec
            time = self.start_time + delta_time
        else:
            time_s = self.start_time + np.arange(num_of_quaternions) * delta_time_s
            time = time_s

        sun_earth_angles_rad = calculate_sun_earth_angles_rad(time)

        scanning_strategy.all_boresight_to_ecliptic(
            result_matrix=self.bore2ecliptic_quats,
            sun_earth_angles_rad=sun_earth_angles_rad,
            time_vector_s=time_s,
        )

    def get_pointings(self, detector_quat):
        assert len(detector_quat) == 4
        assert self.pointing_freq_hz is not None, (
            "no pointing quaternions, did you forgot to call "
            "Observation.generate_pointing_information?"
        )
        assert (
            self.bore2ecliptic_quats.shape[0] > 1
        ), "having only one quaternion is still unsupported"

        pp = PointingProvider(0.0, self.pointing_freq_hz, self.bore2ecliptic_quats)
        det2ecliptic_quats = pp.get_rotated_quaternions(
            0.0, self.sampling_rate_hz, detector_quat, self.nsamples,
        )

        # Compute the pointing direction for each sample
        pointings_and_polangle = np.empty((self.nsamples, 3))
        all_compute_pointing_and_polangle(
            result_matrix=pointings_and_polangle, quat_matrix=det2ecliptic_quats
        )

        return pointings_and_polangle
