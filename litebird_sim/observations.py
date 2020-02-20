# -*- encoding: utf-8 -*-

import astropy.time as astrotime
import numpy as np


class Observation:
    """An observation made by a detector over some time window

    This class encodes the data acquired by a detector over a finite
    amount of time, and it is the fundamental block of a
    simulation. The characteristics of the detector are assumed to be
    stationary over the time span covered by a simulation; these include:

    - Noise parameters
    - Gain

    To access the TOD, use one of the following methods:

    - :py:meth:`get_times` returns the array of time values (one per
      each sample in the TOD)
    - :py:meth:`get_tod` returns the array of samples

    :param detector str: Name of the detector

    :param start_time: Start time of the observation. It can either be
    a `astropy.time.Time` type or a floating-point number. In the
    latter case, if `use_mjd` is ``False``, the number must be
    expressed in seconds; otherwise, it must be a MJD.

    :param sampfreq: The sampling frequency. Regardless of the
    measurement unit used for `start_time`, this *must* be expressed in Hertz.

    :param nsamples int: The number of samples in this observation.

    :param use_mjd bool: If ``True``, the value of `start_time` is
    expressed in MJD.

    """

    def __init__(self, detector, start_time, sampfreq_hz, nsamples, use_mjd=False):
        self.detector = detector
        self.use_mjd = use_mjd

        if isinstance(start_time, astrotime.Time):
            if self.use_mjd:
                self.start_time = start_time.mjd
            else:
                # We use "cxcsec" because of the following features:
                #
                # 1. It's one of the astropy.time.Time formats that
                #    measures time in seconds (alongside with "unix"
                #    and "gps")
                #
                # 2. Of the three choices, "cxcsec" uses a recent date
                #    as reference (1998-01-01, vs. 1990-01-01 for
                #    "gps" and 1980-01-06 for "unix").
                self.start_time = start_time.cxcsec
        else:
            self.start_time = start_time

        self.sampfreq_hz = sampfreq_hz
        self.nsamples = nsamples
        self.tod = None

    def get_times(self):
        """Return a vector containing the time of each sample in the observation

        The measure unit of the result depends whether
        ``self.use_mjd`` is true (return MJD) or false (return
        seconds). The number of elements in the vector is equal to
        ``self.nsamples``.

        See also :py:meth:`get_tod`.

        """
        if self.use_mjd:
            delta = astrotime.TimeDelta(1.0 / self.sampfreq_hz, format="sec")
            vec = (
                astrotime.Time(self.start_time, format="mjd")
                + np.arange(self.nsamples) * delta
            )
            return vec.mjd
        else:
            return self.start_time + np.arange(self.nsamples) / self.sampfreq_hz

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
