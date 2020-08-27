# -*- encoding: utf-8 -*-

from typing import Union

import astropy.time
import numpy as np

from .scanning import (
    Spin2EclipticQuaternions,
    all_compute_pointing_and_polangle,
)


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

    def __init__(
        self,
        detector,
        start_time: Union[float, astropy.time.Time],
        sampling_rate_hz: float,
        nsamples: int,
    ):
        self.detector = detector

        self.start_time = start_time

        self.sampling_rate_hz = sampling_rate_hz
        self.nsamples = int(nsamples)

        self.tod = None

    def get_times(self, normalize=False, astropy_times=False):
        """Return a vector containing the time of each sample in the observation

        The measure unit of the result depends on the value of
        `astropy_times`: if it's true, times are returned as
        `astropy.time.Time` objects, which can be converted to several
        units (MJD, seconds, etc.); if `astropy_times` is false (the
        default), times are expressed in seconds. In the latter case,
        you should interpret these times as sidereal.

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
            assert isinstance(self.start_time, astropy.time.Time), (
                "to use astropy_times=True you must specify an astropy.time.Time "
                "object in Observation.__init__"
            )
            delta = astropy.time.TimeDelta(
                1.0 / self.sampling_rate_hz, format="sec", scale="tdb"
            )
            return self.start_time + np.arange(self.nsamples) * delta
        else:
            if isinstance(self.start_time, astropy.time.Time):
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

    def get_det2ecl_quaternions(
        self,
        spin2ecliptic_quats: Spin2EclipticQuaternions,
        detector_quat,
        bore2spin_quat,
    ):
        """Return the detector-to-Ecliptic quaternions

        This function returns a ``(N, 4)`` matrix containing the
        quaternions that convert a vector in detector's coordinates
        into the frame of reference of the Ecliptic. The number of
        quaternions is equal to the number of samples hold in this
        observation.

        Given that the z axis in the frame of reference of a detector
        points along the main beam axis, this means that if you use
        these quaternions to rotate the vector `z = [0, 0, 1]`, you
        will end up with the sequence of vectors pointing towards the
        points in the sky (in Ecliptic coordinates) that are observed
        by the detector.

        This is a low-level method; you should usually call the method
        :meth:`.get_pointings`, which wraps this function to compute
        both the pointing direction and the polarization angle.

        See also the method :meth:`.get_ecl2det_quaternions`, which
        mirrors this one.

        """
        return spin2ecliptic_quats.get_detector_quats(
            detector_quat=detector_quat,
            bore2spin_quat=bore2spin_quat,
            time0=self.start_time,
            sampling_rate_hz=self.sampling_rate_hz,
            nsamples=self.nsamples,
        )

    def get_ecl2det_quaternions(
        self,
        spin2ecliptic_quats: Spin2EclipticQuaternions,
        detector_quat,
        bore2spin_quat,
    ):
        """Return the Ecliptic-to-detector quaternions

        This function returns a ``(N, 4)`` matrix containing the ``N``
        quaternions that convert a vector in Ecliptic coordinates into
        the frame of reference of the detector itself. The number of
        quaternions is equal to the number of samples hold in this
        observation.

        This method is useful when you want to simulate how a point
        source is observed by the detector's beam: if you know the
        Ecliptic coordinates of the point sources, you can easily
        derive the location of the source with respect to the
        reference frame of the detector's beam.
        """

        quats = self.get_det2ecl_quaternions(
            spin2ecliptic_quats, detector_quat, bore2spin_quat
        )
        quats[:, 0:3] *= -1  # Apply the quaternion conjugate
        return quats

    def get_pointings(
        self,
        spin2ecliptic_quats: Spin2EclipticQuaternions,
        detector_quat,
        bore2spin_quat,
    ):
        """Return the time stream of pointings for the detector

        Given a :class:`Spin2EclipticQuaternions` and a quaternion
        representing the transformation from the reference frame of a
        detector to the boresight reference frame, compute a set of
        pointings for the detector that encompass the time span
        covered by this observation (i.e., starting from
        `self.start_time` and including `self.nsamples` pointings).

        The parameter `spin2ecliptic_quats` can be easily retrieved by
        the field `spin2ecliptic_quats` in a object of
        :class:`.Simulation` object, once the method
        :meth:`.Simulation.generate_spin2ecl_quaternions` is called.

        The parameter `detector_quat` is typically one of the
        following:

        - The field `quat` of an instance of the class
           :class:`.Detector`

        - If all you want to do is a simulation using a boresight
           direction, you can pass the value ``np.array([0., 0., 0.,
           1.])``, which represents the null rotation.

        The parameter `bore2spin_quat` is calculated through the class
        :class:`.Instrument`, which has the field ``bore2spin_quat``.
        If all you have is the angle β between the boresight and the
        spin axis, just pass ``quat_rotation_y(β)`` here.

        The return value is a ``(N × 3)`` matrix: the colatitude (in
        radians) is stored in column 0 (e.g., ``result[:, 0]``), the
        longitude (ditto) in column 1, and the polarization angle
        (ditto) in column 2. You can extract the three vectors using
        the following idiom::

            pointings = obs.get_pointings(...)
            # Extract the colatitude (theta), longitude (psi), and
            # polarization angle (psi) from pointings
            theta, phi, psi = [pointings[:, i] for i in (0, 1, 2)]

        """
        det2ecliptic_quats = self.get_det2ecl_quaternions(
            spin2ecliptic_quats, detector_quat, bore2spin_quat,
        )

        # Compute the pointing direction for each sample
        pointings_and_polangle = np.empty((self.nsamples, 3))
        all_compute_pointing_and_polangle(
            result_matrix=pointings_and_polangle, quat_matrix=det2ecliptic_quats
        )

        return pointings_and_polangle
