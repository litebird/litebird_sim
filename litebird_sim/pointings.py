# -*- encoding: utf-8 -*-

from typing import Optional, Union

import astropy.time
import numpy as np
import numpy.typing as npt

from .hwp import HWP
from .scanning import (
    all_compute_pointing_and_orientation,
    RotQuaternion,
)


class PointingProvider:
    def __init__(
        self,
        # Note that we require here *boresight*→Ecliptic instead of *spin*→Ecliptic
        bore2ecliptic_quats: RotQuaternion,
        hwp: Optional[HWP] = None,
    ):
        self.bore2ecliptic_quats = bore2ecliptic_quats
        self.hwp = hwp

    def has_hwp(self):
        """Return ``True`` if a HWP has been set.

        If the function returns ``True``, you can access the field `hwp`, which
        is an instance of one of the descendeants of class :class:`.HWP`."""
        return self.hwp is not None

    def get_pointings(
        self,
        detector_quat: RotQuaternion,
        start_time: Union[float, astropy.time.Time],
        start_time_global: Union[float, astropy.time.Time],
        sampling_rate_hz: float,
        nsamples: int,
        pointing_buffer: Optional[npt.NDArray] = None,
        hwp_buffer: Optional[npt.NDArray] = None,
        pointings_dtype=np.float64,
    ) -> Union[npt.NDArray, Optional[npt.NDArray]]:
        """

        :param detector_quat: An instance of the class :class:`.RotQuaternion`
        :param start_time: The time of the first sample for which pointings are needed.
            It can either be a floating-point number or a ``astropy.time.Time`` object.
        :param start_time_global: The time of the first sample in the *simulation*.
            It *must* be of the same type as `start_time`.
        :param sampling_rate_hz: The nominal sampling rate of the pointings
        :param nsamples: The number of pointings to compute for this detector
        :param pointing_buffer: A NumPy array with shape ``(nsamples, 3)`` that will be
            filled with the pointings (θ, φ, ψ) in radians. If ``None``, a new NumPy
            array will be allocated.
        :param hwp_buffer: A NumPy array with shape ``(nsamples,)`` that will be filled
            with the angles of the HWP. If ``None``, a new NumPy array will be allocated,
            unless this :class:`.PointingProvider` object has no HWP associated, i.e.,
            the parameter ``hwp`` to the constructor ``__init__()`` was set to ``None``:
            in this case, no buffer will be allocated.
        :param pointings_dtype: The type to use for the arrays `pointing_buffer` and
            `hwp_buffer`, if they have not been provided. (If `pointing_buffer` and
            `hwp_buffer` are not ``None``, the original datatype will be kept unchanged.)
        :return: A pair containing `(pointing_buffer, hwp_buffer)`.
        """

        assert (np.isscalar(start_time) and np.isscalar(start_time_global)) or (
            isinstance(start_time_global, astropy.time.Time)
            and isinstance(start_time, astropy.time.Time)
        ), (
            "The parameters start_time= and start_time_global= must be of the same "
            "type (either floats or astropy.time.Time objects), but they are "
            "{type1} (start_time) and {type2} (start_time_global)"
        ).format(type1=str(type(start_time)), type2=str(type(start_time_global)))

        full_quaternions = (self.bore2ecliptic_quats * detector_quat).slerp(
            start_time=start_time,
            sampling_rate_hz=sampling_rate_hz,
            nsamples=nsamples,
        )

        if self.hwp is not None:
            if hwp_buffer is None:
                hwp_buffer = np.empty(nsamples, dtype=pointings_dtype)

            start_time_s = start_time - start_time_global
            if isinstance(start_time_s, astropy.time.TimeDelta):
                start_time_s = start_time_s.to("s").value

            self.hwp.get_hwp_angle(
                output_buffer=hwp_buffer,
                start_time_s=start_time_s,
                delta_time_s=1.0 / sampling_rate_hz,
            )
        else:
            hwp_buffer = None

        if pointing_buffer is None:
            pointing_buffer = np.empty(shape=(nsamples, 3), dtype=pointings_dtype)

        all_compute_pointing_and_orientation(
            result_matrix=pointing_buffer,
            quat_matrix=full_quaternions,
        )

        return pointing_buffer, hwp_buffer

    def get_hwp_angle(
        self,
        start_time: Union[float, astropy.time.Time],
        start_time_global: Union[float, astropy.time.Time],
        sampling_rate_hz: float,
        nsamples: int,
        hwp_buffer: Optional[npt.NDArray] = None,
        pointings_dtype=np.float64,
    ) -> npt.NDArray:
        """

        :param start_time: The time of the first sample for which pointings are needed.
            It can either be a floating-point number or a ``astropy.time.Time`` object.
        :param start_time_global: The time of the first sample in the *simulation*.
            It *must* be of the same type as `start_time`.
        :param sampling_rate_hz: The nominal sampling rate of the pointings
        :param nsamples: The number of pointings to compute for this detector
        :param hwp_buffer: A NumPy array with shape ``(nsamples,)`` that will be filled
            with the angles of the HWP. If ``None``, a new NumPy array will be allocated,
            unless this :class:`.PointingProvider` object has no HWP associated, i.e.,
            the parameter ``hwp`` to the constructor ``__init__()`` was set to ``None``:
            in this case, no buffer will be allocated.
        :param pointings_dtype: The type to use for the arrays `pointing_buffer` and
            `hwp_buffer`, if they have not been provided. (If `pointing_buffer` and
            `hwp_buffer` are not ``None``, the original datatype will be kept unchanged.)
        :return: hwp_buffer
        """

        assert (np.isscalar(start_time) and np.isscalar(start_time_global)) or (
            isinstance(start_time_global, astropy.time.Time)
            and isinstance(start_time, astropy.time.Time)
        ), (
            "The parameters start_time= and start_time_global= must be of the same "
            "type (either floats or astropy.time.Time objects), but they are "
            "{type1} (start_time) and {type2} (start_time_global)"
        ).format(type1=str(type(start_time)), type2=str(type(start_time_global)))

        if self.hwp is not None:
            if hwp_buffer is None:
                hwp_buffer = np.empty(nsamples, dtype=pointings_dtype)

            start_time_s = start_time - start_time_global
            if isinstance(start_time_s, astropy.time.TimeDelta):
                start_time_s = start_time_s.to("s").value

            self.hwp.get_hwp_angle(
                output_buffer=hwp_buffer,
                start_time_s=start_time_s,
                delta_time_s=1.0 / sampling_rate_hz,
            )
        else:
            hwp_buffer = None

        return hwp_buffer
