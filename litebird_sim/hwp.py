# -*- encoding: utf-8 -*-

import astropy.time
import numpy as np
from numba import njit


class HWP:
    """
    Abstract class that represents a generic HWP

    Being an abstract class, you should never instantiate it. It is used
    to signal the type of parameters to some functions (e.g.,
    :func:`.get_pointings`).

    If you need to use a HWP object, you should better use derived
    classes like :class:`.IdealHWP`.
    """

    def add_hwp_angle(self, pointing_buffer, start_time_s: float, delta_time_s: float):
        """
        Modify pointings so that they include the effect of the HWP

        This method must be redefined in derived classes. The parameter
        ``pointing_buffer`` must be a D×N×3 matrix representing the three angles
        ``(colatitude, longitude, polangle)`` for D detectors and N measurements.
        The function only alters ``polangle`` and returns nothing.

        The parameters `start_time_s` and `delta_time_s` specify the time of the
        first sample in `pointings` and must be floating-point values; this means
        that you should already have converted any AstroPy time to a plain scalar
        before calling this method.
        """
        raise NotImplementedError(
            "You should not use the HWP class in your code, use IdealHWP instead"
        )

    def __str__(self):
        raise NotImplementedError(
            "You should not use the HWP class in your code, use IdealHWP instead"
        )


@njit
def _add_ideal_hwp_angle(
    pointing_buffer, start_time_s, delta_time_s, start_angle_rad, ang_speed_radpsec
):
    detectors, samples, _ = pointing_buffer.shape
    for det_idx in range(detectors):
        for sample_idx in range(samples):
            angle = (
                start_angle_rad
                + (start_time_s + delta_time_s * sample_idx) * 2 * ang_speed_radpsec
            ) % (2 * np.pi)

            pointing_buffer[det_idx, sample_idx, 2] += angle


class IdealHWP(HWP):
    r"""
    A ideal half-wave plate that spins regularly

    This class represents a perfect HWP that spins with constant angular velocity.
    The constructor accepts the angular speed, expressed in rad/sec, and the
    start angle (in radians). The latter should be referred to the first time
    sample in the simulation, i.e., the earliest sample simulated in any of the
    MPI processes used for the simulation.

    Given a polarization angle :math:`\psi`, this class turns it into
    :math:`\psi + \psi_\text{hwp,0} + 2 \omega_\text{hwp} t`, where
    :math:`\psi_\text{hwp,0}` is the start angle specified in the constructor
    and :math:`\omega_\text{hwp}` is the angular speed of the HWP.
    """

    def __init__(self, ang_speed_radpsec: float, start_angle_rad=0.0):
        self.ang_speed_radpsec = ang_speed_radpsec
        self.start_angle_rad = start_angle_rad

    def add_hwp_angle(self, pointing_buffer, start_time_s: float, delta_time_s: float):
        _add_ideal_hwp_angle(
            pointing_buffer=pointing_buffer,
            start_time_s=start_time_s,
            delta_time_s=delta_time_s,
            start_angle_rad=self.start_angle_rad,
            ang_speed_radpsec=self.ang_speed_radpsec,
        )

    def __str__(self):
        return (
            f"Ideal HWP, with rotating speed {self.ang_speed_radpsec} rad/sec "
            f"and θ₀ = {self.start_angle_rad}"
        )
