# -*- encoding: utf-8 -*-

from typing import Iterable, List
import warnings

import astropy.time
import numpy as np

from .detectors import DetectorInfo
from .hwp import _get_ideal_hwp_angle
from .observations import Observation
from .quaternions import (
    quat_rotation_brdcast,
    quat_rotation_x,
    quat_rotation_x_brdcast,
    quat_rotation_y,
    quat_rotation_y_brdcast,
    quat_rotation_z,
    quat_rotation_z_brdcast,
)
from .scanning import RotQuaternion
from .simulations import Simulation


def get_detector_orientation(detector: DetectorInfo):
    # TODO: This function depends on the detector name format.
    # In the IMo-v2, it is confirmed to be worked.
    # We need to think of a robust way to get the orientation.
    """This function returns the orientation of the detector in the focal plane.

    Args:
        detector (:class:`.DetectorInfo`): The detector to get the orientation.
    """

    telescope = detector.name.split("_")[0]
    if telescope == "000" or telescope == "002":  # LFT and HFT
        orient_angle = 0.0
        handiness = ""
        if telescope == "LFT":
            handiness = detector.name.split("_")[3][1]
        if detector.orient == "Q":
            if detector.pol == "T":
                orient_angle = 0.0
            else:
                orient_angle = np.pi / 2
        else:
            if detector.pol == "T":
                orient_angle = np.pi / 4
            else:
                orient_angle = np.pi / 4 + np.pi / 2
        if handiness == "B":
            orient_angle = -orient_angle
        return orient_angle
    else:  # MFT
        orient_angle = np.deg2rad(float(detector.orient))
        handiness = detector.name.split("_")[3][-1]
        if detector.pol == "B":
            orient_angle += np.pi / 2
        if handiness == "B":
            orient_angle = -orient_angle
    return orient_angle


def _rotate_z_vectors_brdcast(result_matrix: np.ndarray, quat_matrix: np.ndarray):
    """Rotate the z vectors using the quaternions `quat_matrix`.
    This function is a broadcast version of `rotate_z_vector` function.

    Args:
        result_matrix (`numpy.ndarray` with shape [`N`,3]): The matrix to store the rotated vectors.

        quat_matrix (`numpy.ndarray` with shape [`N`,4]):   The matrix of quaternions to rotate the vectors.
    """
    result_matrix[:, 0] = 2 * (
        quat_matrix[:, 3] * quat_matrix[:, 1] + quat_matrix[:, 0] * quat_matrix[:, 2]
    )
    result_matrix[:, 1] = 2 * (
        quat_matrix[:, 1] * quat_matrix[:, 2] - quat_matrix[:, 3] * quat_matrix[:, 0]
    )
    result_matrix[:, 2] = 1.0 - 2 * (quat_matrix[:, 0] ** 2 + quat_matrix[:, 1] ** 2)


def _ecl2focalplane(angle, axis):
    """Convert the axis and offset from the ecliptic coordinate to the focal plane.

    Args:
        angle (`float` or `array`): The angle which is to be converted.

        axis (`str`): The axis which is to be converted. It must be one of 'x', 'y', or 'z'.
    """
    if isinstance(angle, list):
        angle = np.array(angle)
    if axis.lower() == "x":
        axis = "y"
    elif axis.lower() == "y":
        axis = "x"
        angle = -angle
    elif axis.lower() == "z":
        axis = "z"
        angle = -angle
    return (angle, axis)


def _ecl2spacecraft(angle, axis):
    """Convert the axis and offset from the ecliptic coordinate to the spacecraft
    (Payload module: PLM) coordinate.

    Args:
        angle (`float or `array`): The angle which is to be converted.

        axis (`str`): The axis which is to be converted. It must be one of 'x', 'y', or 'z'.
    """
    if isinstance(angle, list):
        angle = np.array(angle)
    if axis.lower() == "x":
        axis = "y"
    elif axis.lower() == "y":
        axis = "x"
        angle = -angle
    elif axis.lower() == "z":
        axis = "z"
    return (angle, axis)


def _get_rotator(axis, broadcast=False):
    """Get the rotation function for the given axis.

    Args:
        axis (`str`): The axis in the reference frame around which the rotation is to be performed. It must be one of 'x', 'y', or 'z'.

        broadcast (`bool`): If `True`, the broadcast version of the rotation function is returned.
    """
    if axis.lower() == "x":
        rotation_func = quat_rotation_x
        if broadcast:
            rotation_func = quat_rotation_x_brdcast
    elif axis.lower() == "y":
        rotation_func = quat_rotation_y
        if broadcast:
            rotation_func = quat_rotation_y_brdcast
    elif axis.lower() == "z":
        rotation_func = quat_rotation_z
        if broadcast:
            rotation_func = quat_rotation_z_brdcast
    else:
        raise ValueError("Invalid axis. The axis must be one of 'x', 'y', or 'z'. ")
    return rotation_func


def left_multiply_syst_quats(
    result: RotQuaternion,
    syst_quats: RotQuaternion,
    detector: DetectorInfo,
    start_time,
    sampling_rate_hz,
):
    """Multiply the systematic quaternions `syst_quats` to the `result` quaternions.

    Args:
        result (:class:`.RotQuaternion`): The quaternion to which the rotation is to be added.

        syst_quats (:class:`.RotQuaternion`): The quaternions to be multiplied to the `result`.

        detector (:class:`.DetectorInfo`): The detector to which the rotation is to be added.

        start_time: Either a floating-point number or an `astropy.time.Time` object. It can be `None` if and only if there is just *one* quaternion in `quats`.

        sampling_rate_hz: The sampling frequency of the quaternions, in Hertz. It can be `None` if and only if there is just *one* quaternion in `quats`.
    """
    # get the orientation angle of the detector in the focal plane
    orient_rad = get_detector_orientation(detector)

    # make the pointing direction (vec_matrix: xyz vector) of each detector in the focal plane.
    # It will be used to unify the reference coordinates of each detector.
    vec_matrix = np.empty([syst_quats.quats.shape[0], 3])
    _rotate_z_vectors_brdcast(vec_matrix, result.quats)

    # generate quaternions which rotate the detector's reference frame by
    # `-orient_rad` around its pointing direction.
    _orient_quat = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=np.array(quat_rotation_brdcast(-orient_rad, vec_matrix)),
    )

    # Add the systematic rotation to the detector pointings.
    # The operation: (_orient_quat * result) unifies the reference coordinates of each detector.
    # Without this, we cannot rotate the detectors around the specified axis.
    interim_quat = syst_quats * _orient_quat * result

    # Reuse the vec_matrix to make the pointing direction of each detector in the focal plane.
    _rotate_z_vectors_brdcast(vec_matrix, interim_quat.quats)
    # Generate quaternions to back-rotate the detector's reference frame by `orient_rad`
    # around its pointing direction.
    _orient_quat = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=np.array(quat_rotation_brdcast(orient_rad, vec_matrix)),
    )
    # we back-rotate the detectors to their original reference frame.
    # By this operation, the detectors polarization angle is brought back to the original position.
    _result = _orient_quat * interim_quat

    result.start_time = syst_quats.start_time
    result.sampling_rate_hz = syst_quats.sampling_rate_hz
    result.quats = _result.quats


class FocalplaneCoord:
    """This class create an instance of focal plane to add offset and disturbance to the detectors.

    Methods in this class multiply systematic quaternions to :attr:`.Observation.quat` (List[:class:`.RotQuaternion`]).

    Args:
        sim (`Simulation`): :class:`.Simulation` instance.

        obs (`Observation`): :class:`.Observation` instance whose :attr:`.Observation.quat` is injected with the systematics.

        detectors (`List[DetectorInfo]`): List of :class:`.DetectorInfo` instances.
    """

    def __init__(
        self, sim: Simulation, obs: Observation, detectors: List[DetectorInfo]
    ):
        self.sim = sim
        self.obs = obs
        self.start_time = obs.start_time
        self.sampling_rate_hz = obs.sampling_rate_hz
        self.detectors = detectors

    def add_offset(self, offset_rad, axis: str):
        """Add a rotational offset to the detectors in the focal plane by the specified axis.

        This method multiplies systematic quaternions to :attr:`.Observation.quat` (List[:class:`.RotQuaternion`]).

        If the `offset_rad` is a scalar, it will be added to all the detectors in the focal plane.
        If the `offset_rad` is an array with same length of the list of detectors,
        it will be added to the detectors in the focal plane one by one.
        In this case, the length of the array must be equal to the number of detectors.

        Args:
            offset_rad
                (in case of `float`): The same offset to be added to all detectors on the focal plane in the specified direction by `axis`, in radians.
                (in case of `array`): The offset to be added to all dtectors on the focal plane individually, in the specified direction by `axis`, in radians.

            axis (`str`): The axis in the reference frame around which the rotation is to be performed. It must be one of 'x', 'y', or 'z'.
        """
        offset_rad, axis = _ecl2focalplane(offset_rad, axis)
        rotation_func = _get_rotator(axis)

        if isinstance(offset_rad, Iterable):
            # Detector by detecgtor
            assert len(offset_rad) == len(self.detectors), (
                "The length of the offset_rad must be equal to the number of detectors."
            )
            for idx in self.obs.det_idx:
                syst_quat = RotQuaternion(
                    quats=np.array(rotation_func(offset_rad[idx]))
                )
                left_multiply_syst_quats(
                    self.obs.quat[idx],
                    syst_quat,
                    self.detectors[idx],
                    self.start_time,
                    self.sampling_rate_hz,
                )
        else:
            # Global in the focal plane
            syst_quat = RotQuaternion(quats=np.array(rotation_func(offset_rad)))
            for idx in self.obs.det_idx:
                left_multiply_syst_quats(
                    self.obs.quat[idx],
                    syst_quat,
                    self.detectors[idx],
                    self.start_time,
                    self.sampling_rate_hz,
                )

    def add_disturb(self, noise_rad_matrix: np.ndarray, axis: str):
        """Add a rotational disturbance to the detectors in the focal plane by the specified axis.

        This method multiplies systematic quaternions to :attr:`.Observation.quat` (List[:class:`.RotQuaternion`]).

        If the `noise_rad_matrix` has the shape [`N`, `t`] where `N` is the number of detectors,
        `t` is the number of timestamps, the disturbance will be added to the detectors
        in the focal plane detector by detector. This represents the independent case
        where each detector has its own disturbance.

        Args:
            noise_rad_matrix
                (`numpy.ndarray` with shape [`N`, `t`]): The disturbance to be added to all detectors on the focal plane individually, in the specified direction by `axis`, in radians.

                (`numpy.ndarray` with shape [, `t`]):  The common-mode disturbance to be added to all detectors on the focal plane in the specified direction by `axis`, in radians.

            axis (`str`): The axis in the reference frame around which the rotation is to be performed. It must be one of 'x', 'y', or 'z'.
        """
        noise_rad_matrix, axis = _ecl2focalplane(noise_rad_matrix, axis)
        rotation_func = _get_rotator(axis, broadcast=True)

        if noise_rad_matrix.ndim == 1:
            # Global in the focal plane
            for idx in self.obs.det_idx:
                syst_quat = RotQuaternion(
                    start_time=self.start_time,
                    sampling_rate_hz=self.sampling_rate_hz,
                    quats=np.array(rotation_func(noise_rad_matrix)),
                )
                left_multiply_syst_quats(
                    self.obs.quat[idx],
                    syst_quat,
                    self.detectors[idx],
                    self.start_time,
                    self.sampling_rate_hz,
                )
        else:
            # Detector by detecgtor
            assert noise_rad_matrix.shape[0] == len(self.detectors), (
                "The number of detectors must be equal to the number of rows in noise_rad_matrix."
            )
            for idx in self.obs.det_idx:
                syst_quat = RotQuaternion(
                    start_time=self.start_time,
                    sampling_rate_hz=self.sampling_rate_hz,
                    quats=np.array(rotation_func(noise_rad_matrix[idx])),
                )
                left_multiply_syst_quats(
                    self.obs.quat[idx],
                    syst_quat,
                    self.detectors[idx],
                    self.start_time,
                    self.sampling_rate_hz,
                )


class SpacecraftCoord:
    """This class create an instance of spacecraft to add offset and disturbance to the instrument.

    Methods in this class multiply systematic quaternions to :attr:`.Simulation.spin2ecliptic_quats` (:class:`.RotQuaternion`).

    Args:
        sim (`Simulation`): :class:`.Simulation` instance.

        obs (`Observation`): :class:`.Observation` instance whose :attr:`.Observation.quat` is injected with the systematics.

        detectors (`List[DetectorInfo]`): List of :class:`.DetectorInfo` instances.
    """

    def __init__(
        self, sim: Simulation, obs: Observation, detectors: List[DetectorInfo]
    ):
        self.sim = sim
        self.obs = obs
        self.start_time = obs.start_time
        self.sim.spin2ecliptic_quats.start_time = obs.start_time
        self.sampling_rate_hz = obs.sampling_rate_hz
        self.instrument = sim.instrument
        self.detectors = detectors

    def add_offset(self, offset_rad, axis: str):
        """Add a rotational offset to the instrument in the spacecraft by the specified axis.

        This method multiplies systematic quaternions to :attr:`.Simulation.spin2ecliptic_quats` (:class:`.RotQuaternion`).

        Args:
            offset_rad (`float`): The offset to be added in the specified direction by `axis`, in radians.

            axis (`str`): The axis in the reference frame around which the rotation is to be performed. It must be one of 'x', 'y', or 'z'.
        """
        offset_rad, axis = _ecl2spacecraft(offset_rad, axis)
        rotation_func = _get_rotator(axis)
        syst_quat = RotQuaternion(quats=np.array(rotation_func(offset_rad)))
        self.sim.spin2ecliptic_quats *= syst_quat

    def add_disturb(self, noise_rad: np.ndarray, axis):
        """Add a rotational disturbance to the instrument in the spacecraft by the specified axis.

        This method multiplies systematic quaternions to :attr:`.Simulation.spin2ecliptic_quats` (:class:`.RotQuaternion`).

        Args:
            noise_rad (1d-`numpy.ndarray`): The disturbance to be added in the specified direction by `axis`, in radians.

            axis (`str`): The axis in the reference frame around which the rotation is to be performed. It must be one of 'x', 'y', or 'z'.
        """
        noise_rad, axis = _ecl2spacecraft(noise_rad, axis)
        rotation_func = _get_rotator(axis, broadcast=True)
        syst_quat = RotQuaternion(
            start_time=self.start_time,
            sampling_rate_hz=self.sampling_rate_hz,
            quats=np.array(rotation_func(noise_rad)),
        )
        self.sim.spin2ecliptic_quats *= syst_quat


class HWPCoord:
    """A class to add pointing disturbance due to the spinning HWP.

    Methods in this class multiply systematic quaternions to :attr:`DetectorInfo.quat` (:class:`.RotQuaternion`).

    Args:
        sim (`Simulation`): :class:`.Simulation` instance.

        obs (`Observation`): :class:`.Observation` instance whose :attr:`.Observation.quat` is injected with the systematics.

        detectors (`List[DetectorInfo]`): List of :class:`.DetectorInfo` instances.
    """

    def __init__(
        self, sim: Simulation, obs: Observation, detectors: List[DetectorInfo]
    ):
        """Initialize the HWPCoord class.

        Description of the internal instance variables:
            start_time: The start time of the simulation.

            sampling_rate_hz (`float`): The sampling rate of the detectors.

            detectors (`List[DetectorInfo]`): List of :class:`.DetectorInfo` to which offset and disturbance are to be added.

            ang_speed_radpsec (`float`): The angular speed of the spinning HWP.

            tilt_angle_rad (`float`): The tilted pointing angle from the expected pointing direction.

            tilt_phase_rad (`float`): The phase of the tilted HWP.
        """
        self.sim = sim
        self.obs = obs
        self.start_time = obs.start_time
        self.sampling_rate_hz = obs.sampling_rate_hz
        self.detectors = detectors
        self.ang_speed_radpsec = None
        self.tilt_angle_rad = None
        self.tilt_phase_rad = 0.0

    @staticmethod
    def get_wedgeHWP_pointing_shift_angle(wedge_angle: float, refractive_index: float):
        """Calculate the pointing angle shift due to the wedged HWP.
        It assumes that the HWP has same refractive index between its ordinary
        and extraordinary optic axes.

        Args:
            wedge_angle (`float`): angle of the wedge HWP in radian.

            refractive_index (`float`): refractive index of the HWP.

        Returns:
            `float`, the pointing angle shift due to the wedged HWP.
        """
        return (refractive_index - 1.0) * wedge_angle

    def add_hwp_rot_disturb(
        self,
        tilt_angle_rad: float,
        ang_speed_radpsec: float,
        tilt_phase_rad=0.0,
    ):
        """Add a circular pointing disturbance synchrinized with the HWP to detectors in the focal plane.

        After the systematics injection, the pointings will be rotated around an
        expected pointing direction far from a angular distance of `self.tilt_angle_rad`.
        The pointing rotation frequency is determined by `self.ang_speed_radpsec`.

        Args:
            tilt_angle_rad (`float`): The tilted pointing angle from the expected pointing direction.

            ang_speed_radpsec (`float`): The angular speed of the pointing circular disturbance.

            tilt_phase_rad (`float`): Angle at which the gradient direction of HWP's wedge.
        """
        self.ang_speed_radpsec = ang_speed_radpsec
        self.tilt_angle_rad = tilt_angle_rad
        self.tilt_phase_rad = tilt_phase_rad

        _start_time = self.obs.start_time - self.obs.start_time_global
        _delta_time = self.obs.get_delta_time()
        n_samples = self.sim.spin2ecliptic_quats.quats.shape[0]
        pointing_rot_angles = np.empty(n_samples)

        if isinstance(_start_time, astropy.time.TimeDelta):
            start_time_s = _start_time.to("s").value
            delta_time_s = _delta_time.to("s").value
        else:
            start_time_s = _start_time
            delta_time_s = _delta_time

        _get_ideal_hwp_angle(
            output_buffer=pointing_rot_angles,
            start_time_s=start_time_s,
            delta_time_s=delta_time_s,
            start_angle_rad=self.sim.hwp.start_angle_rad,
            ang_speed_radpsec=self.ang_speed_radpsec,
        )
        # Set initial phase of pointing disturbance
        pointing_rot_angles += self.tilt_phase_rad

        # It decompose a pointing rotation around z-axis in forcal plane reference france to
        # a simple harmonic oscillation around x- and y-axis.
        rotational_quats_x = RotQuaternion(
            start_time=self.start_time,
            sampling_rate_hz=self.sampling_rate_hz,
            quats=quat_rotation_x_brdcast(
                self.tilt_angle_rad * np.cos(pointing_rot_angles)
            ),
        )
        rotational_quats_y = RotQuaternion(
            start_time=self.start_time,
            sampling_rate_hz=self.sampling_rate_hz,
            quats=quat_rotation_y_brdcast(
                self.tilt_angle_rad * np.sin(pointing_rot_angles)
            ),
        )
        # generate quaternions it makes rotational disturbance to pointings around z-axis
        disturb_quats = rotational_quats_x * rotational_quats_y
        # multiply them to obs.quat
        for idx in self.obs.det_idx:
            left_multiply_syst_quats(
                self.obs.quat[idx],
                disturb_quats,
                self.detectors[idx],
                self.start_time,
                self.sampling_rate_hz,
            )


class PointingSys:
    """This class provide an interface to add offset and disturbance to the instrument and detectors.

    Args:
        sim (`Simulation`): :class:`.Simulation` instance whose .spin2ecliptic_quats is injected with the systematics.

        obs (`Observation`): :class:`.Observation` instance whose :attr:`.Observation.quat` is injected with the systematics.

        detectors (`List[Detector]`): List of :class:`.DetectorInfo`.
    """

    def __init__(
        self, sim: Simulation, obs: Observation, detectors: List[DetectorInfo]
    ):
        for detector in detectors:
            assert detector.sampling_rate_hz == detectors[0].sampling_rate_hz, (
                "Not all detectors have the same `.sampling_rate_hz`"
            )

        if isinstance(
            sim.spin2ecliptic_quats.start_time, astropy.time.Time
        ) and isinstance(obs.start_time, astropy.time.Time):
            assert sim.spin2ecliptic_quats.start_time == obs.start_time, (
                "The `Simulation.spin2ecliptic_quats.start_time` and the `Observation.start_time` must be the same `astropy.time.Time`."
            )
        elif isinstance(sim.spin2ecliptic_quats.start_time, np.float64) and isinstance(
            obs.start_time, np.float64
        ):
            assert np.isclose(sim.spin2ecliptic_quats.start_time, obs.start_time), (
                "The `Simulation.spin2ecliptic_quats.start_time` and the `Observation.start_time` must be the same value."
            )
        else:
            if np.isclose(sim.spin2ecliptic_quats.start_time, obs.start_time):
                # call warning
                sim.spin2ecliptic_quats.start_time = obs.start_time
                warnings.warn(
                    "\nThe `Simulation.spin2ecliptic_quats.start_time` and the `Observation.start_time` are not same type, but they are same number. \nSo, the `Simulation.spin2ecliptic_quats.start_time` is updated to the `Observation.start_time`."
                )

        self.focalplane = FocalplaneCoord(sim, obs, detectors)
        self.hwp = HWPCoord(sim, obs, detectors)
        self.spacecraft = SpacecraftCoord(sim, obs, detectors)
