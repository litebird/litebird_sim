# -*- encoding: utf-8 -*-

import numpy as np
import astropy.time

from ..scanning import RotQuaternion
from ..quaternions import (
    quat_rotation,
    quat_rotation_brdcast,
    quat_rotation_x,
    quat_rotation_y,
    quat_rotation_z,
    quat_rotation_x_brdcast,
    quat_rotation_y_brdcast,
    quat_rotation_z_brdcast,
    rotate_z_vector,
)
from ..detectors import DetectorInfo, InstrumentInfo
from typing import Union, List, Iterable


def get_detector_orientation(detector: DetectorInfo):
    # TODO: This infomation should be included in IMo.
    """This function returns the orientation of the detector in the focal plane."""

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
        result_matrix (np.ndarray with shape [N,3]): The matrix to store the rotated vectors.

        quat_matrix (np.ndarray with shape [N,4]):   The matrix of quaternions to rotate the vectors.
    """
    result_matrix[:, 0] = 2 * (
        quat_matrix[:, 3] * quat_matrix[:, 1] + quat_matrix[:, 0] * quat_matrix[:, 2]
    )
    result_matrix[:, 1] = 2 * (
        quat_matrix[:, 1] * quat_matrix[:, 2] - quat_matrix[:, 3] * quat_matrix[:, 0]
    )
    result_matrix[:, 2] = 1.0 - 2 * (quat_matrix[:, 0] ** 2 + quat_matrix[:, 1] ** 2)


def left_multiply_offset2det(detector: DetectorInfo, offset_rad: float, axis: str):
    """Add a rotation around the given axis to the quaternion and update the quaternion.

    Args:
        `detector` (DetectorInfo): The instance of `DetectorInfo` type to which `offset_rad` is to be added.
                                   The instance will be destructively updated.

        `offset_rad` (float):      The offset to be added in the specified direction by `axis`, in radians.

        `axis` (str):              The axis in the reference frame around which the rotation is to be performed.
    """
    if axis.lower() == "x":
        rotation_func = quat_rotation_x
    elif axis.lower() == "y":
        rotation_func = quat_rotation_y
    elif axis.lower() == "z":
        rotation_func = quat_rotation_z
    else:
        raise ValueError(f"Invalid axis {axis}, expected 'x', 'y', or 'z")

    orient_rad = get_detector_orientation(detector)
    offset_quat = RotQuaternion(
        quats=np.array(rotation_func(offset_rad)),
    )
    vect = np.empty(3)
    rotate_z_vector(vect, *detector.quat.quats[0])

    orient_quat = RotQuaternion(quats=np.array(quat_rotation(-orient_rad, *vect)))

    interim_quat = offset_quat * orient_quat * detector.quat
    rotate_z_vector(vect, *interim_quat.quats[0])

    orient_quat = RotQuaternion(quats=np.array(quat_rotation(orient_rad, *vect)))
    detector.quat = orient_quat * interim_quat


def left_multiply_disturb2det(
    detector: DetectorInfo,
    start_time,
    sampling_rate_hz,
    noise_rad: np.ndarray,
    axis: str,
):
    """Add a rotation around the given axis to the quaternion and update the quaternion.

    Args:
        `detector` (DetectorInfo):  The instance of `DetectorInfo` type to which
                                    `noise_rad` is to be added. The instance will
                                    be destructively updated.

        `start_time`: Either a floating-point number or an
                     `astropy.time.Time` object. It can be `None` if and
                      only if there is just *one* quaternion in `quats`.

        `sampling_rate_hz`: The sampling frequency of the quaternions, in Hertz.
                            It can be `None` if and only if there is just *one* quaternion in `quats`.

        `noise_rad` (1d-numpy.ndarray): The noise to be added in the specified direction by `axis`,
                                        in radians. It must have shape of 1d NumPy array.

        `axis` (str): The axis in the reference frame around which the rotation is to be performed.
    """
    if axis.lower() == "x":
        rotation_func = quat_rotation_x_brdcast
    elif axis.lower() == "y":
        rotation_func = quat_rotation_y_brdcast
    elif axis.lower() == "z":
        rotation_func = quat_rotation_z_brdcast
    else:
        raise ValueError(f"Invalid axis {axis}, expected 'x', 'y', or 'z")

    orient_rad = get_detector_orientation(detector)
    noise_quats = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=rotation_func(noise_rad),
    )

    vec_matrix = np.empty([len(noise_rad), 3])
    _rotate_z_vectors_brdcast(vec_matrix, detector.quat.quats)

    orient_quat = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=np.array(quat_rotation_brdcast(-orient_rad, vec_matrix)),
    )

    interim_quat = noise_quats * orient_quat * detector.quat
    _rotate_z_vectors_brdcast(vec_matrix, interim_quat.quats)

    orient_quat = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=np.array(quat_rotation_brdcast(orient_rad, vec_matrix)),
    )
    detector.quat = orient_quat * interim_quat


def left_multiply_offset2quat(result: RotQuaternion, offset_rad: float, axis: str):
    """Add a rotation around the given axis to the quaternion and update the quaternion.

    Args:
        `result` (RotQuaternion): Input instance of `RotQuaternion` type to which
                                  `offset_rad` is to be added. The instance will be
                                  destructively updated.

        `offset_rad` (float): The offset to be added in the specified direction
                              by `axis`, in radians.

        `axis` (str): The axis in the reference frame around which the rotation is to be performed.
    """
    if axis.lower() == "x":
        rotation_func = quat_rotation_x
    elif axis.lower() == "y":
        rotation_func = quat_rotation_y
    elif axis.lower() == "z":
        rotation_func = quat_rotation_z
    else:
        raise ValueError(f"Invalid axis {axis}, expected 'x', 'y', or 'z")

    offset_quat = RotQuaternion(
        quats=np.array(rotation_func(offset_rad)),
    )

    _result = offset_quat * result
    result.start_time = _result.start_time
    result.sampling_rate_hz = _result.sampling_rate_hz
    result.quats = _result.quats


def left_multiply_disturb2quat(
    result: RotQuaternion,
    start_time: Union[float, astropy.time.Time],
    sampling_rate_hz: float,
    noise_rad: np.ndarray,
    axis: str,
):
    """Add given noise to the quaternion around specific axis and update the quaternion.

    Args:
        `result` (RotQuaternion): Input instance of `RotQuaternion` type to which noise is to be added.
                                  The instance will be destructively updated.

        `start_time`:       Either a floating-point number or an `astropy.time.Time` object.
                            It can be `None` if and only
                            if there is just *one* quaternion in `quats`.

        `sampling_rate_hz`: The sampling frequency of the quaternions, in Hertz. It can be `None` if and only
                            if there is just *one* quaternion in `quats`.

        `noise_rad` (1d-numpy.ndarray): The noise to be added in the specified direction by `axis`,
                                        in radians. It must have shape of 1d NumPy array.

        `axis` (str): The axis in the reference frame around which the rotation is to be performed.
    """

    if axis.lower() == "x":
        rotation_func = quat_rotation_x_brdcast
    elif axis.lower() == "y":
        rotation_func = quat_rotation_y_brdcast
    elif axis.lower() == "z":
        rotation_func = quat_rotation_z_brdcast
    else:
        raise ValueError(f"Invalid axis {axis}, expected 'x', 'y', or 'z")

    noise_quats = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=rotation_func(noise_rad),
    )

    _result = noise_quats * result
    result.start_time = _result.start_time
    result.sampling_rate_hz = _result.sampling_rate_hz
    result.quats = _result.quats


def _ecl2focalplane(angle, axis):
    """Convert the axis and offset from the ecliptic coordinate to the focal plane.

    Args:
        angle (float or array): The angle which is to be converted.

        axis (str): The axis which is to be converted.
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
        angle (float or array): The angle which is to be converted.

        axis (str): The axis which is to be converted.
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


class FocalplaneCoord:
    """This class create an instans of focal plane to add offset and disturbance to the detectors.

    Args:
        detectors: List of `Detectorinfo` to which offset and disturbance are to be added.
    """

    def __init__(self, detectors: List[DetectorInfo]):
        self.detectors = detectors

    def add_offset(self, offset_rad, axis: str):
        """Add a rotational offset to the detectors in the focal plane by specified axis.

        If the `offset_rad` is a scalar, it will be added to all the detectors in the focal plane.
        If the `offset_rad` is an array with same length of the list of detectors,
        it will be added to the detectors in the focal plane one by one.
        In this case, the length of the array must be equal to the number of detectors.

        Args:
            offset_rad (in case of `float`): The same offset to be added to all detectors on the focal plane
                                             in the specified direction by `axis`, in radians.
                       (in case of `array`): The offset to be added to all dtectors on the focal plane
                                             individually, in the specified direction by `axis`, in radians.

            axis (str): The axis in the reference frame around which the rotation is to be performed.
        """
        offset_rad, axis = _ecl2focalplane(offset_rad, axis)

        if isinstance(offset_rad, Iterable):
            # Detector by detecgtor
            assert len(offset_rad) == len(
                self.detectors
            ), "The length of the offset_rad must be equal to the number of detectors."
            for i, det in enumerate(self.detectors):
                left_multiply_offset2det(det, offset_rad[i], axis)
        else:
            # Global in the focal plane
            for det in self.detectors:
                left_multiply_offset2det(det, offset_rad, axis)

    def add_disturb(
        self, start_time, sampling_rate_hz, noise_rad_matrix: np.ndarray, axis: str
    ):
        """Add a rotational disturbance to the detectors in the focal plane by specified axis.

        If the `noise_rad_matrix` has the shape [N,t] where N is the number of detectors,
        t is the number of timestamps, the disturbance will be added to the detectors
        in the focal plane detector by detector. This represents the independent case
        where each detector has its own disturbance.

        Args:
            start_time: It is either a floating-point number or an `astropy.time.Time` object.
            It can be `None` if and only if there is just *one* quaternion in `quats`.

            sampling_rate_hz: The sampling frequency of the quaternions, in Hertz.
            It can be `None` if and only if there is just *one* quaternion in `quats`.

            noise_rad_matrix
                (numpy.ndarray with shape [N,t]): The disturbance to be added to all detectors on the focal plane
                                                  individually, in the specified direction by `axis`, in radians.
                (numpy.ndarray with shape [,t]):  The common-mode disturbance to be added to all detectors on the
                                                  focal plane in the specified direction by `axis`, in radians.

            axis (str): The axis in the reference frame around which the rotation is to be performed.
        """
        noise_rad_matrix, axis = _ecl2focalplane(noise_rad_matrix, axis)

        if noise_rad_matrix.ndim == 1:
            # Global in the focal plane
            for det in self.detectors:
                left_multiply_disturb2det(
                    det, start_time, sampling_rate_hz, noise_rad_matrix, axis
                )
        else:
            # Detector by detecgtor
            assert (
                noise_rad_matrix.shape[0] == len(self.detectors)
            ), "The number of detectors must be equal to the number of rows in noise_rad_matrix."
            for i, det in enumerate(self.detectors):
                left_multiply_disturb2det(
                    det, start_time, sampling_rate_hz, noise_rad_matrix[i], axis
                )


class SpacecraftCoord:
    """This class create an instans of spacecraft to add offset and disturbance to the instrument.

    Args:
        instrument: `Instrumentinfo` to which offset and disturbance are to be added.
    """

    def __init__(self, instrument: InstrumentInfo):
        self.instrument = instrument

    def add_offset(self, offset_rad, axis: str):
        """Add a rotational offset to the instrument in the spacecraft by specified axis.

        Args:
            offset_rad (float): The offset to be added in the specified direction by `axis`, in radians.

            axis (str): The axis in the reference frame around which the rotation is to be performed.
        """
        offset_rad, axis = _ecl2spacecraft(offset_rad, axis)
        left_multiply_offset2quat(self.instrument.bore2spin_quat, offset_rad, axis)

    def add_disturb(self, start_time, sampling_rate_hz, noise_rad: np.ndarray, axis):
        """Add a rotational disturbance to the instrument in the spacecraft by specified axis.

        Args:
            start_time: It is either a floating-point number or an `astropy.time.Time` object.
            It can be `None` if and only if there is just *one* quaternion in `quats`.

            sampling_rate_hz: The sampling frequency of the quaternions, in Hertz.
            It can be `None` if and only if there is just *one* quaternion in `quats`.

            noise_rad (1d-numpy.ndarray): The disturbance to be added in the specified
                                          direction by `axis`, in radians.

            axis (str): The axis in the reference frame around which the rotation is to be performed.
        """
        noise_rad, axis = _ecl2spacecraft(noise_rad, axis)
        left_multiply_disturb2quat(
            self.instrument.bore2spin_quat,
            start_time,
            sampling_rate_hz,
            noise_rad,
            axis,
        )


class PointingSys:
    """This class provide an interface to add offset and disturbance to the instrument and detectors.

    Args: instrument (InstrumentInfo): The instance of `InstrumentInfo` to which offset and disturbance are to be added.

          detectors (List[DetectorInfo]): List of `Detectorinfo` to which offset and disturbance are to be added.
    """

    def __init__(self, instrument: InstrumentInfo, detectors: List[DetectorInfo]):
        self.spacecraft = SpacecraftCoord(instrument)
        self.focalplane = FocalplaneCoord(detectors)
