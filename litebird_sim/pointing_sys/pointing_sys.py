# -*- encoding: utf-8 -*-

from typing import Union

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
from ..simulations import Simulation
from typing import Union, List, Optional, Iterable


def get_detector_orientation(detector:DetectorInfo):
    """This function returns the orientation of the detector in the focal plane."""
    if detector.wafer[0]:
        telescope = detector.wafer[0] + 'FT'
        if telescope == 'LFT' or telescope == 'HFT':
            orient_angle = 0.
            handiness = ""
            if telescope == 'LFT':
                handiness = detector.name.split('_')[3][1]
            if detector.orient == 'Q':
                if detector.pol == 'T':
                    orient_angle = 0.
                else:
                    orient_angle = np.pi/2
            else:
                if detector.pol == 'T':
                    orient_angle = np.pi/4
                else:
                    orient_angle = np.pi/4 + np.pi/2
            if handiness == 'B':
                orient_angle = -orient_angle
            return orient_angle
        else: # MFT
            orient_angle = np.deg2rad(float(detector.orient))
            handiness = detector.name.split('_')[3][-1]
            if detector.pol == 'B':
                orient_angle += np.pi/2
            if handiness == 'B':
                orient_angle = -orient_angle
    else:
        orient_angle = 0.0
    return orient_angle


def _rotate_z_vectors_brdcast(result_matrix, quat_matrix):
    """Rotate the z vector using the quaternion `quat`

    Prototype::

        rotate_z_vector(
            result: numpy.array[N, 3],
            v: numpy.array[N, 4],
        )

    This function is equivalent to ``rotate_vector(result, v, [0, 0, 1])``, but it's faster.

    """
    result_matrix[:, 0] = 2 * (quat_matrix[:, 3] * quat_matrix[:, 1] + quat_matrix[:, 0] * quat_matrix[:, 2])
    result_matrix[:, 1] = 2 * (quat_matrix[:, 1] * quat_matrix[:, 2] - quat_matrix[:, 3] * quat_matrix[:, 0])
    result_matrix[:, 2] = 1.0 - 2 * (quat_matrix[:, 0]**2 + quat_matrix[:, 1]**2)


def left_multiply_offset2det(detector:DetectorInfo, offset_rad, axis):
    """Add a rotation around the given axis to the quaternion and update the quaternion.

    Args:
        `detector`:     The instance of `DetectorInfo` type to which offset is to be added.
                        The instance will be destructively updated.

        `offset_rad`:   The offset to be added in the specified direction by `axis`, in radians.

        `axis`:         The axis in the reference frame around which the rotation is to be performed.
    """
    if axis.lower() == 'x':
        rotation_func = quat_rotation_x
    elif axis.lower() == 'y':
        rotation_func = quat_rotation_y
    elif axis.lower() == 'z':
        rotation_func = quat_rotation_z
    else:
        raise ValueError(f"Invalid axis {axis}, expected 'x', 'y', or 'z")

    orient_rad = get_detector_orientation(detector)
    offset_quat = RotQuaternion(
        quats=np.array(rotation_func(offset_rad)),
    )
    vect = np.empty(3)
    rotate_z_vector(vect, *detector.quat.quats[0])


    orient_quat = RotQuaternion(
        quats=np.array(quat_rotation(-orient_rad, *vect))
    )

    interim_quat = offset_quat * orient_quat * detector.quat
    rotate_z_vector(vect, *interim_quat.quats[0])

    orient_quat = RotQuaternion(
        quats=np.array(quat_rotation(orient_rad, *vect))
    )
    detector.quat = orient_quat * interim_quat


def left_multiply_disturb2det(detector:DetectorInfo, start_time, sampling_rate_hz, noise_rad, axis):
    """Add a rotation around the given axis to the quaternion and update the quaternion.

    Args:
        `detector`:     The instance of `DetectorInfo` type to which offset is to be added.
                        The instance will be destructively updated.

        `offset_rad`:   The offset to be added in the specified direction by `axis`, in radians.

        `axis`:         The axis in the reference frame around which the rotation is to be performed.
    """
    if axis.lower() == 'x':
        rotation_func = quat_rotation_x_brdcast
    elif axis.lower() == 'y':
        rotation_func = quat_rotation_y_brdcast
    elif axis.lower() == 'z':
        rotation_func = quat_rotation_z_brdcast
    else:
        raise ValueError(f"Invalid axis {axis}, expected 'x', 'y', or 'z")

    orient_rad = get_detector_orientation(detector)
    noise_quats = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=rotation_func(noise_rad)
        )

    vec_matrix = np.empty([len(noise_rad), 3])
    _rotate_z_vectors_brdcast(vec_matrix, detector.quat.quats)

    orient_quat = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=np.array(quat_rotation_brdcast(-orient_rad, vec_matrix))
    )

    interim_quat = noise_quats * orient_quat * detector.quat
    _rotate_z_vectors_brdcast(vec_matrix, interim_quat.quats)

    orient_quat1 = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=np.array(quat_rotation_brdcast(orient_rad, vec_matrix))
    )
    detector.quat = orient_quat1 * interim_quat

def left_multiply_offset2quat(result, offset_rad, axis):
    """Add a rotation around the given axis to the quaternion and update the quaternion.

    Args:
        `result`:       Input instance of `RotQuaternion` type to which offset is to be added.
                        The instance will be destructively updated.

        `offset_rad`:   The offset to be added in the specified direction by `axis`, in radians.

        `axis`:         The axis in the reference frame around which the rotation is to be performed.
    """
    if axis.lower() == 'x':
        rotation_func = quat_rotation_x
    elif axis.lower() == 'y':
        rotation_func = quat_rotation_y
    elif axis.lower() == 'z':
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
        `result`:           Input instance of `RotQuaternion` type to which noise is to be added.
                            The instance will be destructively updated.

        `start_time`:       Either a floating-point number or an `astropy.time.Time` object.
                            It can be `None` if and only
                            if there is just *one* quaternion in `quats`.

        `sampling_rate_hz`: The sampling frequency of the quaternions, in Hertz. It can be `None` if and only
                            if there is just *one* quaternion in `quats`.

        `noise_rad`:        The noise to be added in the specified direction by `axis`, in radians.
                            It must have shape of 1d NumPy array.

        `axis`:             The axis in the reference frame around which the rotation is to be performed.
    """

    if axis.lower() == 'x':
        rotation_func = quat_rotation_x_brdcast
    elif axis.lower() == 'y':
        rotation_func = quat_rotation_y_brdcast
    elif axis.lower() == 'z':
        rotation_func = quat_rotation_z_brdcast
    else:
        raise ValueError(f"Invalid axis {axis}, expected 'x', 'y', or 'z")

    noise_quats = RotQuaternion(
        start_time=start_time,
        sampling_rate_hz=sampling_rate_hz,
        quats=rotation_func(noise_rad)
        )

    _result = noise_quats * result
    result.start_time = _result.start_time
    result.sampling_rate_hz = _result.sampling_rate_hz
    result.quats = _result.quats


def _ecl2focalplane(offset_rad, axis):
    if axis.lower() == 'x':
        axis = 'y'
    elif axis.lower() == 'y':
        axis = 'x'
    elif axis.lower() == 'z':
        axis = 'z'
        offset_rad = -offset_rad
    return (offset_rad, axis)

def _ecl2spacecraft(offset_rad, axis):
    if axis.lower() == 'x':
        axis = 'y'
    elif axis.lower() == 'y':
        axis = 'x'
    elif axis.lower() == 'z':
        axis = 'z'
    return (offset_rad, axis)


class FocalplaneCoord:
    def __init__(self, detectors: List[DetectorInfo]):
        self.detectors = detectors

    def add_offset(self, offset_rad, axis):
        offset_rad, axis = _ecl2focalplane(offset_rad, axis)
        if isinstance(offset_rad, Iterable):
            # Detector by detecgtor
            assert len(offset_rad) == len(self.detectors)
            for i, det in enumerate(self.detectors):
                left_multiply_offset2det(det, offset_rad[i], axis)
        else:
            # Global in the focal plane
            for i, det in enumerate(self.detectors):
                left_multiply_offset2det(det, offset_rad, axis)

    def add_disturb(self, start_time, sampling_rate_hz, noise_rad_matrix, axis):
        offset_rad, axis = _ecl2focalplane(None, axis)
        if noise_rad_matrix.shape[0] == len(self.detectors):
            # Detector by detecgtor
            for i, det in enumerate(self.detectors):
                left_multiply_disturb2det(det, start_time, sampling_rate_hz, noise_rad_matrix[i], axis)
        else:
            # Global in the focal plane
            for i, det in enumerate(self.detectors):
                left_multiply_disturb2det(det, start_time, sampling_rate_hz, noise_rad_matrix, axis)

class SpacecraftCoord:
    def __init__(self, instrument: InstrumentInfo):
        self.instrument = instrument

    def add_offset(self, offset_rad, axis):
        offset_rad, axis = _ecl2spacecraft(offset_rad, axis)
        left_multiply_offset2quat(self.instrument.bore2spin_quat, offset_rad, axis)

    def add_disturb(self, start_time, sampling_rate_hz, noise_rad, axis):
        offset_rad, axis = _ecl2spacecraft(None, axis)
        left_multiply_disturb2quat(self.instrument.bore2spin_quat, start_time, sampling_rate_hz, noise_rad, axis)

class PointingSys:
    def __init__(self, instrument: InstrumentInfo, detectors: List[DetectorInfo]):
        self.spacecraft = SpacecraftCoord(instrument)
        self.focalplane = FocalplaneCoord(detectors)
