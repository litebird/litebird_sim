# -*- encoding: utf-8 -*-
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
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

    def get_hwp_angle(
        self, output_buffer, start_time_s: float, delta_time_s: float
    ) -> None:
        """
        Calculate the rotation angle of the HWP

        This method must be redefined in derived classes. The parameter `start_time_s`
        specifies the time of the first sample in `pointings` and must be a floating-point
        value; this means that you should already have converted any AstroPy time to a plain
        scalar before calling this method. The parameter `delta_time_s` is the inverse
        of the sampling frequency and must be expressed in seconds.

        The result will be saved in `output_buffer`, which must have already been allocated
        with the appropriate number of samples.
        """
        raise NotImplementedError(
            "You should not use the HWP class in your code, use IdealHWP instead"
        )

    def add_hwp_angle(
        self, pointing_buffer, start_time_s: float, delta_time_s: float
    ) -> None:
        """
        Modify pointings so that they include the effect of the HWP

        This method must be redefined in derived classes. The parameter
        ``pointing_buffer`` must be a D×N×3 matrix representing the three angles
        ``(colatitude, longitude, orientation)`` for D detectors and N measurements.
        The function only alters ``orientation`` and returns nothing.

        The parameters `start_time_s` and `delta_time_s` specify the time of the
        first sample in `pointings` and must be floating-point values; this means
        that you should already have converted any AstroPy time to a plain scalar
        before calling this method.

        *Warning*: this changes the interpretation of the ψ variable in the pointings!
        You should better use :meth:`.get_hwp_angle` and keep ψ and the α angle
        of the HWP separate. This method is going to be deprecated in a future
        release of the LiteBIRD Simulation Framework.
        """
        raise NotImplementedError(
            "You should not use the HWP class in your code, use IdealHWP instead"
        )

    def apply_hwp_to_pointings(
        self,
        start_time_s: float,
        delta_time_s: float,  # This will be 1/(19 Hz) in most cases
        bore2ecl_quaternions_inout: npt.NDArray,  # Boresight→Ecliptic quaternions at 19 Hz
        hwp_angle_out: npt.NDArray,
    ) -> None:
        """
        Simulates the presence of a HWP on the pointings and the HWP angle

        This method must be redefined in derived classes. The parameter `start_time_s`
        specifies the time of the first sample in `pointings` and must be a floating-point
        value; this means that you should already have converted any AstroPy time to a plain
        scalar before calling this method. The parameter `delta_time_s` is the inverse
        of the sampling frequency and must be expressed in seconds.

        The result is saved in the NumPy array `hwp_angle_out`, which must have already been
        allocated with the appropriate number of samples. However, if the HWP is not ideal,
        it is possible that the `bore2ecl_quaternions_inout` matrix, whose shape is
        ``(N, 4)`` and contains ``N`` quaternions, is changed by this method, typically
        via a multiplication with a rotation simulating a wobbling effect. (Multiplying
        a quaternion on the *left* of ``bore2ecl_quaternions_inout`` means that you
        are adding a rotation at the *end* of the chain of transformations, i.e.,
        after the conversion to the Ecliptic reference frame. Multiplying a quaternion
        on the *right* means that you are introducing a new rotation between the
        reference frame of the detector and of the boresight of the focal plane.)

        :param start_time_s:
        :param delta_time_s:
        :param bore2ecl_quaternions_inout:
        :param hwp_angle_out:
        :return:
        """
        raise NotImplementedError(
            "You should not use the HWP class in your code, use IdealHWP instead"
        )

    def write_to_hdf5(
        self, output_file: h5py.File, field_name: str, compression: Optional[str] = None
    ) -> h5py.Dataset:
        """Write the definition of the HWP into a HDF5 file

        You should never call this function directly. It is used to save :class:`.Observation`
        objects to disk.
        """
        raise NotImplementedError(
            "You should not use the HWP class in your code, use IdealHWP instead"
        )

    def read_from_hdf5(self, input_dataset: h5py.Dataset) -> None:
        raise NotImplementedError(
            "You should not use the HWP class in your code, use IdealHWP instead"
        )

    def __str__(self):
        raise NotImplementedError(
            "You should not use the HWP class in your code, use IdealHWP instead"
        )


@njit
def _get_ideal_hwp_angle(
    output_buffer, start_time_s, delta_time_s, start_angle_rad, ang_speed_radpsec
):
    for sample_idx in range(output_buffer.size):
        angle = (
            start_angle_rad
            + (start_time_s + delta_time_s * sample_idx) * 2 * ang_speed_radpsec
        ) % (2 * np.pi)

        output_buffer[sample_idx] = angle


def _add_ideal_hwp_angle(
    pointing_buffer, start_time_s, delta_time_s, start_angle_rad, ang_speed_radpsec
):
    detectors, samples, _ = pointing_buffer.shape
    hwp_angles = np.empty(samples, dtype=pointing_buffer.dtype)
    _get_ideal_hwp_angle(
        output_buffer=hwp_angles,
        start_time_s=start_time_s,
        delta_time_s=delta_time_s,
        start_angle_rad=start_angle_rad,
        ang_speed_radpsec=ang_speed_radpsec,
    )

    for det_idx in range(detectors):
        pointing_buffer[det_idx, :, 2] += hwp_angles


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

    def get_hwp_angle(
        self, output_buffer, start_time_s: float, delta_time_s: float
    ) -> None:
        _get_ideal_hwp_angle(
            output_buffer=output_buffer,
            start_time_s=start_time_s,
            delta_time_s=delta_time_s,
            start_angle_rad=self.start_angle_rad,
            ang_speed_radpsec=self.ang_speed_radpsec,
        )

    def add_hwp_angle(
        self, pointing_buffer, start_time_s: float, delta_time_s: float
    ) -> None:
        _add_ideal_hwp_angle(
            pointing_buffer=pointing_buffer,
            start_time_s=start_time_s,
            delta_time_s=delta_time_s,
            start_angle_rad=self.start_angle_rad,
            ang_speed_radpsec=self.ang_speed_radpsec,
        )

    def apply_hwp_to_pointings(
        self,
        start_time_s: float,
        delta_time_s: float,  # This will be 1/(19 Hz) in most cases
        bore2ecl_quaternions_inout: npt.NDArray,  # Boresight→Ecliptic quaternions at 19 Hz
        hwp_angle_out: npt.NDArray,
    ) -> None:
        # We do not touch `bore2ecl_quaternions_inout`, as an ideal HWP does not
        # alter the (θ, φ) direction of the boresight nor the orientation ψ
        self.get_hwp_angle(
            output_buffer=hwp_angle_out,
            start_time_s=start_time_s,
            delta_time_s=delta_time_s,
        )

    def write_to_hdf5(
        self, output_file: h5py.File, field_name: str, compression: Optional[str] = None
    ) -> h5py.Dataset:
        # For an ideal HWP, we just save an empty dataset with a few attributes
        # This means that we must *not* use the "compression" field here, otherwise
        # h5py will complain that “empty datasets don't support chunks/filters”…
        new_dataset = output_file.create_dataset(
            name=field_name,
            dtype=np.float64,
        )

        new_dataset.attrs["class_name"] = "IdealHWP"
        new_dataset.attrs["ang_speed_radpsec"] = self.ang_speed_radpsec
        new_dataset.attrs["start_angle_rad"] = self.start_angle_rad

        return new_dataset

    def read_from_hdf5(self, input_dataset: h5py.Dataset) -> None:
        assert input_dataset.attrs["class_name"] == "IdealHWP"

        self.ang_speed_radpsec = input_dataset.attrs["ang_speed_radpsec"]
        self.start_angle_rad = input_dataset.attrs["start_angle_rad"]

    def __str__(self):
        return (
            f"Ideal HWP, with rotating speed {self.ang_speed_radpsec} rad/sec "
            f"and θ₀ = {self.start_angle_rad}"
        )


def read_hwp_from_hdf5(input_file: h5py.File, field_name: str) -> HWP:
    dataset = input_file[field_name]
    class_name = dataset.attrs["class_name"]

    if class_name == "IdealHWP":
        # Let's pass dummy values to each field. They will be
        # fixed once the data are read from the file
        result = IdealHWP(
            ang_speed_radpsec=0.0,
            start_angle_rad=0.0,
        )
    else:
        # If new derived classes from HWP are implemented, add them here with an `elif`
        assert (
            False
        ), f"read_hwp_from_hdf5() does not support a HWP of type {class_name}"

    result.read_from_hdf5(input_dataset=dataset)
    return result
