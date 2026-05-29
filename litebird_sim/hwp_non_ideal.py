from enum import Enum, auto

import h5py
import numpy as np

from litebird_sim.hwp_jones_parameters import HWPJonesParams
from .hwp import HWP, _get_ideal_hwp_angle, _add_ideal_hwp_angle


class HWPFormalism(Enum):
    JONES = auto()
    MUELLER = auto()


class NonIdealHWP(HWP):
    r"""
    A non-ideal half-wave plate that spins regularly

    This class represents a non-ideal HWP that spins with constant angular velocity.
    The constructor accepts the angular speed, expressed in rad/sec, and the
    start angle (in radians). The latter should be referred to the first time
    sample in the simulation, i.e., the earliest sample simulated in any of the
    MPI processes used for the simulation. The constructor also accepts two booleans:
    "harmonic_expansion", which indicates if the hwp matrices are expanded into harmonics
    of the rotation frequency, and "calculus" which selects between jones or mueller calculus.

    Given a polarization angle :math:`\psi`, this class turns it into
    :math:`\psi + \psi_\text{hwp,0} + 2 \omega_\text{hwp} t`, where
    :math:`\psi_\text{hwp,0}` is the start angle specified in the constructor
    and :math:`\omega_\text{hwp}` is the angular speed of the HWP.
    """

    def __init__(
        self,
        ang_speed_radpsec: float,
        harmonic_expansion: bool,
        calculus: HWPFormalism,
        jones_parameters: HWPJonesParams | None = None,
        start_angle_rad=0.0,
    ):
        self.ang_speed_radpsec = ang_speed_radpsec
        self.start_angle_rad = start_angle_rad
        self.harmonic_expansion = harmonic_expansion
        self.calculus = calculus
        self.jones_parameters = jones_parameters

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

    def write_to_hdf5(
        self, output_file: h5py.File, field_name: str, compression: str | None = None
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
            f"Non Ideal HWP, with rotating speed {self.ang_speed_radpsec} rad/sec "
            f", θ₀ = {self.start_angle_rad}, {self.calculus} calculus and harmonics expansion set to {self.harmonic_expansion}"
        )
