# -*- encoding: utf-8 -*-

import numpy as np

from .quaternions import quat_rotation_y, quat_rotation_z, quat_left_multiply


class Instrument:
    def __init__(
        self,
        name="",
        boresight_rotangle_deg=0.0,
        spin_boresight_angle_deg=0.0,
        spin_rotangle_deg=0.0,
    ):
        self.name = name
        self.boresight_rotangle_rad = np.deg2rad(boresight_rotangle_deg)
        self.spin_boresight_angle_rad = np.deg2rad(spin_boresight_angle_deg)
        self.spin_rotangle_rad = np.deg2rad(spin_rotangle_deg)
        self.bore2spin_quat = self._compute_bore2spin_quat()

    def _compute_bore2spin_quat(self):
        result = np.array(quat_rotation_z(self.boresight_rotangle_rad))
        quat_left_multiply(result, *quat_rotation_y(self.spin_boresight_angle_rad))
        quat_left_multiply(result, *quat_rotation_z(self.spin_rotangle_rad))
        return result

    def __str__(self):
        return "Instrument({0}, ψ={1}°, β={2}°, φ={3}°)".format(
            self.name,
            np.rad2deg(self.boresight_rotangle_rad),
            np.rad2deg(self.spin_boresight_angle_rad),
            np.rad2deg(self.spin_rotangle_rad),
        )

    def __repr__(self):
        return (
            'Instrument(name="{0}", boresight_rotangle_rad={1:.4e}, '
            "spin_boresight_angle_rad={2:.4e}, spin_rotangle_rad={3:.4e})"
        ).format(
            self.name,
            self.boresight_rotangle_rad,
            self.spin_boresight_angle_rad,
            self.spin_rotangle_rad,
        )

    @staticmethod
    def from_dict(self, d):
        self.boresight_rotangle_rad = np.deg2rad(d["boresight_rotangle_deg"])
        self.spin_boresight_angle_rad = np.deg2rad(d["spin_boresight_angle_deg"])
        self.spin_rotangle_rad = np.deg2rad(d["spin_rotangle_deg"])
