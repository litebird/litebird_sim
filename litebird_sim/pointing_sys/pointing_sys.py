import litebird_sim as lbs
import numpy as np
import healpy as hp
from typing import Union, List, Optional
from ..detectors import InstrumentInfo, DetectorInfo
from ..quaternions import (
    quat_rotation_x,
    quat_rotation_y,
    quat_rotation_z,
    quat_left_multiply,
    rotate_x_vector,
    rotate_y_vector,
    rotate_z_vector,
    rotate_vector,
)
from ..hwp import HWP
from ..observations import (
    Observation,
    Spin2EclipticQuaternions,
)
from ..scanning import (
    get_det2ecl_quaternions,
    polarization_angle,
)
from ..pointings import (
    apply_hwp_to_obs,
    get_pointing_buffer_shape,
)
from ..coordinates import CoordinateSystem
from numba import njit


class ConstantPointingOffset:
    def __init__(self, instrument: InstrumentInfo, detectors: List[DetectorInfo]):
        self.detectors  = detectors
        self.instrument = instrument

    def add_offset_to_spacecraft(self, offset_angle_rad: float, axis_in_PLM_coord: str):
        """This function rotates the entire spacecraft uniformly by offset_angle_rad under the PLM coordinate system by appliying the rotation to the bore2spin quaternion. This is used to represent the poinitng offset systematics of the entire spacecraft. All pointings projected through this function to the healpix map are in the direction from the focal plane to the sky, i.e., conform to the IMO-v2.1 or later definition.

        Args:
            offset_angle_rad (float): The angle to rotate the spacecraft by in radians.
            axis (str): The axis to rotate the spacecraft about. Must be one of 'x', 'y', or 'z' in PLM-coordinate.
        """
        if axis_in_PLM_coord.lower() == 'x':
            rotation_func = quat_rotation_y
        elif axis_in_PLM_coord.lower() == 'y':
            rotation_func = quat_rotation_x
            offset_angle_rad = -offset_angle_rad
        elif axis_in_PLM_coord.lower() == 'z':
            rotation_func = quat_rotation_z
            offset_angle_rad = -offset_angle_rad
        else:
            raise ValueError(f"Invalid axis {axis_in_PLM_coord}, expected 'x', 'y', or 'z")
        quat_left_multiply(self.instrument.bore2spin_quat, *rotation_func(offset_angle_rad))

    def add_offset_to_telescope(self, offset_angle_rad: float, axis_in_aperture_coord: str):
        """This function rotates the telescopes boresight under the aperture coordinate by applying the rotation to the detector quaternions. This is used to represent the pointing offset systematics of the telescope.
        All pointings projected through this function to the healpix map are in the direction from the focal plane to the sky, i.e., conform to the IMO-v2.1 or later definition.

        Args:
            offset_angle_rad (float): The angle to rotate the telescope by in radians.
            axis (str): The axis to rotate the telescope about. Must be one of 'x', 'y', or 'z' in the aperture coordinate.
        """
        if axis_in_aperture_coord.lower() == 'x':
            rotation_func = quat_rotation_y
        elif axis_in_aperture_coord.lower() == 'y':
            rotation_func = quat_rotation_x
            offset_angle_rad = -offset_angle_rad
        elif axis_in_aperture_coord.lower() == 'z':
            rotation_func = quat_rotation_z
        else:
            raise ValueError(f"Invalid axis {axis_in_aperture_coord}, expected 'x', 'y', or 'z'")
        for detector in self.detectors:
            quat_left_multiply(detector.quat, *rotation_func(offset_angle_rad))

    def add_offset_to_detector(self, offsets_theta, offsets_phi):
        """This function rotates the detectors pointing under the detector coordinate.
        The detector coordinate system is defined as a left-handed system in which the direction of observation for each detector is strung along the z-axis; the magnitude of the offset is given as `offsets_theta`, which rotates the direction of observation by a given angle around the y-axis; the direction of the `offsets_phi` is given by phi, which is given as a rotation around the z-axis; and the direction of the offset is given by theta, which is given as a rotation around the z-axis.
        Note: The MFT seems to have a different coordinate system for each detector and offsets in strange directions, and the behaviour is not understood. The design of this function itself should be discussed (Yusuke).
        """
        assert len(offsets_theta) == len(self.detectors), "Length of quats and detectors must be the same"
        assert len(offsets_theta) == len(offsets_phi), "Length of theta and phi must be the same"
        ex_idet = np.zeros(3)
        ey_idet = np.zeros(3)
        ez_idet = np.zeros(3)
        ex_idet_orient = np.zeros(3)
        ey_idet_orient = np.zeros(3)
        ez_idet_orient = np.zeros(3)
        for i,detector in enumerate(self.detectors):
            qx,qy,qz,w = detector.quat
            rotate_x_vector(ex_idet,qx,qy,qz,w)
            rotate_y_vector(ey_idet,qx,qy,qz,w)
            rotate_z_vector(ez_idet,qx,qy,qz,w)

            """Since detector.quat rotates not only the (theta,phi) of the pointing of the detector but also the psi, each detector has its xy-axis rotated around the z-axis by the amount of the transformation caused by the detector.orient, detector.pol and handedness of the detector. The xy-axis is rotated around the z-axis. If this rotation is not corrected for each detector, when the x- or y-axis rotation is applied to a specific detector, the offset is given in a different direction for each detector. To compensate for this, quat_rotation_specific_axis (-orient_angle, ez_idet) returns the xy-axis direction for each detector."""
            orient_angle = get_detector_orientation(detector)
            qx,qy,qz,w   = quat_rotation_specific_axis(-orient_angle, ez_idet)

            rotate_vector(ex_idet_orient,qx,qy,qz,w,ex_idet)
            rotate_vector(ey_idet_orient,qx,qy,qz,w,ey_idet)
            rotate_vector(ez_idet_orient,qx,qy,qz,w,ez_idet)
            quat_left_multiply(detector.quat, *quat_rotation_specific_axis(offsets_theta[i], ey_idet_orient))
            quat_left_multiply(detector.quat, *quat_rotation_specific_axis(-offsets_phi[i],  ez_idet_orient))

def quat_rotation_specific_axis(theta_rad, vect):
    """This function rotates a quaternion by theta_rad about a specific axis.
    Args:
        theta_rad (float): The angle to rotate the quaternion by in radians.
        vect (np.ndarray): The vector to rotate the quaternion about.
    """
    s = np.sin(theta_rad / 2)
    return (vect[0]*s, vect[1]*s, vect[2]*s, np.cos(theta_rad / 2))

def get_detector_orientation(detector:DetectorInfo):
    """This function returns the orientation of the detector in the focal plane."""
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
    else:
        orient_angle = float(detector.orient)
        handiness = detector.name.split('_')[3][-1]
        if detector.pol == 'B':
            orient_angle += np.pi/2
        if handiness == 'B':
            orient_angle = -orient_angle
        return orient_angle


def get_pointings_with_disturbance(
    obs,
    spin2ecliptic_quats: Spin2EclipticQuaternions,
    bore2spin_quat,
    pointing_disturbance,
    detector_quats=None,
    quaternion_buffer=None,
    pointing_buffer=None,
    dtype_pointing=np.float32,
    hwp: Optional[HWP] = None,
    store_pointings_in_obs=True,
):
    """Return the time stream of pointings for the detector with specified disturbance.
    This function follows a structure that `get_pointings` has, but it adds the disturbance to the pointing direction.
    The argument `pointing_disturbance` accsepts the numpy.array[N,3] as a vector which add the disturbance to the pointing direction. Of these, the disturbance which is synchronized with a focal plane is added to the pointing direction of each detector.
    Note that the xyz-coordinates of pointings are defined in the ecliptic coordinate system, and vibrations, etc. identified in the spacecraft coordinate system will not be emulated correctly without appropriate coordinate transformations. However, in the case of random oscillations driven by Gaussians in the spacecraft coordinate system, it is expected that the 2D Gaussians will be properly projected to the sky.
    """

    if detector_quats is None:
        assert "quat" in dir(obs), (
            "No detector quaternions found, have you passed "
            + '"detectors=" to Simulation.create_observations?'
        )
        detector_quats = obs.quat

    bufshape = get_pointing_buffer_shape(obs)
    if pointing_buffer is None:
        pointing_buffer = np.empty(bufshape, dtype=dtype_pointing)
    else:
        assert (
            pointing_buffer.shape == bufshape
        ), f"error, wrong pointing buffer size: {pointing_buffer.size} != {bufshape}"

    for idx, cur_quat in enumerate(detector_quats):

        assert (
            cur_quat.dtype == float
        ), f"error, quaternion must be float, type: {cur_quat.dtype}"

        det2ecliptic_quats = get_det2ecl_quaternions(
            obs,
            spin2ecliptic_quats,
            cur_quat.reshape((1, 4)),
            bore2spin_quat,
            quaternion_buffer=quaternion_buffer,
        )

        # Compute the pointing direction for each sample
        all_compute_pointing_and_polangle_with_disturb(
            result_matrix=pointing_buffer[idx, :, :].reshape(
                (1, pointing_buffer.shape[1], 3)
            ),
            quat_matrix=det2ecliptic_quats,
            disturbance_matrix=pointing_disturbance,
        )

    if hwp:
        apply_hwp_to_obs(obs=obs, hwp=hwp, pointing_matrix=pointing_buffer)

    if store_pointings_in_obs:
        obs.pointings = pointing_buffer[:, :, 0:2]
        obs.psi = pointing_buffer[:, :, 2]
        obs.pointing_coords = CoordinateSystem.Ecliptic

    return pointing_buffer


@njit
def compute_pointing_and_polangle_with_disturb(result, quaternion, disturbance):
    """Store in "result" the pointing direction and polarization angle.

    Prototype::

        compute_pointing_and_polangle(
            result: numpy.array[3],
            quaternion: numpy.array[4],
            disturbance: numpy.array[3],
        )

    The function assumes that `quaternion` encodes a rotation which
    transforms the z axis into the direction of a beam in the sky,
    i.e., it assumes that the beam points towards z in its own
    reference frame and that `quaternion` transforms the reference
    frame to celestial coordinates.

    The variable `result` is used to save the result of the
    computation, and it should be a 3-element NumPy array. On exit,
    its values will be:

    - ``result[0]``: the colatitude of the sky direction, in radians

    - ``result[1]``: the longitude of the sky direction, in radians

    - ``result[2]``: the polarization angle (assuming that in the beam
      reference frame points towards x), measured with respect to the
      North and East directions in the celestial sphere

    This function does *not* support broadcasting; use
    :func:`all_compute_pointing_and_polangle` if you need to
    transform several quaternions at once.

    Example::

        import numpy as np
        result = np.empty(3)
        compute_pointing_and_polangle(result, np.array([
            0.0, np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2,
        ])

    """

    vx, vy, vz, w = quaternion

    # Dirty trick: as "result" is a vector of three floats (θ, φ, ψ),
    # we're reusing it over and over again to compute intermediate
    # vectors before the final result. First, we use it to compute the
    # (x, y, z) pointing direction
    rotate_z_vector(result, vx, vy, vz, w)
    result += disturbance
    result /= np.sqrt(np.sum(result ** 2))
    theta_pointing = np.arctan2(np.sqrt(result[0] ** 2 + result[1] ** 2), result[2])
    phi_pointing = np.arctan2(result[1], result[0])

    # Now reuse "result" to compute the polarization direction
    rotate_x_vector(result, vx, vy, vz, w)

    # Compute the polarization angle
    pol_angle = polarization_angle(
        theta_rad=theta_pointing, phi_rad=phi_pointing, poldir=result
    )

    # Finally, set "result" to the true result of the computation
    result[0] = theta_pointing
    result[1] = phi_pointing
    result[2] = pol_angle

@njit
def all_compute_pointing_and_polangle_with_disturb(result_matrix, quat_matrix, disturbance_matrix):
    """Repeatedly apply :func:`compute_pointing_and_polangle`

    Prototype::

        all_compute_pointing_and_polangle(
            result_matrix: numpy.array[D, N, 3],
            quat_matrix: numpy.array[N, D, 4],
            disturbance_matrix: numpy.array[N, 3],
        )

    Assuming that `result_matrix` is a (D, N, 3) matrix and `quat_matrix` a (N, D, 4)
    matrix, iterate over all the N samples and D detectors and apply
    :func:`compute_pointing_and_polangle` to every item.

    """

    n_dets, n_samples, _ = result_matrix.shape

    assert result_matrix.shape[2] == 3
    assert quat_matrix.shape[0] == n_samples
    assert quat_matrix.shape[1] == n_dets
    assert quat_matrix.shape[2] == 4

    for det_idx in range(n_dets):
        for sample_idx in range(n_samples):
            compute_pointing_and_polangle_with_disturb(
                result_matrix[det_idx, sample_idx, :],
                quat_matrix[sample_idx, det_idx, :],
                disturbance_matrix[sample_idx, :],
            )

def _precompile():
    """Trigger Numba's to pre-compile a few functions defined in this module"""
    result = np.empty((1, 10, 3))
    all_compute_pointing_and_polangle_with_disturb(
        result_matrix=result,
        quat_matrix=np.random.rand(result.shape[1], result.shape[0], 4),
        disturbance_matrix=np.random.rand(result.shape[1], 3),
    )

_precompile()
