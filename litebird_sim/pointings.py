# -*- encoding: utf-8 -*-

from typing import Optional

import numpy as np
import astropy.time

from .observations import (
    Observation,
    Spin2EclipticQuaternions,
)

from .hwp import HWP

from .scanning import (
    get_det2ecl_quaternions,
    all_compute_pointing_and_polangle,
)

from .coordinates import CoordinateSystem


def apply_hwp_to_obs(obs, hwp: HWP, pointing_matrix):
    """Modify a pointing matrix to consider the effect of a HWP

    This function modifies the variable `pointing_matrix` (a D×N×3 matrix,
    with D the number of detectors and N the number of samples) so that the
    polarization angle considers the behavior of the half-wave plate in
    `hwp`.
    """

    start_time = obs.start_time - obs.start_time_global
    if isinstance(start_time, astropy.time.TimeDelta):
        start_time_s = start_time.to("s").value
    else:
        start_time_s = start_time

    hwp.add_hwp_angle(
        pointing_matrix,
        start_time_s,
        1.0 / obs.sampling_rate_hz,
    )


def get_pointing_buffer_shape(obs: Observation):
    """Return the shape of the pointing matrix for a given observation.

    This function can be used to determine the size to be passed to NumPy
    methods that allocate a new matrix in memory, such as ``numpy.empty``
    and ``numpy.zeros``.
    """
    return obs.n_detectors, obs.n_samples, 3


def get_pointings(
    obs,
    spin2ecliptic_quats: Spin2EclipticQuaternions,
    detector_quats,
    bore2spin_quat,
    quaternion_buffer=None,
    dtype_quaternion=np.float64,
    pointing_buffer=None,
    dtype_pointing=np.float32,
    hwp: Optional[HWP] = None,
    store_pointings_in_obs=True,
):
    """Return the time stream of pointings for the detector

    Given a :class:`Spin2EclipticQuaternions` and a quaternion
    representing the transformation from the reference frame of a
    detector to the boresight reference frame, compute a set of
    pointings for the detector that encompasses the time span
    covered by observation `obs` (i.e., starting from
    `obs.start_time` and including `obs.n_samples` pointings).
    The parameter `spin2ecliptic_quats` can be easily retrieved by
    the field `spin2ecliptic_quats` in a object of
    :class:`.Simulation` object, once the method
    :meth:`.Simulation.generate_spin2ecl_quaternions` is called.
    The parameter `detector_quats` is a stack of detector quaternions. For
    example, it can be:

    - The stack of the field `quat` of an instance of the class
       :class:`.DetectorInfo`

    - If all you want to do is a simulation using a boresight
       direction, you can pass the value ``np.array([[0., 0., 0.,
       1.]])``, which represents the null rotation.

    If you passed an array of :class:`.DetectorInfo` objects to the
    method :meth:`.Simulation.create_observations` through the
    parameter ``detectors``, you can pass ``None`` and it will use the
    detector quaternions from the same :class:`.DetectorInfo` objects.

    The parameter `bore2spin_quat` is calculated through the class
    :class:`.Instrument`, which has the field ``bore2spin_quat``.
    If all you have is the angle β between the boresight and the
    spin axis, just pass ``quat_rotation_y(β)`` here.

    If `HWP` is not ``None``, this specifies the HWP to use for the
    computation of proper polarization angles.

    The return value is a ``(D x N × 3)`` tensor: the colatitude (in
    radians) is stored in column 0 (e.g., ``result[:, :, 0]``), the
    longitude (ditto) in column 1, and the polarization angle
    (ditto) in column 2. You can extract the three vectors using
    the following idiom::

        pointings = obs.get_pointings(...)

        # Extract the colatitude (theta), longitude (psi), and
        # polarization angle (psi) from pointings
        theta, phi, psi = [pointings[:, :, i] for i in (0, 1, 2)]

    If you plan to call this function repeatedly, you can save
    some running time by pre-allocating the buffer used to hold
    the pointings and the quaternions with the parameters
    `pointing_buffer` and `quaternion_buffer`. Both must be a
    NumPy floating-point array whose shape can be computed using
    :func:`.get_quaternion_buffer_shape` and
    :func:`.get_pointing_buffer_shape`. If you use
    these parameters, the return value will be a pointer to the
    `pointing_buffer`.

    """

    if detector_quats is None:
        assert "quat" in dir(obs), (
            "No detector quaternions found, have you passed "
            + '"detectors=" to Simulation.create_observations?'
        )
        detector_quats = obs.quat

    det2ecliptic_quats = get_det2ecl_quaternions(
        obs,
        spin2ecliptic_quats,
        detector_quats,
        bore2spin_quat,
        quaternion_buffer=quaternion_buffer,
        dtype=dtype_quaternion,
    )

    bufshape = get_pointing_buffer_shape(obs)
    if pointing_buffer is None:
        pointing_buffer = np.empty(bufshape, dtype=dtype_pointing)
    else:
        assert (
            pointing_buffer.shape == bufshape
        ), f"error, wrong pointing buffer size: {pointing_buffer.size} != {bufshape}"

    # Compute the pointing direction for each sample
    all_compute_pointing_and_polangle(
        result_matrix=pointing_buffer,
        quat_matrix=det2ecliptic_quats,
    )

    if hwp:
        apply_hwp_to_obs(obs=obs, hwp=hwp, pointing_matrix=pointing_buffer)

    if store_pointings_in_obs:
        obs.pointings = pointing_buffer[:, :, 0:2]
        obs.psi = pointing_buffer[:, :, 2]
        obs.pointing_coords = CoordinateSystem.Ecliptic

    return pointing_buffer
