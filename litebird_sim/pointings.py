# -*- encoding: utf-8 -*-

from typing import Optional, List, Union

import numpy as np
import numpy.typing as npt
import astropy.time

from .observations import (
    Observation,
)

from .hwp import HWP

from .scanning import (
    get_det2ecl_quaternions,
    all_compute_pointing_and_orientation,
    RotQuaternion,
)

from .coordinates import CoordinateSystem


def apply_hwp_to_obs(obs, hwp: HWP, pointing_matrix):
    """Modify a pointing matrix to consider the effect of a HWP

    This function modifies the variable `pointing_matrix` (a D×N×3 matrix,
    with D the number of detectors and N the number of samples) so that the
    orientation angle considers the behavior of the half-wave plate in
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
    spin2ecliptic_quats: RotQuaternion,
    bore2spin_quat: RotQuaternion,
    detector_quats: Optional[Union[np.ndarray, List[RotQuaternion]]] = None,
    quaternion_buffer=None,
    pointing_buffer=None,
    dtype_pointing=np.float32,
    hwp: Optional[HWP] = None,
    store_pointings_in_obs=True,
):
    """Return the time stream of pointings for *one* observation

    Given a :class:`RotQuaternion` and a quaternion
    representing the transformation from the reference frame of a
    detector to the boresight reference frame, compute a set of
    pointings for the detector that encompasses the time span
    covered by observation `obs` (i.e., starting from
    `obs.start_time` and including `obs.n_samples` pointings).
    The parameter `spin2ecliptic_quats` can be easily retrieved by
    the field `spin2ecliptic_quats` in a object of
    :class:`.Simulation` object, once the method
    :meth:`.Simulation.set_scanning_strategy` is called.

    The parameter `bore2spin_quat` is calculated through the class
    :class:`.Instrument`, which has the field ``bore2spin_quat``.
    If all you have is the angle β between the boresight and the
    spin axis, just pass ``quat_rotation_y(β)`` here.

    The parameter `detector_quats` is optional. By default, this is
    ``None``: in this case, if you passed an array of :class:`.DetectorInfo`
    objects to the method :meth:`.Simulation.create_observations`
    through the parameter ``detectors``, get_pointings will use
    the detector quaternions from the same :class:`.DetectorInfo` objects.
    Otherwise, it can contain a stack of detector quaternions. For example,
    it can be:

    - The stack of the field `quat` of an instance of the class
       :class:`.DetectorInfo`

    - If all you want to do is a simulation using a boresight
       direction, you can pass the value ``np.array([[0., 0., 0.,
       1.]])``, which represents the null rotation.

    If `HWP` is not ``None``, this specifies the HWP to use for the
    computation of proper polarization angles.

    **Warning**: if `hwp` is not ``None``, the code adds the α angle of the
    HWP to the orientation angle ψ, which is generally not correct! This
    is going to be fixed in the next release of the LiteBIRD Simulation
    Framework.

    The return value is a ``(D x N × 3)`` tensor: the colatitude (in
    radians) is stored in column 0 (e.g., ``result[:, :, 0]``), the
    longitude (ditto) in column 1, and the polarization angle
    (ditto) in column 2. You can extract the three vectors using
    the following idiom::

        pointings = obs.get_pointings(...)

        # Extract the colatitude (theta), longitude (phi), and
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
        detector_quats = [RotQuaternion(q) for q in obs.quat]
    else:
        assert isinstance(detector_quats, list) or (
            isinstance(detector_quats, np.ndarray)
            and isinstance(detector_quats[0], RotQuaternion)
        ), (
            "`detector_quats` is a {} object, but starting from version 0.13.0 it must"
            " be a list of `RotQuaternion` objects"
        ).format(str(type(detector_quats)))

    bufshape = get_pointing_buffer_shape(obs)
    if pointing_buffer is None:
        pointing_buffer = np.empty(bufshape, dtype=dtype_pointing)
    else:
        assert (
            pointing_buffer.shape == bufshape
        ), f"error, wrong pointing buffer size: {pointing_buffer.size} != {bufshape}"

    for idx, cur_quat in enumerate(detector_quats):
        assert isinstance(cur_quat, RotQuaternion)

        # Get the quaternions at the same sampling frequency of the TOD.
        # Since the call is just for ONE detector, we are not going to
        # waste too much memory. (Moreover, we are re-using `quaternion_buffer`
        # over and over again in this `for` loop.)
        det2ecliptic_quats = get_det2ecl_quaternions(
            obs=obs,
            spin2ecliptic_quats=spin2ecliptic_quats,
            detector_quats=[RotQuaternion(cur_quat)],
            bore2spin_quat=bore2spin_quat,
            quaternion_buffer=quaternion_buffer,
        )

        # Compute the pointing direction for each sample using the
        # quaternions in `det2ecliptic_quats`. This is just a matter of
        # applying the rotation encoded by each quaternion to the
        # z-axis, which represents the beam axis in the beam reference
        # frame.
        all_compute_pointing_and_orientation(
            result_matrix=pointing_buffer[idx, :, :].reshape(
                (1, pointing_buffer.shape[1], 3)
            ),
            quat_matrix=det2ecliptic_quats,
        )

    if hwp:
        apply_hwp_to_obs(obs=obs, hwp=hwp, pointing_matrix=pointing_buffer)

    if store_pointings_in_obs:
        obs.pointings = pointing_buffer[:, :, 0:2]
        obs.psi = pointing_buffer[:, :, 2]
        obs.pointing_coords = CoordinateSystem.Ecliptic

    return pointing_buffer


def get_pointings_for_observations(
    obs: Union[Observation, List[Observation]],
    spin2ecliptic_quats: RotQuaternion,
    bore2spin_quat,
    hwp: Optional[HWP] = None,
    store_pointings_in_obs=True,
    dtype_pointing=np.float32,
):
    """Obtain pointings for a list of observations

    This is a wrapper around the :func:`.get_pointings` function that computes
    pointing information for a list of observations and returns a list of pointings.
    If a single observation is passed then a single pointing array is returned, and,
    practically, this function only calls :func:`.get_pointings`.

    If `store_pointings_in_obs` is ``True``, then the pointings are stored in each
    :class:`.Observation` object in `obs`.
    """

    if isinstance(obs, Observation):
        quaternion_buffer = np.zeros((obs.n_samples, 1, 4), dtype=np.float64)
        pointings = get_pointings(
            obs,
            spin2ecliptic_quats,
            bore2spin_quat,
            quaternion_buffer=quaternion_buffer,
            dtype_pointing=dtype_pointing,
            hwp=hwp,
            store_pointings_in_obs=store_pointings_in_obs,
        )
    else:
        pointings = []
        for ob in obs:
            quaternion_buffer = np.zeros((ob.n_samples, 1, 4), dtype=np.float64)
            pointings.append(
                get_pointings(
                    ob,
                    spin2ecliptic_quats,
                    bore2spin_quat,
                    quaternion_buffer=quaternion_buffer,
                    dtype_pointing=dtype_pointing,
                    hwp=hwp,
                    store_pointings_in_obs=store_pointings_in_obs,
                )
            )

    return pointings


class PointingProvider:
    def __init__(
        self,
        # Note that we require here *boresight*→Ecliptic instead of *spin*→Ecliptic
        bore2ecliptic_quats: RotQuaternion,
        hwp: Optional[HWP] = None,
    ):
        self.bore2ecliptic_quats = bore2ecliptic_quats
        self.hwp = hwp

    def get_pointings(
        self,
        detector_quat: RotQuaternion,
        start_time: Union[float, astropy.time.Time],
        start_time_global: Union[float, astropy.time.Time],
        sampling_rate_hz: float,
        nsamples: int,
        pointing_buffer: Optional[npt.NDArray] = None,
        hwp_buffer: Optional[npt.NDArray] = None,
    ) -> (npt.NDArray, Optional[npt.NDArray]):
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
                hwp_buffer = np.empty(nsamples, dtype=np.float64)

            start_time_s = start_time - start_time_global
            if isinstance(start_time_s, astropy.time.TimeDelta):
                start_time_s = start_time_s.to("s").value

            self.hwp.apply_hwp_to_pointings(
                start_time_s=start_time_s,
                delta_time_s=1.0 / sampling_rate_hz,
                bore2ecl_quaternions_inout=full_quaternions,
                hwp_angle_out=hwp_buffer,
            )
        else:
            hwp_buffer = None

        if pointing_buffer is None:
            pointing_buffer = np.empty(shape=(nsamples, 3), dtype=np.float64)

        all_compute_pointing_and_orientation(
            result_matrix=pointing_buffer,
            quat_matrix=full_quaternions,
        )

        return pointing_buffer, hwp_buffer
