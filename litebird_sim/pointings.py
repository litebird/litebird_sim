import astropy.time
import numpy as np
import numpy.typing as npt

from .hwp import HWP
from .scanning import (
    all_compute_pointing_and_orientation,
    RotQuaternion,
)

DEFAULT_INTERNAL_BUFFER_SIZE_FOR_POINTINGS_MB = 256.0


class PointingProvider:
    """Provides detector pointing angles and HWP angles based on scanning geometry.

    This class computes time-dependent pointing angles (θ, φ, ψ) and optionally
    Half-Wave Plate (HWP) angles for detectors in a LiteBIRD-like scanning configuration.
    It transforms detector-frame quaternions into sky coordinates using a pre-defined
    rotation from boresight to Ecliptic coordinates.

    Optionally, it also manages HWP angle computation if an `HWP` model is associated.

    Parameters:
        bore2ecliptic_quats (RotQuaternion):
            A time-dependent quaternion representing the rotation from the instrument
            boresight frame to the Ecliptic frame. Typically provided by the scanning simulation.

        hwp (HWP | None):
            An instance of a Half-Wave Plate model. If provided, HWP angles will be
            computed as part of the pointing information.

    Attributes:
        bore2ecliptic_quats (RotQuaternion):
            Rotation from the instrument's boresight frame to the Ecliptic frame.

        hwp (HWP | None):
            Associated Half-Wave Plate model, if any.

    Example:
        provider = PointingProvider(bore2ecliptic_quats=q, hwp=hwp_model)
        pointings, hwp = provider.get_pointings(
        ...     detector_quat=det_q,
        ...     start_time=0.0,
        ...     start_time_global=0.0,
        ...     sampling_rate_hz=10.0,
        ...     nsamples=1000
        ... )
    """

    def __init__(
        self,
        # Note that we require here *boresight*→Ecliptic instead of *spin*→Ecliptic
        bore2ecliptic_quats: RotQuaternion,
        hwp: HWP | None = None,
        maximum_internal_buffer_mem_mb: float = DEFAULT_INTERNAL_BUFFER_SIZE_FOR_POINTINGS_MB,
    ):
        self.bore2ecliptic_quats = bore2ecliptic_quats
        self.maximum_internal_buffer_mem_mb = maximum_internal_buffer_mem_mb
        self.hwp = hwp

    def _optimal_block_lengths(self, total_nsamples: int) -> list[int]:
        # Size of one quaternion, in bytes
        quaternion_size_bytes = 4 * self.bore2ecliptic_quats.quats.itemsize

        # Average number of quaternions in each block, to make sure that no more than
        # a fixed number of MB is ever needed to store them
        quaternions_per_block = int(
            self.maximum_internal_buffer_mem_mb * 1024 * 1024 / quaternion_size_bytes
        )

        # How many blocks of quaternions will need to be processed
        number_of_blocks = total_nsamples // quaternions_per_block

        # If something was left out of the previous calculation, include an additional block
        if total_nsamples % quaternions_per_block != 0:
            number_of_blocks += 1

        # Instead of making the first N−1 blocks of the same size and add any leftover to the
        # last block, which might even have 1 sample, try to create the blocks so that they all
        # have roughly the same number of elements
        result = []
        quaternions_left = total_nsamples
        while quaternions_left > 0:
            current_block_length = quaternions_left // (number_of_blocks - len(result))
            result.append(current_block_length)
            quaternions_left -= current_block_length

        assert sum(result) == total_nsamples
        return result

    def has_hwp(self):
        """Return ``True`` if a HWP has been set.

        If the function returns ``True``, you can access the field `hwp`, which
        is an instance of one of the descendeants of class :class:`.HWP`."""
        return self.hwp is not None

    def get_pointings(
        self,
        detector_quat: RotQuaternion,
        start_time: int | float | astropy.time.Time,
        start_time_global: int | float | astropy.time.Time,
        sampling_rate_hz: float,
        nsamples: int,
        pointing_buffer: npt.NDArray | None = None,
        hwp_buffer: npt.NDArray | None = None,
        pointings_dtype=np.float64,
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Compute the time-dependent pointing angles and (optionally) HWP angles for a detector.

        This method computes the pointing angles (θ, φ, ψ) in radians for a detector as a
        function of time, based on its quaternion orientation and the boresight-to-ecliptic
        transformation. If a Half-Wave Plate (HWP) is present, it also computes the HWP angle.

        Args:
            detector_quat (RotQuaternion):
                A time-dependent quaternion representing the rotation from the detector frame
                to the boresight frame.

            start_time (float or astropy.time.Time):
                The timestamp of the first sample for which pointings are required. Can be either
                a float (in seconds) or an `astropy.time.Time` object.

            start_time_global (float or astropy.time.Time):
                The absolute start time of the simulation. Must be of the same type as `start_time`.

            sampling_rate_hz (float):
                Sampling rate in Hz used to compute quaternions and time intervals.

            nsamples (int):
                Number of time samples (i.e., the number of pointings to compute).

            pointing_buffer (np.ndarray, optional):
                A NumPy array of shape `(nsamples, 3)` to store the pointing angles
                (θ, φ, ψ) in radians. If `None`, a new array will be allocated.

            hwp_buffer (np.ndarray, optional):
                A NumPy array of shape `(nsamples,)` to store the HWP angles in radians.
                If `None`, a new array will be allocated only if an HWP is configured.
                If no HWP is present, `hwp_buffer` will be `None`.

            pointings_dtype (data-type, optional):
                Data type to use when allocating `pointing_buffer` or `hwp_buffer`
                (defaults to `np.float64`). If pre-allocated buffers are passed,
                their data types are preserved.

        Returns:
            tuple[np.ndarray, np.ndarray | None]:
                A pair `(pointing_buffer, hwp_buffer)`, containing the computed pointing
                angles and (if applicable) HWP angles.

        Raises:
            AssertionError:
                If `start_time` and `start_time_global` are of different types (i.e.,
                one is a float and the other is an `astropy.time.Time`).
        """

        if isinstance(start_time, astropy.time.Time):
            assert isinstance(start_time_global, astropy.time.Time), (
                "The start_time is a astropy.time.Time object, so start_time_global must also be an astropy.time.Time object."
            )
            start_time_s = (start_time - start_time_global).to("s").value
        else:
            assert isinstance(start_time_global, (int, float)), (
                "The start_time is a float, so start_time_global must also be a float."
            )
            start_time_s = start_time - start_time_global

        if self.hwp is not None:
            if hwp_buffer is None:
                hwp_buffer = np.empty(nsamples, dtype=pointings_dtype)

            self.hwp.get_hwp_angle(
                output_buffer=hwp_buffer,
                start_time_s=start_time_s,
                delta_time_s=1.0 / sampling_rate_hz,
            )
        else:
            hwp_buffer = None

        if pointing_buffer is None:
            pointing_buffer = np.empty(shape=(nsamples, 3), dtype=pointings_dtype)

        block_lengths = self._optimal_block_lengths(total_nsamples=nsamples)

        det_to_ecliptic_quats = self.bore2ecliptic_quats * detector_quat
        cur_time = start_time
        start_sample = 0
        for cur_block_length in block_lengths:
            cur_quaternions = det_to_ecliptic_quats.slerp(
                start_time=cur_time,
                sampling_rate_hz=sampling_rate_hz,
                nsamples=cur_block_length,
            )
            all_compute_pointing_and_orientation(
                result_matrix=pointing_buffer[
                    start_sample : (start_sample + cur_block_length), :
                ],
                quat_matrix=cur_quaternions,
            )

            if isinstance(cur_time, astropy.time.Time):
                cur_time += astropy.time.TimeDelta(
                    cur_block_length / sampling_rate_hz, format="sec"
                )
            else:
                cur_time += cur_block_length / sampling_rate_hz

            start_sample += cur_block_length

        return pointing_buffer, hwp_buffer

    def get_hwp_angle(
        self,
        start_time: int | float | astropy.time.Time,
        start_time_global: int | float | astropy.time.Time,
        sampling_rate_hz: float,
        nsamples: int,
        hwp_buffer: npt.NDArray | None = None,
        pointings_dtype=np.float64,
    ) -> npt.NDArray | None:
        """Compute the Half-Wave Plate (HWP) angle as a function of time.

        This method computes the HWP angle for each sample in a timeline, starting
        from a specified time and advancing based on the sampling rate. The HWP angle
        is returned in a buffer, which is either provided or newly allocated.

        Args:
            start_time (float or astropy.time.Time):
                Time of the first sample for which the HWP angle is computed.
                Can be a float (in seconds) or an `astropy.time.Time` object.

            start_time_global (float or astropy.time.Time):
                Absolute start time of the simulation. Must be the same type as `start_time`.

            sampling_rate_hz (float):
                Sampling rate in Hz, defining the time interval between samples.

            nsamples (int):
                Number of time samples (i.e., number of HWP angle values to compute).

            hwp_buffer (np.ndarray, optional):
                A NumPy array of shape `(nsamples,)` to store the computed HWP angles,
                in radians. If `None`, a new array will be allocated unless the HWP is not present.

            pointings_dtype (data-type, optional):
                Data type to use for the HWP buffer if allocation is needed.
                Defaults to `np.float64`.

        Returns:
            np.ndarray or None:
                The buffer containing computed HWP angles, or `None` if no HWP is configured.

        Raises:
            AssertionError:
                If `start_time` and `start_time_global` are not of the same type.
        """

        if self.hwp is not None:
            if hwp_buffer is None:
                hwp_buffer = np.empty(nsamples, dtype=pointings_dtype)

            if isinstance(start_time, astropy.time.Time):
                assert isinstance(start_time_global, astropy.time.Time), (
                    "The start_time is a astropy.time.Time object, so start_time_global must also be an astropy.time.Time object."
                )
                start_time_s = (start_time - start_time_global).to("s").value
            else:
                assert isinstance(start_time_global, (int, float)), (
                    "The start_time is a float, so start_time_global must also be a float."
                )
                start_time_s = start_time - start_time_global

            self.hwp.get_hwp_angle(
                output_buffer=hwp_buffer,
                start_time_s=start_time_s,
                delta_time_s=1.0 / sampling_rate_hz,
            )
        else:
            hwp_buffer = None

        return hwp_buffer
