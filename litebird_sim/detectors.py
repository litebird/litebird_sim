from dataclasses import dataclass, field, fields
from typing import Any

from uuid import UUID
import numpy as np
import logging as log

import astropy.time as astrotime
from .imo import Imo
from .quaternions import quat_rotation_y, quat_rotation_z, quat_left_multiply
from .scanning import RotQuaternion

from .bandpasses import BandPassInfo


def url_to_uuid(url: str) -> UUID:
    return UUID([x for x in url.split("/") if x != ""][-1])


def normalize_time_dependent_quaternion(
    quat: np.ndarray | RotQuaternion,
) -> RotQuaternion:
    """Make sure that a quaternion is represented as a :class:`TimeDependentQuaternion` object"""

    if isinstance(quat, RotQuaternion):
        return quat

    if isinstance(quat, np.ndarray):
        quat = quat.reshape(4)
    else:
        quat = np.array([float(x) for x in quat])

    assert quat.size == 4

    # Normalize the quaternion
    quat /= np.sqrt(quat.dot(quat))

    return RotQuaternion(quats=quat)


@dataclass
class DetectorInfo:
    """A class encapsulating the basic information about a LiteBIRD detector.

    This data class stores the key properties of a detector, including its geometry,
    spectral and noise characteristics, polarization, pointing, and optional systematic
    effects.

    Initialization Methods:
        The class can be instantiated in one of the following ways:

        1. Using the default constructor (recommended to specify at least `name` and `sampling_rate_hz`)::

               det = DetectorInfo(name="dummy", sampling_rate_hz=10.0)

        2. Using the class method :meth:`.from_dict`, which builds a detector from a dictionary of fields.

        3. Using the class method :meth:`.from_imo`, which extracts detector metadata from the LiteBIRD
           Instrument Model (see :class:`.Imo`).

    Post-initialization Behavior:
        - If the quaternion (`quat`) is not provided, it defaults to the identity quaternion (no rotation).
        - The quaternion is automatically normalized using `normalize_time_dependent_quaternion`.
        - If `band_freqs_ghz` is provided as a NumPy array, `band_weights` must have the same length.
        - If `band_freqs_ghz` is not provided, a simple rectangular bandpass is constructed from
          `bandcenter_ghz` and `bandwidth_ghz` using `BandPassInfo`.

    Dictionary Format:
        When using :meth:`.from_dict`, the input dictionary may contain the following keys:

            name, wafer, pixel, pixtype, channel, squid,
            bandcenter_ghz, bandwidth_ghz, band_freqs_ghz, band_weights,
            sampling_rate_hz, fwhm_arcmin, ellipticity, psi_rad,
            net_ukrts, pol_sensitivity_ukarcmin, fknee_mhz, fmin_hz,
            alpha, pol, orient, quat, pol_angle_rad, pol_efficiency,
            mueller_hwp, mueller_hwp_solver, pointing_theta_phi_psi_deg,
            pointing_u_v, g_one_over_k, amplitude_2f_k

        Quaternions can be specified as:
        - A list of 4 numbers (static rotation), or
        - A dictionary with keys like `start_time`, `quats`, `sampling_rate_hz`, etc., for time-varying rotation.

    Args:
        name (str): The name of the detector. Default is an empty string.

        wafer (str | None): The name of the wafer hosting the detector
            (e.g., ``"H00"``, ``"L07"``, etc.). Default is None.

        pixel (int | None): The index of the pixel within the wafer. Default is None.

        pixtype (str | None): The type of the pixel (e.g., ``"HP1"``, ``"LP3"``). Default is None.

        channel (str | None): The channel name. Default is None.

        squid (int | None): The SQUID number associated with the detector. Default is None.

        sampling_rate_hz (float): Sampling rate of the ADC associated with this detector, in Hz.
            Default is 0.0.

        fwhm_arcmin (float): Full Width at Half Maximum of the beam in arcminutes.
            Defined as fwhm = sqrt(fwhm_max*fwhm_min). Default is 0.0.

        ellipticity (float): Ellipticity of the beam (major/minor axis ratio). Defined as
            fwhm_max/fwhm_min Default is 1 (circular beam). Default is 1.0.

        psi_rad (float): Orientation angle of the beam's major axis with respect to the x-axis,
            in radians. Default is 0.0.

        net_ukrts (float): Noise Equivalent Temperature (NET) in μK√s, representing per-sample noise.
            Used when adding noise to timelines. Default is 0.0.

        pol_sensitivity_ukarcmin (float): Detector polarization sensitivity in μK·arcmin.
            Includes effects such as cosmic ray hits and repointing losses.
            Should **not** be used for timeline noise simulation. Default is 0.0.

        bandcenter_ghz (float): Center frequency of the detector band, in GHz. Default is 0.0.

        bandwidth_ghz (float): Bandwidth of the detector, in GHz. Default is 0.0.

        band_freqs_ghz (np.ndarray | None): Array of sampled frequencies in the band (GHz).
            Default is None.

        band_weights (np.ndarray | None): Corresponding normalized band weights. Default is None.

        fknee_mhz (float): 1/f noise knee frequency in mHz. Default is 0.0.

        fmin_hz (float): Minimum noise frequency used in synthetic noise generation, in Hz.
            Default is 0.0.

        alpha (float): Slope of the 1/f noise power spectrum. Default is 0.0.

        pol (str | None): Polarization type (``"T"``, ``"B"``, etc.). Default is None.

        orient (str | None): Polarization orientation (``"Q"``, ``"U"``, etc.). Default is None.

        quat (:class:`.TimeDependentQuaternion`): Quaternion representing the rotation from
            the detector frame to the boresight frame. Default is identity (no rotation).

        pol_angle_rad (float): Polarization angle relative to the detector frame x-axis, in radians.
            Default is 0.0.

        pol_efficiency (float): Polarization efficiency (γ), as defined in Eq. 15 of astro-ph/0606606.
            Default is 1.0.

        mueller_hwp (None | dict): Mueller matrix of the HWP, expanded into three harmonics
            of the HWP rotation frequency. Default is None (no HWP).

        mueller_hwp_solver (None | dict): Mueller matrix used in the mapmaking solver to
            model a non-ideal HWP. Also decomposed into three harmonics. Default is None (no HWP).

        pointing_theta_phi_psi_deg (None | np.ndarray): Array of pointing angles (θ, φ, ψ)
            in degrees: colatitude, longitude, and orientation. Default is None.

        pointing_u_v (None | np.ndarray): Detector pointing in focal plane coordinates (u, v),
            with (0, 0) representing the central axis of the focal plane. Default is None.

        g_one_over_k (float): Gain conversion factor (1/K), used in calibration models. Default is 0.0.

        amplitude_2f_k (float): Amplitude of the 2f systematic signal component in Kelvin. Default is 0.0.

    """

    name: str = ""
    wafer: str | None = None
    pixel: int | None = None
    pixtype: str | None = None
    channel: str | None = None
    squid: int | None = None
    sampling_rate_hz: float = 0.0
    fwhm_arcmin: float = 0.0
    ellipticity: float = 1.0
    psi_rad: float = 0.0
    bandcenter_ghz: float = 0.0
    bandwidth_ghz: float = 0.0
    band_freqs_ghz: None | np.ndarray = None
    band_weights: None | np.ndarray = None
    net_ukrts: float = 0.0
    pol_sensitivity_ukarcmin: float = 0.0
    fknee_mhz: float = 0.0
    fmin_hz: float = 0.0
    alpha: float = 0.0
    pol: str | None = None
    orient: str | None = None
    quat: Any = None
    pol_angle_rad: float = 0.0
    pol_efficiency: float = 1.0
    mueller_hwp: None | dict = None
    mueller_hwp_solver: None | dict = None
    pointing_theta_phi_psi_deg: None | np.ndarray = None
    pointing_u_v: None | np.ndarray = None
    g_one_over_k: float = 0.0
    amplitude_2f_k: float = 0.0

    def __post_init__(self):
        if self.quat is None:
            self.quat = (0.0, 0.0, 0.0, 1.0)

        self.quat = normalize_time_dependent_quaternion(self.quat)

        if isinstance(self.band_freqs_ghz, np.ndarray):
            assert len(self.band_freqs_ghz) == len(self.band_weights)

        # Warn if mueller_hwp is not a 4x4 numpy array
        if self.mueller_hwp is not None:
            if not (
                isinstance(self.mueller_hwp, np.ndarray)
                and self.mueller_hwp.shape == (4, 4)
            ):
                log.warning(
                    f"Detector '{self.name}': mueller_hwp is not a 4x4 numpy array "
                    f"(found type {type(self.mueller_hwp)}, shape {getattr(self.mueller_hwp, 'shape', None)})"
                )

    @staticmethod
    def from_dict(dictionary: dict[str, Any]):
        """Create a detector from the contents of a dictionary

        The parameter `dictionary` must contain one key for each of
        the fields in this dataclass:

        - ``name``
        - ``wafer``
        - ``pixel``
        - ``pixtype``
        - ``channel``
        - ``squid``
        - ``bandcenter_ghz``
        - ``bandwidth_ghz``
        - ``band_freqs_ghz``
        - ``band_weights``
        - ``sampling_rate_hz``
        - ``fwhm_arcmin``
        - ``ellipticity``
        - ``psi_rad``
        - ``net_ukrts``
        - ``pol_sensitivity_ukarcmin``
        - ``fknee_mhz``
        - ``fmin_hz``
        - ``alpha``
        - ``pol``
        - ``orient``
        - ``quat``
        - ``pol_angle_rad``
        - ``pol_efficiency``
        - ``mueller_hwp``
        - ``mueller_hwp_solver``
        - ``pointing_theta_phi_psi_deg``
        - ``pointing_u_v``
        - ``g_one_over_k``
        - ``amplitude_2f_k``

        """
        result = DetectorInfo()
        for param in fields(DetectorInfo):
            if param.name not in dictionary:
                continue

            if param.name != "quat":
                setattr(result, param.name, dictionary[param.name])
            else:
                # Quaternions need a more complicated algorithm…
                time_dependent_quat = dictionary[param.name]
                if isinstance(time_dependent_quat, dict):
                    start_time = time_dependent_quat["start_time"]
                    if isinstance(start_time, str):
                        start_time = astrotime.Time(start_time)

                    if "quats" in time_dependent_quat:
                        quats = np.array(time_dependent_quat["quats"])
                    elif "quaternion_matrix_shape" in time_dependent_quat:
                        quats = np.zeros(
                            shape=time_dependent_quat["quaternion_matrix_shape"]
                        )
                    else:
                        quats = None

                    result.quat = RotQuaternion(
                        quats=quats,
                        start_time=start_time,
                        sampling_rate_hz=time_dependent_quat["sampling_rate_hz"],
                    )
                else:
                    # The quaternion is just a list of 4 numbers
                    result.quat = RotQuaternion(np.array(time_dependent_quat))

        if not isinstance(result.band_freqs_ghz, np.ndarray):
            result.band = BandPassInfo(result.bandcenter_ghz, result.bandwidth_ghz)
            result.band_freqs_ghz, result.band_weights = [
                result.band.freqs_ghz,
                result.band.weights,
            ]

        # Force initializers to be called again
        result.__post_init__()
        return result

    @staticmethod
    def from_imo(imo: Imo, url: UUID | str):
        """Create a `DetectorInfo` object from a definition in the IMO

        The `url` must either specify a UUID or a full URL to the
        object.

        Example::

            import litebird_sim as lbs
            imo = Imo()
            det = DetectorInfo.from_imo(
                imo=imo,
                url="/releases/v1.0/satellite/LFT/L1-040/L00_008_QA_040T/detector_info",
            )

        """
        obj = imo.query(url)
        return DetectorInfo.from_dict(obj.metadata)


@dataclass
class FreqChannelInfo:
    bandcenter_ghz: float
    channel: str | None = None
    bandwidth_ghz: float = 0.0
    band_freqs_ghz: None | np.ndarray = None
    band_weights: None | np.ndarray = None
    net_detector_ukrts: float = 0.0
    net_channel_ukrts: float = 0.0
    pol_sensitivity_channel_ukarcmin: float = 0.0
    sampling_rate_hz: float = 0.0
    fwhm_arcmin: float = 0.0
    ellipticity: float = 1.0
    psi_rad: float = 0.0
    fknee_mhz: float = 0.0
    fmin_hz: float = 1e-5
    alpha: float = 1.0
    number_of_detectors: int = 0
    detector_names: list[str] = field(default_factory=list)
    detector_objs: list[UUID] = field(default_factory=list)

    """A data class representing the configuration of a frequency channel in LiteBIRD.

    This class encapsulates the spectral, noise, and beam properties of a frequency
    channel, along with metadata about the associated detectors.

    Initialization Methods:
        - Direct instantiation (requires at least `bandcenter_ghz`)
        - :meth:`.from_dict`: load from a dictionary
        - :meth:`.from_imo`: load from a LiteBIRD Instrument Model (IMO)

    Post-initialization Behavior:
        - If `channel` is not specified, it defaults to "<bandcenter_ghz> GHz".
        - If `number_of_detectors` is provided:
            - Validates or generates `detector_names` and `detector_objs`.
        - If `number_of_detectors` is not provided:
            - It is inferred from `detector_names` or `detector_objs`, defaulting to 1 detector named "det0".
        - If either `net_channel_ukrts` or `net_detector_ukrts` is zero, it is computed from the other.
        - All entries in `detector_objs` are converted to UUIDs.
        - If `band_freqs_ghz` is not provided, a default rectangular bandpass is generated from
          `bandcenter_ghz` and `bandwidth_ghz`.

    Methods:
        - from_dict(dict): Create an instance from a plain dictionary.
        - from_imo(imo, url): Load metadata from an IMO object using a UUID or URL.
        - get_boresight_detector(name): Return a simplified `DetectorInfo` object
          with boresight-aligned geometry and inherited channel parameters.

    Args:
        bandcenter_ghz (float): Center frequency of the channel in GHz.

        channel (str or None): Channel name. Defaults to "<bandcenter_ghz> GHz".

        bandwidth_ghz (float): Bandwidth of the channel in GHz. Default is 0.0.

        band_freqs_ghz (np.ndarray or None): Sampled frequencies across the band in GHz.
            If not provided, a rectangular bandpass is constructed. Default is None.

        band_weights (np.ndarray or None): Normalized bandpass weights. Default is None.

        net_detector_ukrts (float): Noise Equivalent Temperature (NET) per detector in μK√s.
            Default is 0.0.

        net_channel_ukrts (float): Total NET for the channel in μK√s, accounting for
            all detectors. Default is 0.0.

        pol_sensitivity_channel_ukarcmin (float): Polarization sensitivity of the channel in μK·arcmin.
            Default is 0.0.

        sampling_rate_hz (float): ADC sampling rate in Hz. Default is 0.0.

        fwhm_arcmin (float): Averaged beam Full Width at Half Maximum in arcminutes. Default is 0.0.

        ellipticity (float): Averaged beam ellipticity (major/minor axis ratio). Default is 1.0.

        psi_rad (float): Averaged beam orientation angle (major axis vs x-axis) in radians. Default is 0.0.

        fknee_mhz (float): Knee frequency of 1/f noise in mHz. Default is 0.0.

        fmin_hz (float): Minimum frequency for synthetic noise generation in Hz. Default is 1e-5.

        alpha (float): Slope of the 1/f noise component. Default is 1.0.

        number_of_detectors (int): Number of detectors in the channel. If 0, inferred from
            `detector_names` or `detector_objs`. Default is 0.

        detector_names (list[str]): List of detector names. If not provided, generated automatically.
            Default is an empty list.

        detector_objs (list[UUID]): List of UUIDs (or strings convertible to UUIDs) identifying
            detector entries in the IMO. Default is an empty list.
    """

    def __post_init__(self):
        if self.channel is None:
            self.channel = f"{self.bandcenter_ghz:.1f} GHz"

        if self.number_of_detectors > 0:
            # First hypothesis: we have set the number of detectors. Check
            # that this is consistent with the field "self.detector_names"

            if self.detector_names:
                assert len(self.detector_names) == self.number_of_detectors
            else:
                self.detector_names = [
                    f"det{x:d}" for x in range(self.number_of_detectors)
                ]

            if self.detector_objs:
                assert len(self.detector_objs) == self.number_of_detectors
        else:
            # Second hypothesis: the number of detectors was not set.
            # Check if we have detector names or objects.
            if self.detector_names:
                self.number_of_detectors = len(self.detector_names)
            else:
                if self.detector_objs:
                    self.number_of_detectors = len(self.detector_objs)
                else:
                    self.number_of_detectors = 1
                    self.detector_names = ["det0"]

        # If either net_channel_ukrts or net_detector_ukrts were not
        # set, derive one from another
        if self.net_channel_ukrts == 0 and self.net_detector_ukrts > 0:
            self.net_channel_ukrts = self.net_detector_ukrts / np.sqrt(
                self.number_of_detectors
            )
        elif self.net_channel_ukrts > 0 and self.net_detector_ukrts == 0:
            self.net_detector_ukrts = self.net_channel_ukrts * np.sqrt(
                self.number_of_detectors
            )

        # Convert the items in the list of detector objects into
        # proper UUIDs
        for det_idx, det_obj in enumerate(self.detector_objs):
            if not isinstance(det_obj, UUID):
                if "/" in det_obj:
                    cur_uuid = url_to_uuid(det_obj)
                else:
                    cur_uuid = UUID(det_obj)

                self.detector_objs[det_idx] = cur_uuid

    @staticmethod
    def from_dict(dictionary: dict[str, Any]):
        result = FreqChannelInfo(bandcenter_ghz=0.0)
        for param in fields(FreqChannelInfo):
            if param.name in dictionary:
                setattr(result, param.name, dictionary[param.name])

        if not isinstance(result.band_freqs_ghz, np.ndarray):
            result.band = BandPassInfo(result.bandcenter_ghz, result.bandwidth_ghz)
            result.band_freqs_ghz, result.band_weights = [
                result.band.freqs_ghz,
                result.band.weights,
            ]

        # Force initializers in __post_init__ to be called
        result.__post_init__()
        return result

    @staticmethod
    def from_imo(imo: Imo, url: UUID | str):
        obj = imo.query(url)
        return FreqChannelInfo.from_dict(obj.metadata)

    def get_boresight_detector(self, name="mock") -> DetectorInfo:
        return DetectorInfo(
            name=name,
            channel=self.channel,
            fwhm_arcmin=self.fwhm_arcmin,
            ellipticity=self.ellipticity,
            psi_rad=self.psi_rad,
            net_ukrts=self.net_detector_ukrts,
            fknee_mhz=self.fknee_mhz,
            fmin_hz=self.fmin_hz,
            alpha=self.alpha,
            sampling_rate_hz=self.sampling_rate_hz,
            bandwidth_ghz=self.bandwidth_ghz,
            bandcenter_ghz=self.bandcenter_ghz,
        )


@dataclass
class InstrumentInfo:
    name: str = ""
    boresight_rotangle_rad: float = 0.0
    spin_boresight_angle_rad: float = 0.0
    spin_rotangle_rad: float = 0.0
    bore2spin_quat = np.array([0.0, 0.0, 0.0, 1.0])
    hwp_rpm: float = 0.0
    number_of_channels: int = 0
    channel_names: list[str] = field(default_factory=list)
    channel_objs: list[UUID] = field(default_factory=list)
    wafer_names: list[str] = field(default_factory=list)
    wafer_space_cm: float = 0.0

    """A data class representing a LiteBIRD instrument configuration.

    This class stores metadata about the overall instrument, including
    boresight orientation, spin geometry, channel and wafer associations, and
    HWP rotation parameters.

    Initialization Methods:
        - Direct instantiation (with keyword arguments)
        - :meth:`.from_dict`: construct from a dictionary (e.g., loaded from YAML/JSON)
        - :meth:`.from_imo`: construct from an IMO object and URL/UUID

    Post-initialization Behavior:
        - The quaternion from boresight to spin frame (`bore2spin_quat`) is computed
          from three rotation angles using `__compute_bore2spin_quat__`, and then
          normalized with `normalize_time_dependent_quaternion`.
        - Any entries in `channel_objs` that are strings or URLs are converted to UUIDs.

    Quaternion Construction:
        The rotation from boresight to spin frame is computed as:
        1. A rotation around the Z-axis by `boresight_rotangle_rad`
        2. A rotation around the Y-axis by `spin_boresight_angle_rad`
        3. A rotation around the Z-axis by `spin_rotangle_rad`

    Args:
        name (str): Name of the instrument. Default is an empty string.

        boresight_rotangle_rad (float): Initial rotation around the boresight Z-axis, in radians.
            Default is 0.0.

        spin_boresight_angle_rad (float): Angle between the boresight and spin axis, in radians.
            Default is 0.0.

        spin_rotangle_rad (float): Initial spin phase angle around the spin axis, in radians.
            Default is 0.0.

        bore2spin_quat (np.ndarray): Quaternion representing the full rotation from the boresight
            frame to the spin frame. Automatically computed and normalized in `__post_init__`.

        hwp_rpm (float): Rotation speed of the Half-Wave Plate, in revolutions per minute. Default is 0.0.

        number_of_channels (int): Number of frequency channels in the instrument. Default is 0.

        channel_names (list[str]): List of frequency channel names. Default is an empty list.

        channel_objs (list[UUID]): List of references to channel definitions (UUIDs or convertible strings).
            Automatically converted to UUIDs. Default is an empty list.

        wafer_names (list[str]): List of wafer names used in the instrument. Default is an empty list.

        wafer_space_cm (float): Spacing between wafers, in centimeters. Default is 0.0.

    Methods:
        from_dict(dictionary):
            Construct an `InstrumentInfo` object from a configuration dictionary.

        from_imo(imo, url):
            Query the IMO using a UUID or URL and construct an `InstrumentInfo` from the result.
    """

    def __post_init__(self):
        self.bore2spin_quat = normalize_time_dependent_quaternion(
            self.__compute_bore2spin_quat__()
        )

        for det_idx, det_obj in enumerate(self.channel_objs):
            if not isinstance(det_obj, UUID):
                if "/" in det_obj:
                    cur_uuid = url_to_uuid(det_obj)
                else:
                    cur_uuid = UUID(det_obj)

                self.channel_objs[det_idx] = cur_uuid

    def __compute_bore2spin_quat__(self):
        quat = np.array(quat_rotation_z(self.boresight_rotangle_rad))
        quat_left_multiply(quat, *quat_rotation_y(self.spin_boresight_angle_rad))
        quat_left_multiply(quat, *quat_rotation_z(self.spin_rotangle_rad))
        return quat

    @staticmethod
    def from_dict(dictionary: dict[str, Any]):
        name = dictionary.get("name", "mock-instrument")
        boresight_rotangle_rad = np.deg2rad(
            dictionary.get("boresight_rotangle_deg", 0.0)
        )
        spin_boresight_angle_rad = np.deg2rad(
            dictionary.get("spin_boresight_angle_deg", 0.0)
        )
        spin_rotangle_rad = np.deg2rad(dictionary.get("spin_rotangle_deg", 0.0))
        hwp_rpm = dictionary.get("hwp_rpm", 0.0)
        number_of_channels = dictionary.get("number_of_channels", 0)
        channel_names = dictionary.get("channel_names", [])
        channel_objs = dictionary.get("channel_objs", [])
        wafer_names = dictionary.get("wafers", [])
        wafer_space_cm = dictionary.get("waferspace", 0.0)
        return InstrumentInfo(
            name=name,
            boresight_rotangle_rad=boresight_rotangle_rad,
            spin_boresight_angle_rad=spin_boresight_angle_rad,
            spin_rotangle_rad=spin_rotangle_rad,
            hwp_rpm=hwp_rpm,
            number_of_channels=number_of_channels,
            channel_names=channel_names,
            channel_objs=channel_objs,
            wafer_names=wafer_names,
            wafer_space_cm=wafer_space_cm,
        )

    @staticmethod
    def from_imo(imo: Imo, url: UUID | str):
        obj = imo.query(url)
        return InstrumentInfo.from_dict(obj.metadata)


################################################################################


def detector_list_from_parameters(
    imo: Imo, definition_list: list[Any]
) -> list[DetectorInfo]:
    result = []

    for det_def in definition_list:
        assert isinstance(det_def, dict)
        det = DetectorInfo()

        if "channel_info_obj" in det_def:
            ch = FreqChannelInfo.from_imo(imo, det_def["channel_info_obj"])

            use_only_one_boresight_detector = det_def.get(
                "use_only_one_boresight_detector", False
            )

            if use_only_one_boresight_detector:
                result.append(
                    ch.get_boresight_detector(
                        name=det_def.get("detector_name", "mock_boresight")
                    )
                )
                continue

            ch.number_of_detectors = det_def.get(
                "num_of_detectors_from_channel", ch.number_of_detectors
            )

            result += [
                DetectorInfo.from_imo(imo, url)
                for url in ch.detector_objs[: ch.number_of_detectors]
            ]
            continue

        if "detector_info_obj" in det_def:
            det = DetectorInfo.from_imo(imo, det_def["detector_info_obj"])
        else:
            det = DetectorInfo()

        for param in fields(DetectorInfo):
            if param.name in det_def:
                setattr(det, param.name, det_def[param.name])

        result.append(det)

    return result
