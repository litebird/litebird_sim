# -*- encoding: utf-8 -*-

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Union, List

from uuid import UUID
import numpy as np

import astropy.time as astrotime
from .imo import Imo
from .quaternions import quat_rotation_y, quat_rotation_z, quat_left_multiply
from .scanning import RotQuaternion

from .bandpasses import BandPassInfo


def url_to_uuid(url: str) -> UUID:
    return UUID([x for x in url.split("/") if x != ""][-1])


def normalize_time_dependent_quaternion(
    quat: Union[np.ndarray, RotQuaternion],
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
    """A class wrapping the basic information about a detector.

    This is a data class that encodes the basic properties of a
    LiteBIRD detector. It can be initialized in three ways:

    - Through the default constructor; all its parameters are
      optional, but you probably want to specify at least `name` and
      `sampling_rate_hz`::

          det = DetectorInfo(name="dummy", sampling_rate_hz=10.0)

    - Through the class method :meth:`.from_dict`, which takes a
      dictionary as input.

    - Through the class method :meth:`.from_imo`, which reads the
      definition of a detector from the LiteBIRD Instrument Model (see
      the :class:`.Imo` class).

    Args:

        - name (str): the name of the detector; the default is the
             empty string

        - wafer (Union[str, None]): The name of the wafer hosting the
             detector, e.g. ``H00``, ``L07``, etc. The default is None

        - pixel (Union[int, None]): The number of the pixel within the
             wafer. The default is None

        - pixtype (Union[str, None]): The type of the pixel, e.g.,
             ``HP1``, ``LP3``, etc. The default is None

        - channel (Union[str, None]): The channel. The default is None

        - sampling_rate_hz (float): The sampling rate of the ADC
             associated with this detector. The default is 0.0

        - fwhm_arcmin: (float): The Full Width Half Maximum of the
             radiation pattern associated with the detector, in
             arcminutes. The default is 0.0

        - ellipticity (float): The ellipticity of the radiation
             pattern associated with the detector. The default is 0.0

        - net_ukrts (float): The noise equivalent temperature of the
             signal produced by the detector in nominal conditions,
             expressed in μK/√s. This is the noise per sample to be
             used when adding noise to the timelines. The default is 0.0

        - pol_sensitivity_ukarcmin (float): The detector sensitivity
            in microK_arcmin. This value considers the effect of cosmic ray loss,
            repointing maneuvers, etc., and other issues that cause loss of
            integration time. Therefore, it should **not** be used with the
            functions that add noise to the timelines. The default is 0.0

        - bandcenter_ghz (float): The center frequency of the
             detector, in GHz. The default is 0.0

        - bandwidth_ghz (float): The bandwidth of the detector, in
             GHz. The default is 0.0

        - band_freqs_ghz (float array): band sampled frequencies, in GHz.
             The default is None

        - band_weights (float array): band profile. The default is None

        - fknee_mhz (float): The knee frequency between the 1/f and
             the white noise components in nominal conditions, in mHz.
             The default is 0.0

        - fmin_hz (float): The minimum frequency of the noise when
             producing synthetic noise, in Hz. The default is 0.0

        - alpha (float): The slope of the 1/f component of the noise
             in nominal conditions. The default is 0.0

        - pol (Union[str, None]): The polarization of the detector
             (``T``/``B``). The default is None

        - orient (Union[str, None]): The orientation of the detector
             (``Q``/``U``). The default is None

        - quat (:class:`.TimeDependentQuaternion`): The quaternion
             expressing the rotation from the detector reference frame
             to the boresight reference frame. The default is no
             rotation at all, i.e., the detector is aligned with the
             boresight direction.

    """

    name: str = ""
    wafer: Union[str, None] = None
    pixel: Union[int, None] = None
    pixtype: Union[str, None] = None
    channel: Union[str, None] = None
    sampling_rate_hz: float = 0.0
    fwhm_arcmin: float = 0.0
    ellipticity: float = 0.0
    bandcenter_ghz: float = 0.0
    bandwidth_ghz: float = 0.0
    band_freqs_ghz: Union[None, np.ndarray] = None
    band_weights: Union[None, np.ndarray] = None
    net_ukrts: float = 0.0
    pol_sensitivity_ukarcmin: float = 0.0
    fknee_mhz: float = 0.0
    fmin_hz: float = 0.0
    alpha: float = 0.0
    bandcenter_ghz: float = 0.0
    bandwidth_ghz: float = 0.0
    pol: Union[str, None] = None
    orient: Union[str, None] = None
    quat: Any = None

    def __post_init__(self):
        if self.quat is None:
            self.quat = (0.0, 0.0, 0.0, 1.0)

        self.quat = normalize_time_dependent_quaternion(self.quat)

        if isinstance(self.band_freqs_ghz, np.ndarray):
            assert len(self.band_freqs_ghz) == len(self.band_weights)

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]):
        """Create a detector from the contents of a dictionary

        The parameter `dictionary` must contain one key for each of
        the fields in this dataclass:

        - ``name``
        - ``wafer``
        - ``pixel``
        - ``pixtype``
        - ``channel``
        - ``bandcenter_ghz``
        - ``bandwidth_ghz``
        - ``band_freqs_ghz``
        - ``band_weights``
        - ``sampling_rate_hz``
        - ``fwhm_arcmin``
        - ``ellipticity``
        - ``net_ukrts``
        - ``pol_sensitivity_ukarcmin``
        - ``fknee_mhz``
        - ``fmin_hz``
        - ``alpha``
        - ``pol``
        - ``orient``
        - ``quat``

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
    def from_imo(imo: Imo, url: Union[UUID, str]):
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
    channel: Union[str, None] = None
    bandwidth_ghz: float = 0.0
    band_freqs_ghz: Union[None, np.ndarray] = None
    band_weights: Union[None, np.ndarray] = None
    net_detector_ukrts: float = 0.0
    net_channel_ukrts: float = 0.0
    pol_sensitivity_channel_ukarcmin: float = 0.0
    sampling_rate_hz: float = 0.0
    fwhm_arcmin: float = 0.0
    fknee_mhz: float = 0.0
    fmin_hz: float = 1e-5
    alpha: float = 1.0
    number_of_detectors: int = 0
    detector_names: List[str] = field(default_factory=list)
    detector_objs: List[UUID] = field(default_factory=list)

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
    def from_dict(dictionary: Dict[str, Any]):
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
    def from_imo(imo: Imo, url: Union[UUID, str]):
        obj = imo.query(url)
        return FreqChannelInfo.from_dict(obj.metadata)

    def get_boresight_detector(self, name="mock") -> DetectorInfo:
        return DetectorInfo(
            name=name,
            channel=self.channel,
            fwhm_arcmin=self.fwhm_arcmin,
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
    channel_names: List[str] = field(default_factory=list)
    channel_objs: List[UUID] = field(default_factory=list)
    wafer_names: List[str] = field(default_factory=list)
    wafer_space_cm: float = 0.0

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
    def from_dict(dictionary: Dict[str, Any]):
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
    def from_imo(imo: Imo, url: Union[UUID, str]):
        obj = imo.query(url)
        return InstrumentInfo.from_dict(obj.metadata)


################################################################################


def detector_list_from_parameters(
    imo: Imo, definition_list: List[Any]
) -> List[DetectorInfo]:
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
