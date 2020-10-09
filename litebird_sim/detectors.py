# -*- encoding: utf-8 -*-

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Union, List

from uuid import UUID
import numpy as np

from .imo import Imo
from .quaternions import quat_rotation_y, quat_rotation_z, quat_left_multiply


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

        - bandwidth_ghz (float): The bandwidth of the channel, in GHz

        - bandcenter_ghz (float): The central frequency of the
             channel, in GHz

        - fwhm_arcmin: float): The Full Width Half Maximum of the
             radiation pattern associated with the detector, in
             arcminutes. The default is 0.0

        - ellipticity (float): The ellipticity of the radiation
             pattern associated with the detector. The default is 0.0

        - net_ukrts (float): The noise equivalent temperature of the
             signal produced by the detector in nominal conditions,
             expressed in μK/√s.. The default is 0.0

        - bandcenter_ghz (float): The center frequency of the
             detector, in GHz. The default is 0.0

        - bandwidth_ghz (float): The bandwidth of the detector, in
             GHz. The default is 0.0

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

        - quat (np.array([0.0, 0.0, 0.0, 1.0]): The quaternion
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
    net_ukrts: float = 0.0
    fknee_mhz: float = 0.0
    fmin_hz: float = 0.0
    alpha: float = 0.0
    bandcenter_ghz: float = 0.0
    bandwidth_ghz: float = 0.0
    pol: Union[str, None] = None
    orient: Union[str, None] = None
    quat: Any = np.array([0.0, 0.0, 0.0, 1.0])

    def __post_init__(self):
        assert len(self.quat) == 4

        # Ensure that the quaternion is a NumPy array
        self.quat = np.array([float(x) for x in self.quat])

        # Normalize the quaternion
        self.quat /= np.sqrt(self.quat.dot(self.quat))

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
        - ``sampling_rate_hz``
        - ``fwhm_arcmin``
        - ``ellipticity``
        - ``net_ukrts``
        - ``fknee_mhz``
        - ``fmin_hz``
        - ``alpha``
        - ``pol``
        - ``orient``
        - ``quat``

        """
        return DetectorInfo(**dictionary)

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
                url="/releases/v1.0/satellite/LFT/L03_006_QB_040T",
            )

        """
        obj = imo.query(url)
        return DetectorInfo.from_dict(obj.metadata)


@dataclass
class FreqChannelInfo:
    bandcenter_ghz: float
    channel: Union[str, None] = None
    bandwidth_ghz: float = 0.0
    net_detector_ukrts: float = 0.0
    net_channel_ukrts: float = 0.0
    pol_sensitivity_channel_ukarcmin: float = 0.0
    sampling_rate_hz: float = 0.0
    fwhm_arcmin: float = 0.0
    fknee_mhz: float = 0.0
    fmin_hz: float = 1e-5
    alpha: float = 1.0
    number_of_detectors: Union[int, None] = None
    detector_names: Union[List[str], None] = None

    def __post_init__(self):
        if self.channel is None:
            self.channel = f"{self.bandcenter_ghz:.1f} GHz"

        if self.number_of_detectors is not None:
            # First hypothesis: we have set the number of detectors. Check
            # that this is consistent with the field "self.detector_names"

            if self.detector_names is not None:
                assert len(self.detector_names) == self.number_of_detectors
            else:
                self.detector_names = [
                    f"det{x:d}" for x in range(self.number_of_detectors)
                ]
        else:
            # Second hypothesis: the number of detectors was not set.
            # Check if we have detector names.
            if self.detector_names is not None:
                self.number_of_detectors = len(self.detector_names)
            else:
                self.number_of_detectors = 1
                self.detector_names = ["det0"]

    @staticmethod
    def from_dict(dictionary):
        return FreqChannelInfo(**dictionary)

    @staticmethod
    def from_imo(imo, objref):
        obj = imo.query(objref)
        dictcopy = dict(obj.metadata)

        # This field is in the imo but is redundant, so we do not
        # store it in this class
        if "number_of_detectors" in dictcopy:
            del dictcopy["number_of_detectors"]

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
