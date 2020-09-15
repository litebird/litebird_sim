# -*- encoding: utf-8 -*-

from typing import Any, Dict, Union

from uuid import UUID
import numpy as np

from .imo import Imo


class Detector:
    """A class wrapping the basic information about a detector.

    This is a data class that encodes the basic properties of a
    LiteBIRD detector. It can be initialized in three ways:

    - Through the default constructor; all its parameters are
      optional, but you probably want to specify at least `name` and
      `sampling_rate_hz`::

          det = Detector(name="dummy", sampling_rate_hz=10.0)

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

        - fwhm_arcmin: float): The Full Width Half Maximum of the
             radiation pattern associated with the detector, in
             arcminutes. The default is 0.0

        - ellipticity (float): The ellipticity of the radiation
             pattern associated with the detector. The default is 0.0

        - net_ukrts (float): The noise equivalent temperature of the
             signal produced by the detector in nominal conditions,
             expressed in μK/√s.. The default is 0.0

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

    def __init__(
        self,
        name: str = "",
        wafer: Union[str, None] = None,
        pixel: Union[int, None] = None,
        pixtype: Union[str, None] = None,
        channel: Union[str, None] = None,
        sampling_rate_hz: float = 0.0,
        fwhm_arcmin: float = 0.0,
        ellipticity: float = 0.0,
        net_ukrts: float = 0.0,
        fknee_mhz: float = 0.0,
        fmin_hz: float = 0.0,
        alpha: float = 0.0,
        pol: Union[str, None] = None,
        orient: Union[str, None] = None,
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
    ):
        self.name = name
        self.wafer = wafer
        self.pixel = int(pixel) if pixel is not None else None
        self.pixtype = pixtype
        self.channel = channel
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.fwhm_arcmin = float(fwhm_arcmin)
        self.ellipticity = float(ellipticity)
        self.net_ukrts = float(net_ukrts)
        self.fknee_mhz = float(fknee_mhz)
        self.fmin_hz = float(fmin_hz)
        self.alpha = float(alpha)
        self.pol = pol
        self.orient = orient

        assert len(quat) == 4
        self.quat = np.array([float(x) for x in quat])

    def __repr__(self):
        return (
            'Detector(name="{name}", channel="{channel}", '
            + 'pol="{pol}", orient="{orient}")'
        ).format(name=self.name, channel=self.channel, pol=self.pol, orient=self.orient)

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]):
        """Create a detector from the contents of a dictionary

        The parameter `dictionary` must contain the following keys,
        which correspond to the parameters used in the constructor:

        - ``name``
        - ``wafer``
        - ``pixel``
        - ``pixtype``
        - ``channel``
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
        return Detector(
            name=dictionary["name"],
            wafer=dictionary["wafer"],
            pixel=dictionary["pixel"],
            pixtype=dictionary["pixtype"],
            channel=dictionary["channel"],
            sampling_rate_hz=float(dictionary["sampling_rate_hz"]),
            fwhm_arcmin=float(dictionary["fwhm_arcmin"]),
            ellipticity=float(dictionary["ellipticity"]),
            net_ukrts=float(dictionary["net_ukrts"]),
            fknee_mhz=float(dictionary["fknee_mhz"]),
            fmin_hz=float(dictionary["fmin_hz"]),
            alpha=float(dictionary["alpha"]),
            pol=dictionary["pol"],
            orient=dictionary["orient"],
            quat=np.array([float(x) for x in dictionary["quat"]]),
        )

    @staticmethod
    def from_imo(imo: Imo, url: Union[UUID, str]):
        """Create a `Detector` object from a definition in the IMO

        The `url` must either specify a UUID or a full URL to the
        object.

        Example::

            import litebird_sim as lbs
            imo = Imo()
            det = Detector.from_imo(
                imo=imo,
                url="/releases/v1.0/satellite/LFT/L03_006_QB_040T",
            )

        """
        obj = imo.query(url)
        return Detector.from_dict(obj.metadata)

    def to_dict(self):
        """Conver to a dictionary
        """
        return dict(
            name=self.name,
            wafer=self.wafer,
            pixel=self.pixel,
            pixtype=self.pixtype,
            channel=self.channel,
            sampling_rate_hz=float(self.sampling_rate_hz),
            fwhm_arcmin=float(self.fwhm_arcmin),
            ellipticity=float(self.ellipticity),
            net_ukrts=float(self.net_ukrts),
            fknee_mhz=float(self.fknee_mhz),
            fmin_hz=float(self.fmin_hz),
            alpha=float(self.alpha),
            pol=self.pol,
            orient=self.orient,
            quat=self.quat,
        )
