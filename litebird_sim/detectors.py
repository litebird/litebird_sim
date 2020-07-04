# -*- encoding: utf-8 -*-

from typing import Any, Dict

import numpy as np


class Detector:
    def __init__(
        self,
        name,
        wafer,
        pixel,
        pixtype,
        channel,
        sampling_rate_hz,
        fwhm_arcmin,
        ellipticity,
        net_ukrts,
        fknee_mhz,
        fmin_hz,
        alpha,
        pol,
        orient,
        quat,
    ):
        self.name = name
        self.wafer = wafer
        self.pixel = pixel
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


def read_detector_from_dict(dictionary: Dict[str, Any]) -> Detector:
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


def read_detector_from_imo(imo, url: str) -> Detector:
    obj = imo.query(url)
    return read_detector_from_dict(obj.metadata)
