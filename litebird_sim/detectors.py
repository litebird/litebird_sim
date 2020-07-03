# -*- encoding: utf-8 -*-

from collections import namedtuple
from typing import Any, Dict

Detector = namedtuple(
    "Detector",
    [
        "name",
        "wafer",
        "pixel",
        "pixel_type",
        "channel",
        "sampling_frequency_hz",
        "fwhm_arcmin",
        "ellipticity",
        "net_ukrts",
        "fknee_mhz",
        "fmin_hz",
        "alpha",
        "pol",
        "orientation",
        "quaternion",
    ],
)


def read_detector_from_dict(imo, dictionary: Dict[str, Any]) -> Detector:
    return Detector(
        name=dictionary["name"],
        wafer=dictionary["wafer"],
        pixel=dictionary["pixel"],
        pixel_type=dictionary["pixtype"],
        channel=dictionary["channel"],
        sampling_frequency_hz=dictionary["sampling_frequency_hz"],
        fwhm_arcmin=dictionary["fwhm_arcmin"],
        ellipticity=dictionary["ellipticity"],
        net_ukrts=dictionary["net_ukrts"],
        fknee_mhz=dictionary["fknee_mhz"],
        fmin_hz=dictionary["fmin_hz"],
        alpha=dictionary["alpha"],
        pol=dictionary["pol"],
        orientation=dictionary["orientation"],
        quaternion=dictionary["quat"],
    )


def read_detector_from_imo(imo, url: str) -> Detector:
    obj = imo.query(url)
    return read_detector_from_dict(imo, obj.metadata)
