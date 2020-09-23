# -*- encoding: utf-8 -*-

from pathlib import Path
from uuid import UUID

import numpy as np
import tomlkit

import litebird_sim as lbs


def check_detector(det):
    assert isinstance(det, lbs.DetectorInfo)
    assert det.name == "foo"
    assert det.wafer == "bar"
    assert det.pixel == 10
    assert det.pixtype == "def"
    assert det.channel == "ghi"
    assert det.sampling_rate_hz == 12.0
    assert det.fwhm_arcmin == 34.0
    assert det.ellipticity == 56.0
    assert det.net_ukrts == 78.0
    assert det.fknee_mhz == 90.0
    assert det.fmin_hz == 98.0
    assert det.alpha == 76.0
    assert det.pol == "jkl"
    assert det.orient == "mno"

    # The quaternion should be always normalized
    assert np.allclose(det.quat, [0.00000000, 0.26726124, 0.53452248, 0.80178373])


def test_detector_from_dict():
    det = lbs.DetectorInfo.from_dict(
        {
            "name": "foo",
            "wafer": "bar",
            "pixel": 10,
            "pixtype": "def",
            "channel": "ghi",
            "sampling_rate_hz": 12.0,
            "fwhm_arcmin": 34.0,
            "ellipticity": 56.0,
            "net_ukrts": 78.0,
            "fknee_mhz": 90.0,
            "fmin_hz": 98.0,
            "alpha": 76.0,
            "pol": "jkl",
            "orient": "mno",
            "quat": [0.0, 1.0, 2.0, 3.0],
        }
    )

    check_detector(det)


def test_detector_from_toml():
    doc = tomlkit.parse(
        """
[my_detector]
name = "foo"
wafer = "bar"
pixel = 10
pixtype = "def"
channel = "ghi"
sampling_rate_hz = 12.0
fwhm_arcmin = 34.0
ellipticity = 56.0
net_ukrts = 78.0
fknee_mhz = 90.0
fmin_hz = 98.0
alpha = 76.0
pol = "jkl"
orient = "mno"
quat = [0.0, 1.0, 2.0, 3.0]
"""
    )

    det = lbs.DetectorInfo.from_dict(doc["my_detector"])

    check_detector(det)


def load_mock_imo():
    curpath = Path(__file__).parent
    return lbs.Imo(flatfile_location=curpath / "mock_imo")


def test_detector_from_imo():
    imo = load_mock_imo()

    uuid = UUID("78fe75f1-a011-44b6-86dd-445dc9634416")
    det = lbs.DetectorInfo.from_imo(imo, uuid)

    check_detector(det)
