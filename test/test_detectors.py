# -*- encoding: utf-8 -*-

from pathlib import Path
from dataclasses import fields
from uuid import UUID

import numpy as np
import tomlkit

import litebird_sim as lbs


def check_detector(det):
    assert isinstance(det, lbs.DetectorInfo)
    assert det.name == "foo1"
    assert det.wafer == "bar"
    assert det.pixel == 10
    assert det.pixtype == "def"
    assert det.channel == "ghi"
    assert det.bandcenter_ghz == 65.0
    assert det.bandwidth_ghz == 14.0
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
    assert isinstance(det.quat, lbs.RotQuaternion)
    np.testing.assert_allclose(
        actual=det.quat.quats,
        desired=[[0.00000000, 0.26726124, 0.53452248, 0.80178373]],
    )


def test_detector_from_dict():
    det = lbs.DetectorInfo.from_dict(
        {
            "name": "foo1",
            "wafer": "bar",
            "pixel": 10,
            "pixtype": "def",
            "channel": "ghi",
            "bandcenter_ghz": 65.0,
            "bandwidth_ghz": 14.0,
            "sampling_rate_hz": 12.0,
            "fwhm_arcmin": 34.0,
            "ellipticity": 56.0,
            "net_ukrts": 78.0,
            "fknee_mhz": 90.0,
            "fmin_hz": 98.0,
            "alpha": 76.0,
            "pol": "jkl",
            "orient": "mno",
            "quat": {
                "quats": [0.00000000, 0.26726124, 0.53452248, 0.80178373],
                "start_time": None,
                "sampling_rate_hz": None,
            },
        }
    )

    check_detector(det)


def test_detector_from_toml():
    doc = tomlkit.parse(
        """
[my_detector]
name = "foo1"
wafer = "bar"
pixel = 10
pixtype = "def"
channel = "ghi"
bandcenter_ghz = 65.0
bandwidth_ghz = 14.0
sampling_rate_hz = 12.0
fwhm_arcmin = 34.0
ellipticity = 56.0
net_ukrts = 78.0
fknee_mhz = 90.0
fmin_hz = 98.0
alpha = 76.0
pol = "jkl"
orient = "mno"
quat = [0.00000000, 0.26726124, 0.53452248, 0.80178373]
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


def check_freq_channel(ch: lbs.FreqChannelInfo):
    assert isinstance(ch, lbs.FreqChannelInfo)
    assert ch.channel == "65 GHz"
    assert ch.bandcenter_ghz == 65.0
    assert ch.bandwidth_ghz == 14.0
    assert ch.net_detector_ukrts == 300.0
    assert ch.net_channel_ukrts == 150.0
    assert ch.pol_sensitivity_channel_ukarcmin == 200.0
    assert ch.fwhm_arcmin == 58.0
    assert ch.fknee_mhz == 25.0
    assert ch.fmin_hz == 1.0
    assert ch.alpha == 2.0
    assert ch.number_of_detectors == 4

    assert len(ch.detector_names) == 4
    assert ch.detector_names[0] == "foo1"
    assert ch.detector_names[1] == "foo2"
    assert ch.detector_names[2] == "foo3"
    assert ch.detector_names[3] == "foo4"

    assert len(ch.detector_objs) == 4
    assert ch.detector_objs[0] == UUID("78fe75f1-a011-44b6-86dd-445dc9634416")
    assert ch.detector_objs[1] == UUID("58db7186-3ee7-49e5-8605-4a5ed084f99e")
    assert ch.detector_objs[2] == UUID("8f23f893-c94f-4d24-afcd-4c6633eabb65")
    assert ch.detector_objs[3] == UUID("fde14ce8-6831-48d7-8410-81051a344fea")


def test_freq_channel_creation():
    ch = lbs.FreqChannelInfo(bandcenter_ghz=60)

    assert ch.bandcenter_ghz == 60
    assert ch.channel == "60.0 GHz"
    assert ch.number_of_detectors == 1

    ch = lbs.FreqChannelInfo(bandcenter_ghz=50, number_of_detectors=5)

    assert ch.bandcenter_ghz == 50
    assert ch.number_of_detectors == 5
    assert len(ch.detector_names) == 5

    ch = lbs.FreqChannelInfo(bandcenter_ghz=150, detector_names=["a", "b"])

    assert ch.bandcenter_ghz == 150
    assert ch.number_of_detectors == 2
    assert len(ch.detector_names) == 2
    assert ch.detector_names[0] == "a"
    assert ch.detector_names[1] == "b"


def test_freq_channel_from_imo():
    imo = load_mock_imo()

    uuid = UUID("ff087ba3-d973-4dc3-b72b-b68abb979a90")
    ch = lbs.FreqChannelInfo.from_imo(imo, uuid)

    check_freq_channel(ch)

    det = ch.get_boresight_detector(name="test")

    assert det.name == "test"
    assert det.wafer is None
    assert det.pixel is None
    assert det.pixtype is None
    assert det.channel == "65 GHz"
    assert det.sampling_rate_hz == 12.0
    assert det.fwhm_arcmin == 58.0
    assert det.ellipticity == 0.0
    assert det.net_ukrts == 300.0
    assert det.fknee_mhz == 25.0
    assert det.fmin_hz == 1.0
    assert det.alpha == 2.0
    assert det.pol is None
    assert det.orient is None

    assert isinstance(det.quat, lbs.RotQuaternion)
    # The quaternion should be always normalized
    np.testing.assert_allclose(actual=det.quat.quats, desired=[[0.0, 0.0, 0.0, 1.0]])


def test_freq_channel_noise():
    # Detector → channel
    ch = lbs.FreqChannelInfo(
        bandcenter_ghz=60, net_detector_ukrts=10.0, number_of_detectors=4
    )
    assert ch.net_channel_ukrts == 5.0

    # Channel → detector
    ch = lbs.FreqChannelInfo(
        bandcenter_ghz=60, net_channel_ukrts=10.0, number_of_detectors=4
    )
    assert ch.net_detector_ukrts == 20


def check_instrument(instr: lbs.InstrumentInfo):
    assert isinstance(instr, lbs.InstrumentInfo)

    assert instr.number_of_channels == 1
    assert instr.hwp_rpm == 10.0
    assert np.allclose(instr.spin_boresight_angle_rad, np.deg2rad(10.0))
    assert np.allclose(instr.boresight_rotangle_rad, np.deg2rad(5.0))
    assert np.allclose(instr.spin_rotangle_rad, np.deg2rad(15.0))
    assert instr.wafer_space_cm == 1.0

    assert len(instr.channel_names) == 1
    assert instr.channel_names[0] == "65 GHz"

    assert len(instr.channel_objs) == 1
    assert instr.channel_objs[0] == UUID("ff087ba3-d973-4dc3-b72b-b68abb979a90")

    assert len(instr.wafer_names) == 2
    assert instr.wafer_names[0] == "A"
    assert instr.wafer_names[1] == "B"


def test_instrument_creation():
    instr = lbs.InstrumentInfo(
        name="test",
        boresight_rotangle_rad=np.deg2rad(10.0),
        spin_boresight_angle_rad=np.deg2rad(15.0),
        spin_rotangle_rad=np.deg2rad(20.0),
    )

    assert instr.name == "test"
    assert np.allclose(instr.boresight_rotangle_rad, np.deg2rad(10.0))
    assert np.allclose(instr.spin_boresight_angle_rad, np.deg2rad(15.0))
    assert np.allclose(instr.spin_rotangle_rad, np.deg2rad(20.0))

    assert np.allclose(
        instr.bore2spin_quat.quats,
        np.array([[-0.01137611, 0.1300295, 0.25660481, 0.9576622]]),
    )


def test_instrument_from_imo():
    imo = load_mock_imo()

    uuid = UUID("f58e1c9b-d2fe-4db4-9ad5-65f644f12db7")
    instr = lbs.InstrumentInfo.from_imo(imo, uuid)

    check_instrument(instr)


def test_det_list_from_imo():
    with (Path(__file__).parent / ".." / "docs" / "det_list1.toml").open("rt") as inpf:
        toml_contents = "".join(inpf.readlines())

    doc = tomlkit.parse(toml_contents)

    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)
    det_list = lbs.detector_list_from_parameters(imo, doc["detectors"])

    assert len(det_list) == 6

    assert isinstance(det_list[0], lbs.DetectorInfo)
    assert det_list[0].name == "000_000_003_QA_040_T"

    # Just check a few fields, don't worry checking them all
    assert det_list[0].bandcenter_ghz == 40.0
    assert det_list[0].bandwidth_ghz == 12.0
    assert det_list[0].channel == "L1-040"
    assert det_list[0].ellipticity == 0.0
    assert det_list[0].fknee_mhz == 20.0

    # The second and third detectors should have been created from a
    # channel_info
    assert isinstance(det_list[1], lbs.DetectorInfo)
    assert det_list[1].name == "001_002_030_00A_140_T"
    assert det_list[1].bandcenter_ghz == 140.0
    assert det_list[1].bandwidth_ghz == 42.0

    assert det_list[2].name == "001_002_030_00A_140_B"
    assert det_list[2].bandcenter_ghz == 140.0
    assert det_list[2].bandwidth_ghz == 42.0

    # The fourth detector is a mock detector representative of a
    # frequency channel and aligned with the boresight
    assert det_list[3].name == "foo_boresight"
    assert det_list[3].bandcenter_ghz == 140.0
    np.testing.assert_allclose(det_list[3].quat.quats, np.array([[0, 0, 0, 1]]))

    # The fifth detector must be a Planck-like radiometer
    assert det_list[4].name == "planck30GHz"
    assert det_list[4].channel == "30 GHz"
    np.testing.assert_allclose(det_list[4].fwhm_arcmin, 33.10)
    np.testing.assert_allclose(det_list[4].fknee_mhz, 113.9)
    np.testing.assert_allclose(det_list[4].bandwidth_ghz, 9.89)
    np.testing.assert_allclose(det_list[4].bandcenter_ghz, 28.4)
    np.testing.assert_allclose(det_list[4].sampling_rate_hz, 32.5)

    # Detector det_list[5] must be the same as det_list[0], but with a
    # different sampling rate
    np.testing.assert_allclose(det_list[0].quat.quats, det_list[5].quat.quats)
    for cur_field in fields(lbs.DetectorInfo):
        if cur_field.name in ["sampling_rate_hz", "quat"]:
            continue

        val = getattr(det_list[5], cur_field.name)
        try:
            assert val == getattr(det_list[0], cur_field.name)
        except ValueError:
            # A ValueError exception usually means that `val` is
            # a numpy array. In this case, we perform the comparison
            # element by element.
            assert np.all(val == getattr(det_list[0], cur_field.name))

    assert det_list[5].sampling_rate_hz == 1.0
