# -*- encoding: utf-8 -*-

from uuid import UUID

import numpy as np

import litebird_sim as lbs


def test_imo_release_access():
    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

    # data_file = imo.query_data_file("bfe9a79cbfa142a7bdf4a0ae4d1f8f0b")
    # assert data_file.name == "instrument_info"

    data_file = imo.query(
        "/releases/vPTEP/satellite/LFT/L1-040/000_000_003_QA_040_T/detector_info"
    )
    assert data_file.name == "detector_info"
    assert data_file.uuid == UUID("34b46c91-e196-49b9-9d44-d69b4827f751")


def test_detector_in_mock_imo():
    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

    detector = lbs.DetectorInfo.from_imo(
        imo, UUID("0f2fd0c7-88bf-4f0f-bad0-452ed140e9ba")
    )

    assert detector.name == "002_002_151_Q_402_B"
    assert detector.wafer == "H02"
    assert detector.pixel == 151
    assert detector.pixtype == "HP3"
    assert detector.channel == "H3-402"
    assert detector.fwhm_arcmin == 17.9
    assert detector.ellipticity == 0.0
    assert detector.bandcenter_ghz == 402.0
    assert detector.bandwidth_ghz == 92.0
    assert detector.sampling_rate_hz == 31.0
    assert np.isclose(detector.net_ukrts, 385.69)
    assert np.isclose(detector.pol_sensitivity_ukarcmin, 872.3488622519335)
    assert detector.fknee_mhz == 20.0
    assert detector.fmin_hz == 0.00001
    assert detector.alpha == 1.0
    assert detector.pol == "B"
    assert detector.orient == "Q"
    assert isinstance(detector.quat, lbs.RotQuaternion)
    assert np.allclose(detector.quat.quats, [[0.0, 0.0, 0.0, 1.0]])
    assert np.isclose(detector.pol_angle_rad, 1.5368014127353666)
