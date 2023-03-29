# -*- encoding: utf-8 -*-

import litebird_sim as lbs


def test_imo():
    imo = lbs.Imo()

    # data_file = imo.query_data_file("bfe9a79cbfa142a7bdf4a0ae4d1f8f0b")
    # assert data_file.name == "instrument_info"

    data_file = imo.query("/releases/vPTEP/satellite/LFT/L1-040/000_000_003_QA_040_T")
    assert data_file.name == "detector_info"
