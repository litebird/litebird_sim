# -*- encoding: utf-8 -*-
from uuid import UUID

import litebird_sim as lbs


def test_imo():
    imo = lbs.Imo(flatfile_location=lbs.PTEP_IMO_LOCATION)

    # data_file = imo.query_data_file("bfe9a79cbfa142a7bdf4a0ae4d1f8f0b")
    # assert data_file.name == "instrument_info"

    data_file = imo.query(
        "/releases/vPTEP/satellite/LFT/L1-040/000_000_003_QA_040_T/detector_info"
    )
    assert data_file.name == "detector_info"
    assert data_file.uuid == UUID("34b46c91-e196-49b9-9d44-d69b4827f751")
