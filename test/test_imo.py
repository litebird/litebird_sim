# -*- encoding: utf-8 -*-

from uuid import UUID
from pathlib import Path
import litebird_sim as lbs

import pytest


def load_mock_imo():
    curpath = Path(__file__).parent
    return lbs.ImoFlatFile(curpath / "mock_imo")


def test_imo_key_errors():
    imo = load_mock_imo()

    with pytest.raises(KeyError):
        imo.query("/format_specs/aaaaaaaa-bbbb-cccc-dddd-eeeeeeffffff")
        imo.query("/entities/aaaaaaaa-bbbb-cccc-dddd-eeeeeeffffff")
        imo.query("/quantities/aaaaaaaa-bbbb-cccc-dddd-eeeeeeffffff")
        imo.query("/data_files/aaaaaaaa-bbbb-cccc-dddd-eeeeeeffffff")

        imo.query("/UNKNOWN_TAG/focal_plane/beams/horn01/horn01_grasp")
        imo.query("/1.0/WRONG/PATH/horn01_grasp")
        imo.query("/1.0/focal_plane/beams/horn01/UNKNOWN_QUANTITY")

        # This might look correct, but quantity "horn01_synth" has no
        # data files in release 1.0
        imo.query("/1.0/focal_plane/beams/horn01/horn01_synth")


def test_imo_query_uuid():
    imo = load_mock_imo()

    uuid = UUID("dd32cb51-f7d5-4c03-bf47-766ce87dc3ba")
    entity = imo.query(f"/entities/{uuid}")
    assert entity.uuid == uuid

    uuid = UUID("e9916db9-a234-4921-adfd-6c3bb4f816e9")
    quantity = imo.query(f"/quantities/{uuid}")
    assert quantity.uuid == uuid

    uuid = UUID("37bb70e4-29b2-4657-ba0b-4ccefbc5ae36")
    data_file = imo.query(f"/data_files/{uuid}")
    assert data_file.uuid == uuid
    assert data_file.metadata["ellipticity"] == 0.0
    assert data_file.metadata["fwhm_deg"] == 1.0

    data_file = imo.query(uuid)
    assert data_file.uuid == uuid


def test_imo_query_release():
    imo = load_mock_imo()

    data_file = imo.query("/1.0/focal_plane/beams/horn01/horn01_grasp")
    assert data_file.uuid == UUID("bd8e16eb-2e9d-46dd-a971-f446e953b9dc")
