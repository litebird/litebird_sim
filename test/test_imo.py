# -*- encoding: utf-8 -*-

from pathlib import Path
from uuid import UUID

import pytest

import litebird_sim as lbs


def load_test_imo():
    curpath = Path(__file__).parent
    return lbs.Imo(flatfile_location=curpath / "test_imo")


def test_imo_key_errors():
    imo = load_test_imo()

    with pytest.raises(KeyError):
        imo.query("/format_specs/aaaaaaaa-bbbb-cccc-dddd-eeeeeeffffff")

    with pytest.raises(KeyError):
        imo.query("/entities/aaaaaaaa-bbbb-cccc-dddd-eeeeeeffffff")

    with pytest.raises(KeyError):
        imo.query("/quantities/aaaaaaaa-bbbb-cccc-dddd-eeeeeeffffff")

    with pytest.raises(KeyError):
        imo.query("/data_files/aaaaaaaa-bbbb-cccc-dddd-eeeeeeffffff")

    with pytest.raises(KeyError):
        imo.query("/UNKNOWN_TAG/instrument/beams/horn01/horn01_grasp")

    with pytest.raises(KeyError):
        imo.query("/1.0/WRONG/PATH/horn01_grasp")

    with pytest.raises(KeyError):
        imo.query("/1.0/instrument/beams/horn01/UNKNOWN_QUANTITY")

    with pytest.raises(KeyError):
        # This might look correct, but quantity "horn01_synth" has no
        # data files in release 1.0
        imo.query("/1.0/instrument/beams/horn01/horn01_synth")


def test_imo_query_uuid():
    imo = load_test_imo()

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


def test_imo_get_queried_objects():
    imo = load_test_imo()

    entity_uuid = UUID("dd32cb51-f7d5-4c03-bf47-766ce87dc3ba")
    _ = imo.query(f"/entities/{entity_uuid}")

    quantity_uuid = UUID("e9916db9-a234-4921-adfd-6c3bb4f816e9")
    _ = imo.query(f"/quantities/{quantity_uuid}")

    # This is not being tracked…
    data_file_uuid = UUID("a6dd07ee-9721-4453-abb1-e58aa53a9c01")
    _ = imo.query(f"/data_files/{data_file_uuid}", track=False)

    # …but this will be
    data_file_uuid = UUID("37bb70e4-29b2-4657-ba0b-4ccefbc5ae36")
    _ = imo.query(f"/data_files/{data_file_uuid}")

    release_data_file_uuid = UUID("bd8e16eb-2e9d-46dd-a971-f446e953b9dc")
    _ = imo.query("/1.0/instrument/beams/horn01/horn01_grasp")

    assert tuple(imo.get_queried_entities()) == (entity_uuid,)
    assert tuple(imo.get_queried_quantities()) == (quantity_uuid,)

    queried_files = imo.get_queried_data_files()
    assert len(queried_files) == 2
    assert data_file_uuid in queried_files
    assert release_data_file_uuid in queried_files


def test_imo_query_release():
    imo = load_test_imo()

    uuid = UUID("bd8e16eb-2e9d-46dd-a971-f446e953b9dc")
    data_file = imo.query("/releases/1.0/instrument/beams/horn01/horn01_grasp")
    assert data_file.uuid == uuid

    data_file = imo.query("/1.0/instrument/beams/horn01/horn01_grasp")
    assert data_file.uuid == uuid


def test_imo_entry_hierarchy():
    imo = load_test_imo()

    # This is the "beams" object
    uuid = UUID("04c53542-e8a8-421f-aa3c-201abba1575d")
    child_entity = imo.query_entity(uuid)
    assert child_entity.parent == UUID("2180affe-f9c3-4048-a407-6bd4d3ad71e5")


def test_open_data_file():
    imo = load_test_imo()

    data_file = imo.query_data_file(UUID("37bb70e4-29b2-4657-ba0b-4ccefbc5ae36"))
    with imo.open_data_file(data_file) as f:
        import json

        data = json.load(f)
        assert "test_field" in data
        assert data["test_field"] == 10.0
