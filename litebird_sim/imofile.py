# -*- encoding: utf-8 -*-

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Union
from uuid import UUID

import yaml

from .imo import Imo

IMO_FLATFILE_SCHEMA_FILE_NAME = "schema.yaml"
IMO_FLATFILE_DATA_FILES_DIR_NAME = "data_files"
IMO_FLATFILE_FORMAT_SPEC_DIR_NAME = "format_spec"
IMO_FLATFILE_PLOT_FILES_DIR_NAME = "plot_files"


class FormatSpecification:
    def __init__(
        self,
        uuid: UUID,
        document_ref: UUID,
        title: str,
        doc_file: str,
        doc_file_name: str,
        doc_mime_type: str,
        file_mime_type: str,
    ):
        self.uuid = uuid
        self.document_ref = document_ref
        self.title = title
        self.doc_file = doc_file
        self.doc_file_name = doc_file_name
        self.doc_mime_type = doc_mime_type
        self.file_mime_type = file_mime_type


class Entity:
    def __init__(
        self, uuid: UUID, name: str, full_path: str, parent: Union[UUID, None]
    ):
        self.uuid = uuid
        self.name = name
        self.full_path = full_path
        self.parent = parent
        self.quantities = set()  # type: Set[UUID]


class Quantity:
    def __init__(
        self,
        uuid: UUID,
        name: str,
        format_spec: Union[UUID, None],
        entity: Union[UUID, None],
    ):
        self.uuid = uuid
        self.name = name
        self.format_spec = format_spec
        self.entity = entity
        self.data_files = set()  # type: Set[UUID]


class DataFile:
    def __init__(
        self,
        uuid: UUID,
        name: str,
        upload_date: Union[datetime, None],
        metadata: Dict[str, Any],
        file_data: str,
        quantity: UUID,
        spec_version: str,
        dependencies: Set[UUID],
        plot_file: str,
        plot_mime_type: str,
        comment: str,
    ):
        self.uuid = uuid
        self.name = name
        self.upload_date = upload_date
        self.metadata = metadata
        self.file_data = file_data
        self.quantity = quantity
        self.spec_version = spec_version
        self.dependencies = dependencies
        self.plot_file = plot_file
        self.plot_mime_type = plot_mime_type
        self.comment = comment


class Release:
    def __init__(
        self,
        tag: str,
        rel_date: Union[datetime, None],
        comments: str,
        data_files: Set[UUID],
    ):
        self.tag = tag
        self.rel_date = rel_date
        self.comments = comments
        self.data_files = data_files


def parse_format_spec(obj_dict: Dict[str, Any]):
    return FormatSpecification(
        uuid=UUID(obj_dict["uuid"]),
        document_ref=obj_dict.get("document_ref", ""),
        title=obj_dict.get("title", ""),
        doc_file=obj_dict.get("doc_file", ""),
        doc_file_name=obj_dict.get("doc_file_name", ""),
        doc_mime_type=obj_dict.get("doc_mime_type", ""),
        file_mime_type=obj_dict.get("file_mime_type", ""),
    )


def parse_entity(obj_dict: Dict[str, Any], base_path=""):
    parent = None  # type: Union[UUID, None]
    if "parent" in obj_dict:
        parent = UUID(obj_dict["parent"])

    name = obj_dict["name"]
    return (
        Entity(
            uuid=UUID(obj_dict["uuid"]),
            name=name,
            full_path=f"{base_path}/{name}",
            parent=parent,
        ),
        obj_dict.get("children", []),
    )


def walk_entity_tree_and_parse(
    dictionary: Dict[UUID, Any], objs: List[Dict[str, Any]], base_path=""
):
    for obj_dict in objs:
        obj, children = parse_entity(obj_dict, base_path)
        dictionary[obj.uuid] = obj

        if children:
            walk_entity_tree_and_parse(dictionary, children, f"{base_path}/{obj.name}")


def parse_quantity(obj_dict: Dict[str, Any]):
    format_spec = None  # type: Union[UUID, None]
    if "format_spec" in obj_dict:
        format_spec = UUID(obj_dict["format_spec"])

    entity = None  # type: Union[UUID, None]
    if "entity" in obj_dict:
        entity = UUID(obj_dict["entity"])

    return Quantity(
        uuid=UUID(obj_dict["uuid"]),
        name=obj_dict.get("name", ""),
        format_spec=format_spec,
        entity=entity,
    )


def parse_data_file(obj_dict: Dict[str, Any]):
    upload_date = None  # type: Union[datetime, None]
    if "upload_date" in obj_dict:
        upload_date = datetime.fromisoformat(obj_dict["upload_date"])

    dependencies = set()  # type: Set[UUID]
    if "dependencies" in obj_dict:
        dependencies = set([UUID(x) for x in obj_dict["dependencies"]])

    return DataFile(
        uuid=UUID(obj_dict["uuid"]),
        name=obj_dict.get("name", ""),
        upload_date=upload_date,
        metadata=obj_dict.get("metadata", {}),
        file_data=obj_dict.get("file_data", ""),
        quantity=UUID(obj_dict["quantity"]),
        spec_version=obj_dict.get("spec_version", ""),
        dependencies=dependencies,
        plot_file=obj_dict.get("plot_file", ""),
        plot_mime_type=obj_dict.get("plot_mime_type", ""),
        comment=obj_dict.get("comment", ""),
    )


def parse_release(obj_dict: Dict[str, Any]):
    rel_date = None  # type: Union[datetime, None]
    if "rel_date" in obj_dict:
        rel_date = datetime.fromisoformat(obj_dict["rel_date"])

    return Release(
        tag=obj_dict["tag"],
        rel_date=rel_date,
        comments=obj_dict.get("comments", ""),
        data_files=set([UUID(x) for x in obj_dict.get("data_files", [])]),
    )


def parse_data_file_path(path: str):
    """Split a path to a data file into its components.

    Assuming that the path is in the following form:

        /relname/sequence/of/entities/…/quantity

    the function returns a tuple with three elements:

    1. The name of the release (``relname``)
    2. The full path to the entity owning the quantity
    3. The name of the quantity (``quantity``)
    """

    parts = [x for x in path.split("/") if x != ""]
    if len(parts) < 3:
        raise ValueError(f'Malformed path to data file: "{path}"')

    relname = parts[0]
    middle = parts[1:-1]
    quantity = parts[-1]

    return relname, "/" + ("/".join(middle)), quantity


class ImoFormatError(Exception):
    pass


class ImoFlatFile(Imo):
    def __init__(self, path):
        self.path = Path(path)

        self.format_specs = {}  # type: Dict[UUID, FormatSpecification]
        self.entities = {}  # type: Dict[UUID, Entity]
        self.quantities = {}  # type: Dict[UUID, Quantity]
        self.data_files = {}  # type: Dict[UUID, DataFile]

        self.path_to_entity = {}  # type: Dict[str, UUID]
        self.path_to_quantity = {}  # type: Dict[str, UUID]

        self.check_consistency()
        self.read_schema()

    def check_consistency(self):
        """Perform some basic checks on the structure of the flat-file IMO

        If the structure of the folders is not compliant with the specifications,
        raise a :class:`.ImoFormatError` exception.

        The checks are not comprehensive, but they should spot the most
        obvious errors.
        """

        if not (self.path / IMO_FLATFILE_SCHEMA_FILE_NAME).is_file():
            raise ImoFormatError(
                ("Schema file {filename} not found " 'in "{path}"').format(
                    filename=IMO_FLATFILE_SCHEMA_FILE_NAME, path=self.path.absolute(),
                )
            )

        for dirname in (
            IMO_FLATFILE_DATA_FILES_DIR_NAME,
            IMO_FLATFILE_FORMAT_SPEC_DIR_NAME,
            IMO_FLATFILE_PLOT_FILES_DIR_NAME,
        ):
            if not (self.path / dirname).is_dir():
                raise ImoFormatError(f"Directory {dirname} not found in {self.path}")

    def read_schema(self):
        "Read the YAML file containing the metadata"
        schema_file = self.path / IMO_FLATFILE_SCHEMA_FILE_NAME
        with schema_file.open("rt") as inpf:
            schema = yaml.safe_load(inpf)

        self.parse_schema(schema)

    def parse_schema(self, schema: Dict[str, Any]):
        self.format_specs = {}
        for obj_dict in schema.get("format_specifications", []):
            obj = parse_format_spec(obj_dict)
            self.format_specs[obj.uuid] = obj

        self.entities = {}
        walk_entity_tree_and_parse(self.entities, schema.get("entities", []))

        self.quantities = {}
        for obj_dict in schema.get("quantities", []):
            obj = parse_quantity(obj_dict)
            self.quantities[obj.uuid] = obj

        self.data_files = {}
        for obj_dict in schema.get("data_files", []):
            obj = parse_data_file(obj_dict)
            self.data_files[obj.uuid] = obj

        self.releases = {}
        for obj_dict in schema.get("releases", []):
            obj = parse_release(obj_dict)
            self.releases[obj.tag] = obj

        self.path_to_entity = {
            entity.full_path: uuid for uuid, entity in self.entities.items()
        }

        self.path_to_quantity = {
            self.quantity_path(quantity.uuid): uuid
            for uuid, quantity in self.quantities.items()
        }

        for cur_uuid, cur_quantity in self.quantities.items():
            if cur_quantity.entity:
                entity = self.entities[cur_quantity.entity]
                entity.quantities.add(cur_uuid)

        for cur_uuid, cur_data_file in self.data_files.items():
            quantity = self.quantities[cur_data_file.quantity]
            quantity.data_files.add(cur_uuid)

    def quantity_path(self, uuid: UUID):
        quantity = self.quantities[uuid]
        assert quantity.entity
        entity = self.entities[quantity.entity]
        return f"{entity.full_path}/{quantity.name}"

    def query_data_file(self, identifier: Union[str, UUID]):
        """Retrieve a data file

        The `identifier` parameter can be one of the following types:

        1. A ``uuid.UUID`` object
        2. A string representing a UUID
        3. A full path to an object included in a release. In this
           case, the path has the following form:

           /relname/sequence/of/entities/…/quantity

        """
        if isinstance(identifier, UUID):
            return self.data_files[identifier]
        else:
            try:
                uuid = UUID(identifier)
                return self.data_files[uuid]
            except ValueError:
                # We're dealing with a path
                relname, entity_path, quantity_name = parse_data_file_path(identifier)
                release_uuids = self.releases[relname].data_files
                entity = self.entities[self.path_to_entity[entity_path]]

                # Retrieve the quantity whose name matches the last
                # part of the path
                quantity = None
                for cur_uuid in entity.quantities:
                    cur_quantity = self.quantities[cur_uuid]
                    if cur_quantity.name == quantity_name:
                        quantity = self.quantities[cur_quantity.uuid]
                        break

                if not quantity:
                    raise KeyError(
                        (
                            'wrong path: "{id}" points to entity'
                            '"{path}", which does not have a quantity'
                            'named "{quantity}"'
                        ).format(
                            id=identifier,
                            path=entity.full_path,
                            quantity=quantity_name,
                        )
                    )

                # Now check which data file has a UUID that matches
                # the one listed in the release
                for cur_uuid in quantity.data_files:
                    if cur_uuid in release_uuids:
                        return self.data_files[cur_uuid]

                raise KeyError(
                    (
                        'wrong path: "{id}" points to quantity '
                        '"{quantity}", which does not have data files '
                        'belonging to release "{relname}" '
                        "(data files are: {uuids})"
                    ).format(
                        id=identifier,
                        quantity=quantity_name,
                        relname=relname,
                        uuids=", ".join([str(x)[0:6] for x in quantity.data_files]),
                    )
                )

    def query(self, identifier: Union[str, UUID]):
        """Query an object from the IMO

        The value of `identifier` can be one of the following:

        1. The string ``/quantities/UUID``, with ``UUID`` being the UUID of a
           quantity
        2. The string ``/entities/UUID``, which looks for an entity
        3. The string ``/format_specs/UUID``, which looks for an entity
        4. The string ``/data_files/UUID``, which looks for a data file
        5. A `UUID` object: in this case, the method assumes that a
           data file is being queried.
        6. A path in the form ``/release/entity/tree/…/quantity``; in this case,
           the method looks for the data file belonging to
           ``quantity`` within the entity tree and assigned to the
           specified release.

        The method returns an object belonging to one of the following
        classes: :class:`DataFile`, :class:`Quantity`,
        :class:`Entity`, :class:`FormatSpecification`.

        """
        if isinstance(identifier, UUID):
            return self.data_files[identifier]

        for obj_type_name, collection in [
            ("/data_files", self.data_files),
            ("/quantities", self.quantities),
            ("/entities", self.entities),
            ("/format_specs", self.format_specs),
        ]:
            if identifier.startswith(obj_type_name):
                uuid = UUID(identifier.split("/")[-1])
                return collection[uuid]  # type: ignore

        # No match, assume it's a release name
        return self.query_data_file(identifier)
