# -*- encoding: utf-8 -*-

import logging as log
from pathlib import Path
from uuid import UUID
from typing import Union, Set, Tuple, List

import tomlkit

from .objects import Entity, Quantity, DataFile
from .flatfile import ImoFlatFile

CONFIG_FILE_PATH = Path.home() / ".config" / "litebird_imo" / "imo.toml"


class Imo:
    def __init__(self, flatfile_location=None, url=None, user=None, password=None):
        self.imoobject = None
        if (not flatfile_location) and (not url):
            # Try to load the configuration file

            try:
                with CONFIG_FILE_PATH.open("rt") as inpf:
                    config = tomlkit.loads("".join(inpf.readlines()))
                location = config["repositories"][0]["location"]
                self.imoobject = ImoFlatFile(location)
            except FileNotFoundError:
                log.warning('IMO config file "%s" not found.', str(CONFIG_FILE_PATH))
                log.warning(
                    "Have you run the initial setup procedure ",
                    "(python -m litebird_sim.install_imo)?",
                )
            except tomlkit.exceptions.NonExistentKey:
                log.warning('no repositories in file "%s"', str(CONFIG_FILE_PATH))

        if flatfile_location:
            self.imoobject = ImoFlatFile(flatfile_location)

        if url:
            raise NotImplementedError("access to remote IMOs is not supported yet")

        self.queried_objects = set()  # type: Set[Tuple[type, UUID]]

    def query_entity(self, identifier: UUID, track=True) -> Entity:
        """Return a :class:`.Entity` object from an UUID.

        If ``track`` is `True` (the default), then the UUID of the
        object will be kept in memory and will be returned by the
        method :meth:`.get_queried_entities`.

        """

        if not self.imoobject:
            return None

        result = self.imoobject.query_entity(identifier)
        if result and track:
            self.queried_objects.add((type(result), result.uuid))

        return result

    def query_quantity(self, identifier: UUID, track=True) -> Quantity:
        """Return a :class:`.Quantity` object from an UUID.

        If ``track`` is `True` (the default), then the UUID of the
        object will be kept in memory and will be returned by the
        method :meth:`.get_queried_quantities`.

        """

        if not self.imoobject:
            return None

        result = self.imoobject.query_quantity(identifier)
        if result and track:
            self.queried_objects.add((type(result), result.uuid))

        return result

    def query_data_file(self, identifier: Union[str, UUID], track=True) -> DataFile:
        """Return a :class:`.DataFile` object from an UUID.

        If ``track`` is `True` (the default), then the UUID of the
        object will be kept in memory and will be returned by the
        method :meth:`.get_queried_data_files`.

        """
        if not self.imoobject:
            return None

        result = self.imoobject.query_data_file(identifier)
        if result and track:
            self.queried_objects.add((type(result), result.uuid))

        return result

    def query(self, identifier: Union[str, UUID], track=True):
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

        If ``track`` is `True` (the default), then the UUID of the
        object will be kept in memory and will be returned by the
        method :meth:`.get_queried_data_files`.

        """
        if not self.imoobject:
            return None

        result = self.imoobject.query(identifier)
        if result and track:
            self.queried_objects.add((type(result), result.uuid))
        return result

    def get_list_of_data_files(self, quantity_uuid: UUID, track=False) -> List[UUID]:
        """Return a sorted list of the UUIDs of the data files belonging to a quantity.

        The result is sorted according to their upload date (oldest
        first, newest last).

        If ``track`` is `True`, then the UUID of the object will be
        kept in memory and will be returned by the method
        :meth:`.get_queried_data_files`. The default is ``False``, as
        this function is typically used to check which data files are
        available, not because the caller is going to use each of them.

        """
        quantity = self.query_quantity(quantity_uuid, track=track)
        data_files = [self.query_data_file(x, track=track) for x in quantity.data_files]
        return sorted(data_files, key=lambda x: x.upload_date)

    def get_queried_entities(self):
        """Return a list of the UUIDs of entities queried so far."""

        return [x[1] for x in self.queried_objects if x[0] == Entity]

    def get_queried_quantities(self):
        """Return a list of the UUIDs of quantities queried so far."""

        return [x[1] for x in self.queried_objects if x[0] == Quantity]

    def get_queried_data_files(self):
        """Return a list of the UUIDs of data files queried so far."""

        return [x[1] for x in self.queried_objects if x[0] == DataFile]
