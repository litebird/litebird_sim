# -*- encoding: utf-8 -*-

from uuid import UUID
from typing import Union, Set, Tuple

from .objects import Entity, Quantity, DataFile
from .flatfile import ImoFlatFile


class Imo:
    def __init__(self, flatfile_location=None, url=None, user=None, password=None):
        self.imoobject = None
        if flatfile_location:
            self.imoobject = ImoFlatFile(flatfile_location)

        self.queried_objects = set()  # type: Set[Tuple[type, UUID]]

    def query(self, identifier: Union[str, UUID]):
        if not self.imoobject:
            return None

        result = self.imoobject.query(identifier)
        if result:
            self.queried_objects.add((type(result), result.uuid))

        return result

    def get_queried_entities(self):
        return [x[1] for x in self.queried_objects if x[0] == Entity]

    def get_queried_quantities(self):
        return [x[1] for x in self.queried_objects if x[0] == Quantity]

    def get_queried_data_files(self):
        return [x[1] for x in self.queried_objects if x[0] == DataFile]
