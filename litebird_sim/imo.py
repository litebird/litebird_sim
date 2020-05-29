# -*- encoding: utf-8 -*-

from uuid import UUID
from typing import Union


class Imo:
    def __init__(self):
        pass

    def query(self, identifier: Union[str, UUID]):
        raise NotImplementedError("Imo.query must be redefined in derived classes")
