# -*- encoding: utf-8 -*-
from abc import ABC, abstractmethod
from uuid import UUID
from typing import Union


class Imo(ABC):
    @abstractmethod
    def query(self, identifier: Union[str, UUID]):
        pass
