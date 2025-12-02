from abc import ABCMeta, abstractmethod
from typing import Generator

from pandas import DataFrame


class CVProvider(metaclass=ABCMeta):

    @abstractmethod
    def generate_splits(self, x_data: DataFrame, **kwargs) -> Generator[tuple[object, object]]:
        pass