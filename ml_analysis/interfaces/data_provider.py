import pathlib
from abc import ABCMeta, abstractmethod
from enum import Enum, auto

import pandas as pd


class LabelType(Enum):
    CLASS = auto()
    NUMERIC = auto()


class DataProvider(metaclass=ABCMeta):

    @abstractmethod
    def _load_data(self, data_root: pathlib.Path):
        pass

    @abstractmethod
    def _clean_data(self, data):
        pass

    @abstractmethod
    def _remove_feature_target_duplicates(self, data):
        pass

    @abstractmethod
    def get_data(self, data_root: pathlib.Path) -> pd.DataFrame:
        data = self._load_data(data_root)
        data_cleaned = self._clean_data(data)
        data_removed = self._remove_feature_target_duplicates(data_cleaned)
        return data_removed

    @abstractmethod
    def get_label_type(self) -> LabelType:
        pass

