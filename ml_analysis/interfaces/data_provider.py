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
    def _remove_feature_target_duplicates(self, data):
        pass

    @abstractmethod
    def _load_labels(self, data_root: pathlib.Path):
        pass

    def get_data(self, data_root: pathlib.Path) -> tuple[pd.DataFrame, pd.Series]:
        data = self._load_data(data_root)
        data_removed = self._remove_feature_target_duplicates(data)
        labels = self._load_labels(data_root)
        return data_removed, labels

    @abstractmethod
    def get_label_type(self) -> LabelType:
        pass

