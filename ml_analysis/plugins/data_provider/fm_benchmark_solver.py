import pathlib
from abc import ABC
from collections import Counter

import numpy as np
import pandas as pd

from interfaces.data_provider import DataProvider, LabelType

_sharpsat_names = [
        "approxmc",
        "countantom",
        "d4v2_23",
        "d4v2_24",
        "exactmc_arjun",
        "ganak",
        "sharpsattd"
    ]

_suffix_mc = "_modelCount"
_suffix_time = "_wallclockTimeS"

def load_dataset(path : pathlib.Path, subpath : str) -> pd.DataFrame:
    df = pd.read_csv(path / subpath, header=0, low_memory=False, index_col="modelNo")
    df.replace({False: 0, True: 1, None: pd.NA}, inplace=True)
    for index, type in df.dtypes.items():
        df[index] = pd.to_numeric(df[index], errors="coerce", downcast="float")
    for index, type in df.dtypes.items():
        if type == "float64":
            df[index] = np.log2(df[index])
    df = df.astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def load_fm_bench_data(path: pathlib.Path) -> pd.DataFrame:
    df = load_dataset(path, "featureExtraction/values.csv")
    df.columns = df.columns.map(str)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

class FMBenchProvider(DataProvider, ABC):
    def _remove_feature_target_duplicates(self, data):
        return data

    def _load_data(self, data_root: pathlib.Path):
        return load_fm_bench_data(data_root)

class SATRuntime(FMBenchProvider):
    def _load_labels(self, data_root: pathlib.Path):
        df = load_dataset(data_root, "runtime/sat.csv")
        return df.loc[:, 'wallclockTimeS'].astype(float)

    def get_label_type(self) -> LabelType:
        return LabelType.NUMERIC


class BackboneRuntime(FMBenchProvider):
    def _load_labels(self, data_root: pathlib.Path):
        df = load_dataset(data_root, "runtime/backbone.csv")
        return df.loc[:, 'wallclockTimeS'].astype(float)

    def get_label_type(self) -> LabelType:
        return LabelType.NUMERIC


class SpurRuntime(FMBenchProvider):
    def _load_labels(self, data_root: pathlib.Path):
        df = load_dataset(data_root, f"runtime/spur_100.csv")
        return df.loc[:, 'wallclockTimeS'].astype(float)

    def get_label_type(self) -> LabelType:
        return LabelType.NUMERIC


class SSATValue(FMBenchProvider):
    @staticmethod
    def check_ssat_value(row: pd.Series) -> int:
        row_na = row.dropna()
        counts = [int(row[f"{i}{_suffix_mc}"]) for i in _sharpsat_names if f"{i}{_suffix_mc}" in row_na.index]
        if len(counts) == 0:
            return -1
        else:
            return Counter(counts).most_common(1)[0][0]

    def _load_labels(self, data_root: pathlib.Path):
        df = load_dataset(data_root, "runtime/sharpsat.csv")
        return df.apply(self.check_ssat_value, axis=1)

    def get_label_type(self) -> LabelType:
        return LabelType.NUMERIC

    def _remove_feature_target_duplicates(self, data):
        return data.drop(
            ['DyMMer_9/DyMMer/Number_of_valid_configurations_(NVC)', 'FMBA_18/FMBA/NumberOfValidConfigurationsLog',
             'FM_Characterization_1/FM_Characterization/ANALYSIS/Configurations/value',
             'FM_Characterization_1/FM_Characterization/ANALYSIS/Partial_variability/value',
             'FM_Characterization_1/FM_Characterization/ANALYSIS/Total_variability/value'], axis=1)

class BackboneValue(FMBenchProvider):
    def _load_labels(self, data_root: pathlib.Path):
        df = load_dataset(data_root, "runtime/backbone.csv")
        df = df.fillna(-1)
        return df.loc[:, 'backboneSize'].astype(int)

    def get_label_type(self) -> LabelType:
        return LabelType.NUMERIC

class AlgoSelection(FMBenchProvider):

    @staticmethod
    def check_ssat_value(row: pd.Series) -> int:
        row_na = row.dropna()
        counts = [int(row[f"{i}{_suffix_mc}"]) for i in _sharpsat_names if f"{i}{_suffix_mc}" in row_na.index]
        if len(counts) == 0:
            return -1
        else:
            return Counter(counts).most_common(1)[0][0]

    @staticmethod
    def check_best_sSAT_solver(row: pd.Series) -> str:
        count = AlgoSelection.check_ssat_value(row)
        row_na = row.dropna()
        if count == -1:
            time_dict = {solver: row[solver + _suffix_time] for solver in _sharpsat_names}
        else:
            time_dict = {solver: row[solver + _suffix_time] for solver in _sharpsat_names if
                         f"{solver}{_suffix_mc}" in row_na.index and int(row[solver + _suffix_mc]) == count}
        return min(time_dict, key=time_dict.get)

    def _load_labels(self, data_root: pathlib.Path):
        df = load_dataset(data_root, "runtime/sharpsat.csv")
        return df.apply(AlgoSelection.check_best_sSAT_solver, axis=1)

    def get_label_type(self) -> LabelType:
        return LabelType.CLASS