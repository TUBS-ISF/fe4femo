import pathlib

import numpy as np
import pandas as pd

from interfaces.data_provider import DataProvider, LabelType

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

def load_fm_bench_data(path: str) -> pd.DataFrame:
    df = load_dataset(path, "featureExtraction/values.csv")
    df.columns = df.columns.map(str)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

class SATRuntime(DataProvider):

    def _load_labels(self, data_root: pathlib.Path):
        df = load_dataset(data_root, "runtime/sat.csv")
        return df.loc[:, 'wallclockTimeS'].astype(float)

    def get_label_type(self) -> LabelType:
        return LabelType.NUMERIC

    def _remove_feature_target_duplicates(self, data):
        return data

    def _load_data(self, data_root: pathlib.Path):
        return load_fm_bench_data(data_root)

