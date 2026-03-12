from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from cloudpickle import cloudpickle
from optuna import load_study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from pandas import MultiIndex


def get_pickle_dict(file):
    with open(file, "br") as f:
        return cloudpickle.load(f)

def get_optuna_study(file, study_name):
    journal = JournalStorage(JournalFileBackend(file))
    study = load_study(storage=journal, study_name=study_name)
    return study, journal

def load_multiindex(file, column) -> pd.DataFrame:
    df = pd.read_csv(file, index_col=[0, 1, 2, 3, 4, 5, 6])[column]
    df.index = [df.index.get_level_values(0), df.index.map(lambda idx: f"{idx[1]}_MO" if idx[5] else idx[1]),
                df.index.get_level_values(2), df.index.get_level_values(6)]
    df = df.rename_axis(["ml_task", "feature_selector", "ml_model", "fold"])
    df = df.sort_index()
    df = df.reset_index()
    # filter optuna-combined_MO
    df = df[df['feature_selector'] != "optuna-combined_MO"]
    df.replace(get_replace_dictionary(), inplace=True)
    return df

def get_modified_performance(file) -> pd.DataFrame:
    return load_multiindex(file, 'model_quality')

def get_modified_feature_time(file) -> pd.DataFrame:
    df = pd.read_csv(file, index_col=[0, 2, 3, 4, 5, 6, 7, 8])['feature_time']
    df.index = [df.index.get_level_values(0), df.index.get_level_values(1), df.index.map(lambda idx: f"{idx[2]}_MO" if idx[6] else idx[2]),
                df.index.get_level_values(3), df.index.get_level_values(7)]
    df = df.rename_axis(["model_no", "ml_task", "feature_selector", "ml_model", "fold"])
    df = df.reset_index()
    # filter optuna-combined_MO
    df = df[df['feature_selector'] != "optuna-combined_MO"]
    df.replace(get_replace_dictionary(), inplace=True)
    return df

def get_modified_task_time(file) -> pd.DataFrame:
    df = load_multiindex(file, 'task_time')
    df['task_time'] = pd.to_timedelta(df['task_time'])
    df.replace(get_replace_dictionary(), inplace=True)

    df['task_time'] = df['task_time'].dt.total_seconds()
    mask = ~((df['feature_selector'] == 'Genetic') | (df['feature_selector'] == 'HFMOEA'))
    df.loc[mask, "task_time"] = df.loc[mask, "task_time"] / 150
    return df

def get_reduction(path):
    df_orig = pd.read_csv(path, index_col=[0, 1, 2, 3, 4, 5, 6])
    df_orig = df_orig[df_orig.index.get_level_values(5) == False]
    feature_max = len(df_orig.columns)
    df_orig.replace({False: 0, True: 1}, inplace=True)

    df = df_orig.sum(axis=1).reset_index()
    df = df.rename(columns={df.columns[-1]: "feature_count"})
    df['rel_count'] = df['feature_count'] / feature_max
    df.replace(get_replace_dictionary(), inplace=True)
    return df


def get_replace_dictionary() -> dict:
    return {"SATzilla": "SATZilla", "FM_Chara": "FM Fact Label", "kbest-mutalinfo": "MI Filtering", "multisurf":"MultiSURF",
            "genetic" : "Genetic", "embedded-tree" : "Embedded Tree", "optuna-combined" : "FS as HPO", "SVD-entropy" : "SVD-Entropy",
            "runtime_sat" : "Runtime Kissat", "runtime_backbone" : "Runtime CaDiBack", "runtime_spur" : "Runtime Spur",
            "value_ssat" : "FM Cardinality", "value_backbone" : "Backbone Size", "algo_selection" : "#SAT Algorithm Selection",
            "all": "Complete", "prefilter": "Prefiltering",
            "randomForest" : "Random Forest", "gradboostForest" : "GB Trees", "adaboost": "AdaBoost"}

def get_order() -> list:
    return ["Complete", "Prefiltering", "SATZilla", "SATfeatPy", "FMBA", "FM Fact Label", "MI Filtering", "MultiSURF",
            "mRMR", "RFE", "Genetic", "HFMOEA", "Embedded Tree", "FS as HPO", "SVD-Entropy", "NDFS"]

@dataclass(frozen=True, eq=True)
class ExperimentInstance:
    path_log : Path
    path_pickle : Path
    path_journal : Path
    path_dask : Path
    name: str
    task_count: int
    runtime_limit: int
    fold_no: int
    feature_selector: str
    ml_task: str
    ml_model: str
    hpo_iterations: int
    is_model_hpo: bool
    is_selector_hpo: bool
    is_multi_objective: bool

def list_experiment_instances(config_path: Path, data_path: Path) -> Iterable[ExperimentInstance]:
    with open(config_path, "r") as f:
        next(f)
        for line in f:
            line_array = line.strip().split(" ")
            folds = range(10) if line_array[4] == "-1" else int(line_array[4])
            name = line_array[1]
            for fold in folds:
                yield ExperimentInstance(
                    name=name,
                    task_count=int(line_array[2]),
                    runtime_limit=int(line_array[3]),
                    fold_no=fold,
                    feature_selector=line_array[5],
                    ml_task=line_array[6],
                    ml_model=line_array[7],
                    hpo_iterations=int(line_array[8]),
                    is_model_hpo=line_array[9]  == "True",
                    is_selector_hpo=line_array[10] == "True",
                    is_multi_objective=line_array[11] == "True",

                    path_log= data_path.joinpath(f"{name}.out") if line_array[4] == "-1" else data_path.joinpath(f"{name}#{fold}.out"),
                    path_dask=data_path.joinpath(f"{name}#{fold}.html"),
                    path_pickle=data_path.joinpath(f"{name}#{fold}.pkl"),
                    path_journal=data_path.joinpath(f"{name}#{fold}.journal"),
                )
