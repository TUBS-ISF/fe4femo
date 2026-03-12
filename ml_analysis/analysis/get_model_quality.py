import os
import time
from operator import itemgetter
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from pandas import MultiIndex
from sklearn.metrics import matthews_corrcoef, d2_absolute_error_score

from analysis.analysis_helper import get_pickle_dict, list_experiment_instances, ExperimentInstance
from helper.feature_selection import set_njobs_if_possible
from helper.model_training import is_model_classifier


def get_model_quality(file) -> list[float]:
    dictonary = get_pickle_dict(file)
    ret_list = []
    for trial_container in dictonary["trial_container"]:
        model = trial_container.model
        set_njobs_if_possible(model, 1)
        start_Model = time.time()
        y_pred = model.predict(trial_container.x_test)
        end_Model = time.time()
        y_test = dictonary["y_test"]
        time_elapsed = end_Model - start_Model
        if is_model_classifier(trial_container.model):
            model_qual = matthews_corrcoef(y_test, y_pred)
        else:
            model_qual = d2_absolute_error_score(y_test, y_pred).item()
        ret_list.append((model_qual, time_elapsed, trial_container.time_Model))
    return ret_list

def _parallel_wrapper(experiment_instance : ExperimentInstance)->tuple[tuple, list[float]]:
    print(f"Handling {experiment_instance}")
    qualities = get_model_quality(experiment_instance.path_pickle)
    index_tuple = experiment_instance.ml_task, experiment_instance.feature_selector, experiment_instance.ml_model, experiment_instance.is_model_hpo, experiment_instance.is_selector_hpo, experiment_instance.is_multi_objective, experiment_instance.fold_no
    return index_tuple, qualities

if __name__ == '__main__':
    config_path = Path("/mnt/c/Users/rsd61/IdeaProjects/fe4femo/ml_analysis/config.txt").expanduser()
    data_path = Path("/mnt/d/MA/ml_main/main/").expanduser()
    out_file_qual = Path("model_quality.csv").expanduser()
    out_file_times = Path("model_times.csv").expanduser()

    experiment_instances = list_experiment_instances(config_path, data_path)
    ret_gen = Parallel(n_jobs=16, verbose=10, return_as='generator_unordered')(delayed(_parallel_wrapper)(experiment_instance) for experiment_instance in experiment_instances)
    #ret_gen = (_parallel_wrapper(experiment_instance) for experiment_instance in experiment_instances)

    index_tuples = []
    values_qual = []
    values_time = []
    for index_tuple, qualities in ret_gen:
        index_tuples.append(index_tuple)
        maximum = max(qualities, key=itemgetter(0))
        values_qual.append(maximum[0])
        values_time.append((maximum[1],maximum[2]))

    multi_index = MultiIndex.from_tuples(index_tuples, names=["ml_task", "feature_selector", "ml_model", "model_hpo", "selector_hpo", "multi_objective", "fold"])
    df = pd.Series(values_qual, index=multi_index, name="model_quality")
    df.to_csv(out_file_qual)

    df_time = pd.DataFrame(values_time, index=multi_index, columns=["model_eval", "model_training"])
    df_time.to_csv(out_file_times)