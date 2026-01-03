from abc import ABCMeta, abstractmethod

from pandas import DataFrame, Series

from interfaces.hyperparameter_optimizer import HPOOptimizable
from interfaces.shared import RunStats


class FeatureSelector(HPOOptimizable):


    @abstractmethod
    def select_features(self, x_train: DataFrame, x_test: DataFrame, precomputed: object, run_stats: RunStats, estimator_factory) -> tuple[DataFrame, DataFrame]:
        pass

    @abstractmethod
    def precompute(self, x_train: DataFrame, x_test: DataFrame, y_train: DataFrame, y_test: DataFrame, run_stats: RunStats, input_instance_groups : dict[str, Series]) -> object:
        pass