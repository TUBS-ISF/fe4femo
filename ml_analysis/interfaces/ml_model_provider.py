from abc import abstractmethod, ABCMeta

from interfaces.hyperparameter_optimizer import HPOOptimizable


class FeatureSelector(HPOOptimizable):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class ScoringMetric(metaclass=ABCMeta):
    @abstractmethod
    def score(self, X, y):
        pass