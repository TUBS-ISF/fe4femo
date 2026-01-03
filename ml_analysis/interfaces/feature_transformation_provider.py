from abc import abstractmethod

from interfaces.hyperparameter_optimizer import HPOOptimizable


class FeatureTransformator(HPOOptimizable):

    @abstractmethod
    def transform_features(self, ):
        pass