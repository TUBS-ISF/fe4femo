import uuid
from abc import ABCMeta, abstractmethod

from interfaces.hyperparameters.hyperparameters import _Hyperparameter


def get_parameter_name_modded(name: str, unique_id: int | None = None) -> str:
    if unique_id is None:
        return name
    else:
        return f"{name}_{unique_id}"


class HPOOptimizable(metaclass=ABCMeta):

    @abstractmethod
    def get_hyperparameter_search_space(self, unique_id: int|None = None) -> list[_Hyperparameter]:
        pass


class HPOptimizer(metaclass=ABCMeta):

    @abstractmethod
    def optimize_hyperparameters(self): #todo anders, initialize+andere opti?
        pass

    #TODO how to encode hyperparameters in others? --> wrapper_methods like configspace, translate live for optuna
    #TODO use abstract factory for creation of hyperparameters and constraints --> trees???????