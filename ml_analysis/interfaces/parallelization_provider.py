from abc import ABCMeta, abstractmethod
from typing import Callable


class ParallelizationProvider(metaclass=ABCMeta):

    def __init__(self, n_parallel: int) -> None:
        self.n_parallel = n_parallel


    @abstractmethod
    def distribute_data(self, data: object) -> object:
        pass

    @abstractmethod
    def compute(self, function: Callable, *parameters, **abstract_parameters) -> object:
        pass
