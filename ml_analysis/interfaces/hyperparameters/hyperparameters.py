from dataclasses import dataclass

from interfaces.hyperparameters.conditions import _Condition


@dataclass(frozen=True)
class _Hyperparameter:
    name: str
    activation_condition: _Condition

@dataclass(frozen=True)
class IntHP(_Hyperparameter):
    lower_bound: int
    upper_bound: int
    step: (int|None) = None
    log: bool = False

@dataclass(frozen=True)
class FloatHP(_Hyperparameter):
    lower_bound: float
    upper_bound: float
    step: (int|None) = None
    log: bool = False

@dataclass(frozen=True)
class ChoiceHP(_Hyperparameter):
    choices: list
