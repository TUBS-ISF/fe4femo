import typing
from abc import ABC
from dataclasses import dataclass

from interfaces.hyperparameters.hyperparameters import _Hyperparameter


class _Condition(ABC):
    pass

class TrueCondition(_Condition):
    pass

@dataclass(frozen=True)
class AndCondition(_Condition):
    subconditions: list[_Condition]

@dataclass(frozen=True)
class OrCondition(_Condition):
    subconditions: list[_Condition]

@dataclass(frozen=True)
class NotCondition(_Condition):
    subcondition: _Condition

@dataclass(frozen=True)
class EqualsCondition(_Condition):
    target_hp: _Hyperparameter
    value: typing.Any

@dataclass(frozen=True)
class GreaterCondition(_Condition):
    target_hp: _Hyperparameter
    value: typing.Any

@dataclass(frozen=True)
class LessCondition(_Condition):
    target_hp: _Hyperparameter
    value: typing.Any

@dataclass(frozen=True)
class InCondition(_Condition):
    target_hp: _Hyperparameter
    value: list[typing.Any]