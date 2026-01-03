import typing

from interfaces.data_provider import DataProvider
from plugins.data_provider.fm_benchmark_solver import SATRuntime, BackboneRuntime, SpurRuntime, SSATValue, \
    BackboneValue, AlgoSelection


def get_data_provider(name:str) -> typing.Type[DataProvider]:
    match name:
        case "runtime_sat":
            return SATRuntime
        case "runtime_backbone":
            return BackboneRuntime
        case "runtime_spur":
            return SpurRuntime
        case "value_ssat":
            return SSATValue
        case "value_backbone":
            return BackboneValue
        case "algo_selection":
            return AlgoSelection
        case _:
            raise NotImplementedError(f"Data provider \"{name}\" not implemented!")


def get_