from interfaces.data_provider import DataProvider
from plugins.data_provider.fm_benchmark_solver import SATRuntime


def get_data_provider(name:str) -> type[SATRuntime]:
    match name:
        case "runtime_sat":
            return SATRuntime
        case _:
            raise NotImplementedError(f"Data provider \"{name}\" not implemented!")