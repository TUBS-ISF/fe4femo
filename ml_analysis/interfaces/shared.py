from dataclasses import dataclass


@dataclass(frozen=True)
class RunStats:
    is_classification: bool
    parallelism: int = 1
    verbose: bool = False
