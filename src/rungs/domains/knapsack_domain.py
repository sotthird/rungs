"""The Domain-protocol wrapper around the knapsack plugin — proves the
Allocator/core.py machinery generalizes beyond facility location without any
changes to core.py, sequential.py, or policies.py. See DESIGN.md §12a,
IMPLEMENTATION.md §11.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rungs.core import Result, Rung
from rungs.domains.knapsack import KnapsackInstance, generate_instance
from rungs.domains.knapsack_features import cheap_lower_bound, features
from rungs.domains.knapsack_solvers import solve_exact, solve_greedy, solve_medium


@dataclass(frozen=True)
class _SolverRung:
    name: str
    _solve_fn: Callable[[KnapsackInstance], Result]

    def solve(self, instance: KnapsackInstance) -> Result:
        return self._solve_fn(instance)


class KnapsackDomain:
    sense = "max"  # maximize total value

    def __init__(self, n_items: int, capacity_ratio_range: tuple[float, float]):
        self.n_items = n_items
        self.capacity_ratio_range = capacity_ratio_range
        self._rungs = [
            _SolverRung("greedy", solve_greedy),
            _SolverRung("medium", solve_medium),
            _SolverRung("exact", solve_exact),
        ]

    def generate(self, rng: Any, **params: Any) -> KnapsackInstance:
        return generate_instance(
            rng,
            n_items=params.get("n_items", self.n_items),
            capacity_ratio_range=params.get("capacity_ratio_range", self.capacity_ratio_range),
        )

    def features(self, instance: KnapsackInstance) -> dict[str, float]:
        return features(instance)

    def rungs(self) -> Sequence[Rung]:
        return self._rungs

    def ground_truth_rung(self) -> Rung:
        return self._rungs[-1]

    def cheap_lower_bound(self, instance: KnapsackInstance) -> float:
        return cheap_lower_bound(instance)
