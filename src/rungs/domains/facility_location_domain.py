"""The Domain-protocol wrapper around the facility_location plugin (DESIGN.md
§14, IMPLEMENTATION.md §10). Everything domain-specific lives here and in the
sibling facility_location*.py modules; rungs/core.py never imports this file.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rungs.core import Result, Rung
from rungs.domains.facility_location import FLInstance, generate_instance
from rungs.domains.facility_location_features import cheap_lower_bound, features
from rungs.domains.facility_location_solvers import solve_exact, solve_greedy, solve_medium


@dataclass(frozen=True)
class _SolverRung:
    name: str
    _solve_fn: Callable[[FLInstance], Result]

    def solve(self, instance: FLInstance) -> Result:
        return self._solve_fn(instance)


class FacilityLocationDomain:
    sense = "min"  # minimize total opening + transport cost

    def __init__(
        self,
        n_sites: int,
        n_customers: int,
        tightness_range: tuple[float, float],
        demand_cv_range: tuple[float, float],
        fixed_transport_ratio_range: tuple[float, float],
    ):
        self.n_sites = n_sites
        self.n_customers = n_customers
        self.tightness_range = tightness_range
        self.demand_cv_range = demand_cv_range
        self.fixed_transport_ratio_range = fixed_transport_ratio_range
        # Cheapest first, ground truth last — see rungs/core.py's Allocator.
        self._rungs = [
            _SolverRung("greedy", solve_greedy),
            _SolverRung("medium", solve_medium),
            _SolverRung("exact", solve_exact),
        ]

    def generate(self, rng: Any, **params: Any) -> FLInstance:
        return generate_instance(
            rng,
            n_sites=params.get("n_sites", self.n_sites),
            n_customers=params.get("n_customers", self.n_customers),
            tightness_range=params.get("tightness_range", self.tightness_range),
            demand_cv_range=params.get("demand_cv_range", self.demand_cv_range),
            fixed_transport_ratio_range=params.get(
                "fixed_transport_ratio_range", self.fixed_transport_ratio_range
            ),
        )

    def features(self, instance: FLInstance) -> dict[str, float]:
        return features(instance)

    def rungs(self) -> Sequence[Rung]:
        return self._rungs

    def ground_truth_rung(self) -> Rung:
        return self._rungs[-1]

    def cheap_lower_bound(self, instance: FLInstance) -> float:
        return cheap_lower_bound(instance)
