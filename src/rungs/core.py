from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rungs.policies import OneShotAllocator
from rungs.sequential import (
    RungMetrics,
    SequentialTables,
    compute_quantile_thresholds,
    fit_sequential_tables,
    quantile_bucket,
)


@dataclass(frozen=True)
class Result:
    objective: float | None  # None if no solution produced
    feasible: bool
    solve_time: float  # wall clock, this rung only
    artifact: Any = None  # the solution itself


@runtime_checkable
class Rung(Protocol):
    @property
    def name(self) -> str: ...  # read-only: frozen-dataclass implementations are common

    def solve(self, instance: Any) -> Result: ...


@runtime_checkable
class Domain(Protocol):
    sense: str  # "min" or "max" — which direction is better for this objective

    def generate(self, rng: Any, **params: Any) -> Any: ...
    def features(self, instance: Any) -> dict[str, float]: ...
    def rungs(self) -> Sequence[Rung]: ...
    def ground_truth_rung(self) -> Rung: ...
    def cheap_lower_bound(self, instance: Any) -> float: ...


@dataclass(frozen=True)
class AllocationResult:
    rung_used: str
    rungs_attempted: list[str]
    total_time: float
    objective: float | None
    feasible: bool


def _quality_gap(objective: float | None, feasible: bool, obj_exact: float, sense: str) -> float:
    """Always >= 0, 0 = optimal, regardless of whether the domain minimizes
    (facility location: cost) or maximizes (knapsack: value)."""
    gap_infeasible = 1.0
    if not feasible or objective is None:
        return gap_infeasible
    if sense == "max":
        return (obj_exact - objective) / obj_exact
    return (objective - obj_exact) / obj_exact


@dataclass
class _FitInstance:
    features: dict[str, float]
    metrics: dict[str, RungMetrics]
    lb: float


class Allocator:
    """Domain-agnostic instance-adaptive solver selector.

    Wraps a Domain plugin (a set of Rungs plus feature/lower-bound
    extraction) with one of the two policies developed against the facility
    location domain: "one_shot" (predict-then-choose) or "sequential"
    (run-observe-escalate). See DESIGN.md §9, §10.
    """

    def __init__(self, domain: Domain, lambda_: float, mode: str = "one_shot"):
        if mode not in ("one_shot", "sequential"):
            raise ValueError(f"unknown mode {mode!r}, expected 'one_shot' or 'sequential'")
        self.domain = domain
        self.lambda_ = lambda_
        self.mode = mode
        # Convention: domain.rungs() is ordered cheapest-first, ground truth
        # last (the sequential policy escalates in exactly this order).
        self._path_order: list[str] = [r.name for r in domain.rungs()]
        self._rung_names = self._path_order
        self._exact_name = domain.ground_truth_rung().name
        assert self._path_order[-1] == self._exact_name, (
            "domain.rungs() must end with the ground-truth rung"
        )
        self._one_shot: OneShotAllocator | None = None
        self._sequential_tables: SequentialTables | None = None

    def fit(self, instances: list[Any]) -> None:
        fit_data = self._solve_all_rungs(instances)

        if self.mode == "one_shot":
            feature_keys = sorted(fit_data[0].features)
            X = [[d.features[k] for k in feature_keys] for d in fit_data]
            gaps = {r: [d.metrics[r].gap for d in fit_data] for r in self._rung_names}
            times = {r: [d.metrics[r].solve_time for d in fit_data] for r in self._rung_names}
            self._feature_keys = feature_keys
            self._one_shot = OneShotAllocator(self._rung_names)
            self._one_shot.fit(
                np.array(X),
                {r: np.array(gaps[r]) for r in self._rung_names},
                {r: np.array(times[r]) for r in self._rung_names},
            )
        else:
            metrics_list = [d.metrics for d in fit_data]
            lb_list = [d.lb for d in fit_data]
            bucket_s1, bucket_s2 = self._make_sequential_buckets(metrics_list, lb_list)
            self._sequential_tables = fit_sequential_tables(
                metrics_list, lb_list, self.lambda_, bucket_s1, bucket_s2, min_bucket_size=20
            )
            self._bucket_s1, self._bucket_s2 = bucket_s1, bucket_s2

    def _solve_all_rungs(self, instances: list[Any]) -> list[_FitInstance]:
        domain_rungs = self.domain.rungs()
        exact_rung = self.domain.ground_truth_rung()
        out = []
        for inst in instances:
            exact_result = exact_rung.solve(inst)
            if not exact_result.feasible or exact_result.objective is None:
                continue
            metrics = {}
            for rung in domain_rungs:
                result = exact_result if rung.name == exact_rung.name else rung.solve(inst)
                metrics[rung.name] = RungMetrics(
                    objective=result.objective,
                    feasible=result.feasible,
                    gap=_quality_gap(
                        result.objective, result.feasible, exact_result.objective, self.domain.sense
                    ),
                    solve_time=result.solve_time,
                )
            out.append(
                _FitInstance(
                    features=self.domain.features(inst),
                    metrics=metrics,
                    lb=self.domain.cheap_lower_bound(inst),
                )
            )
        return out

    def _make_sequential_buckets(self, train_metrics, train_lb):
        cheap_names = [r for r in self._rung_names if r != self._exact_name]
        first, second = cheap_names[0], cheap_names[1] if len(cheap_names) > 1 else cheap_names[0]

        ratios = [
            m[first].objective / lb
            for m, lb in zip(train_metrics, train_lb, strict=True)
            if m[first].feasible
        ]
        thresholds = compute_quantile_thresholds(ratios, n_bins=3) if ratios else []

        def bucket_s1(m, lb):
            if not m[first].feasible:
                return "infeasible"
            ratio = m[first].objective / lb
            return f"{first}_{quantile_bucket(ratio, thresholds)}"

        def bucket_s2(m, lb):
            s1 = bucket_s1(m, lb)
            label = "feasible" if m[second].feasible else "infeasible"
            return f"{s1}|{second}_{label}"

        return bucket_s1, bucket_s2

    def solve(self, instance: Any) -> AllocationResult:
        if self.mode == "one_shot":
            return self._solve_one_shot(instance)
        return self._solve_sequential(instance)

    def _solve_one_shot(self, instance: Any) -> AllocationResult:
        assert self._one_shot is not None, "call fit() before solve()"
        feats = self.domain.features(instance)
        x = np.array([feats[k] for k in self._feature_keys])
        chosen_name = self._one_shot.choose(x, self.lambda_)
        rung = next(r for r in self.domain.rungs() if r.name == chosen_name)
        result = rung.solve(instance)
        return AllocationResult(
            rung_used=chosen_name,
            rungs_attempted=[chosen_name],
            total_time=result.solve_time,
            objective=result.objective,
            feasible=result.feasible,
        )

    def _solve_sequential(self, instance: Any) -> AllocationResult:
        assert self._sequential_tables is not None, "call fit() before solve()"
        rungs_by_name = {r.name: r for r in self.domain.rungs()}
        lb = self.domain.cheap_lower_bound(instance)
        attempted: list[str] = []
        total_time = 0.0
        metrics: dict[str, RungMetrics] = {}
        last_result: Result | None = None

        for stage_name in self._path_order:
            result = rungs_by_name[stage_name].solve(instance)
            attempted.append(stage_name)
            total_time += result.solve_time
            last_result = result
            metrics[stage_name] = RungMetrics(
                objective=result.objective,
                feasible=result.feasible,
                gap=0.0,  # unused by bucketing; the true gap isn't knowable here
                solve_time=result.solve_time,
            )
            if stage_name == self._exact_name:
                break

            decisions = (
                self._sequential_tables.s1_decisions
                if len(attempted) == 1
                else self._sequential_tables.s2_decisions
            )
            bucket_fn = self._bucket_s1 if len(attempted) == 1 else self._bucket_s2
            bucket = bucket_fn(metrics, lb)
            if decisions.get(bucket, "escalate") == "accept":
                break

        assert last_result is not None  # loop always runs at least once
        return AllocationResult(
            rung_used=attempted[-1],
            rungs_attempted=attempted,
            total_time=total_time,
            objective=last_result.objective,
            feasible=last_result.feasible,
        )
