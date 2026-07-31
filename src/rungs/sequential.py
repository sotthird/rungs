from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Decision = Literal["accept", "escalate"]


@dataclass(frozen=True)
class RungMetrics:
    objective: float | None
    feasible: bool
    gap: float
    solve_time: float


@dataclass(frozen=True)
class SequentialTables:
    """Fitted state-bucket -> decision lookup, plus the bucketing functions."""

    s1_decisions: dict[str, Decision]
    s2_decisions: dict[str, Decision]
    bucket_s1: Callable[[dict[str, RungMetrics], float], str]
    bucket_s2: Callable[[dict[str, RungMetrics], float], str]


@dataclass(frozen=True)
class SequentialResult:
    gap: float
    total_time: float
    path: list[str]


def compute_quantile_thresholds(values: list[float], n_bins: int) -> list[float]:
    """Interior cut points splitting values into n_bins roughly equal groups."""
    sorted_values = sorted(values)
    n = len(sorted_values)
    cuts = []
    for k in range(1, n_bins):
        idx = min(n - 1, int(round(k * n / n_bins)))
        cuts.append(sorted_values[idx])
    return cuts


def quantile_bucket(value: float, thresholds: list[float]) -> str:
    bin_idx = sum(1 for t in thresholds if value > t)
    return f"q{bin_idx}"


_MERGED_KEY = "__merged__"


def _resolve_with_merge(raw_labels: list[str], min_bucket_size: int) -> dict[str, str]:
    """Map each raw label to itself if it has enough occupancy, else to a
    shared merged key. Any label unseen at fit time also resolves to the
    merged key at apply time via .get() with this mapping's default."""
    counts: dict[str, int] = {}
    for label in raw_labels:
        counts[label] = counts.get(label, 0) + 1
    return {
        label: label if count >= min_bucket_size else _MERGED_KEY for label, count in counts.items()
    }


def fit_sequential_tables(
    metrics_list: list[dict[str, RungMetrics]],
    lb_list: list[float],
    lambda_: float,
    bucket_s1: Callable[[dict[str, RungMetrics], float], str],
    bucket_s2: Callable[[dict[str, RungMetrics], float], str],
    min_bucket_size: int = 20,
) -> SequentialTables:
    """Tabular backward induction over cached, solved training instances.

    V2(s2) = min(E[gap_medium | s2], lambda * E[time_exact | s2])
    V1(s1) = min(E[gap_greedy | s1], lambda * E[time_medium | s1] + E[V2 | s1])

    Buckets with fewer than min_bucket_size training instances are merged
    into a single catch-all bucket (DESIGN.md §17, IMPLEMENTATION.md §8).
    """
    raw_s2 = [bucket_s2(m, lb) for m, lb in zip(metrics_list, lb_list, strict=True)]
    s2_resolve = _resolve_with_merge(raw_s2, min_bucket_size)
    resolved_s2 = [s2_resolve[label] for label in raw_s2]

    s2_gap_sums: dict[str, float] = {}
    s2_exact_time_sums: dict[str, float] = {}
    s2_counts: dict[str, int] = {}
    for label, m in zip(resolved_s2, metrics_list, strict=True):
        s2_gap_sums[label] = s2_gap_sums.get(label, 0.0) + m["medium"].gap
        s2_exact_time_sums[label] = s2_exact_time_sums.get(label, 0.0) + m["exact"].solve_time
        s2_counts[label] = s2_counts.get(label, 0) + 1

    s2_decisions: dict[str, Decision] = {}
    v2_of: dict[str, float] = {}
    for label, count in s2_counts.items():
        mean_medium_gap = s2_gap_sums[label] / count
        mean_exact_time = s2_exact_time_sums[label] / count
        escalate_value = lambda_ * mean_exact_time
        v2_of[label] = min(mean_medium_gap, escalate_value)
        s2_decisions[label] = "accept" if mean_medium_gap <= escalate_value else "escalate"

    v2_per_instance = [v2_of[label] for label in resolved_s2]

    raw_s1 = [bucket_s1(m, lb) for m, lb in zip(metrics_list, lb_list, strict=True)]
    s1_resolve = _resolve_with_merge(raw_s1, min_bucket_size)
    resolved_s1 = [s1_resolve[label] for label in raw_s1]

    s1_gap_sums: dict[str, float] = {}
    s1_medium_time_sums: dict[str, float] = {}
    s1_v2_sums: dict[str, float] = {}
    s1_counts: dict[str, int] = {}
    for label, m, v2 in zip(resolved_s1, metrics_list, v2_per_instance, strict=True):
        s1_gap_sums[label] = s1_gap_sums.get(label, 0.0) + m["greedy"].gap
        s1_medium_time_sums[label] = s1_medium_time_sums.get(label, 0.0) + m["medium"].solve_time
        s1_v2_sums[label] = s1_v2_sums.get(label, 0.0) + v2
        s1_counts[label] = s1_counts.get(label, 0) + 1

    s1_decisions: dict[str, Decision] = {}
    for label, count in s1_counts.items():
        mean_greedy_gap = s1_gap_sums[label] / count
        mean_medium_time = s1_medium_time_sums[label] / count
        mean_v2 = s1_v2_sums[label] / count
        escalate_value = lambda_ * mean_medium_time + mean_v2
        s1_decisions[label] = "accept" if mean_greedy_gap <= escalate_value else "escalate"

    def resolved_bucket_s1(m: dict[str, RungMetrics], lb: float) -> str:
        return s1_resolve.get(bucket_s1(m, lb), _MERGED_KEY)

    def resolved_bucket_s2(m: dict[str, RungMetrics], lb: float) -> str:
        return s2_resolve.get(bucket_s2(m, lb), _MERGED_KEY)

    return SequentialTables(
        s1_decisions=s1_decisions,
        s2_decisions=s2_decisions,
        bucket_s1=resolved_bucket_s1,
        bucket_s2=resolved_bucket_s2,
    )


def apply_sequential_policy(
    metrics: dict[str, RungMetrics],
    lb: float,
    tables: SequentialTables,
    lambda_: float,
) -> SequentialResult:
    total_time = metrics["greedy"].solve_time
    path = ["greedy"]

    s1 = tables.bucket_s1(metrics, lb)
    if tables.s1_decisions.get(s1, "escalate") == "accept":
        return SequentialResult(gap=metrics["greedy"].gap, total_time=total_time, path=path)

    total_time += metrics["medium"].solve_time
    path.append("medium")

    s2 = tables.bucket_s2(metrics, lb)
    if tables.s2_decisions.get(s2, "escalate") == "accept":
        return SequentialResult(gap=metrics["medium"].gap, total_time=total_time, path=path)

    total_time += metrics["exact"].solve_time
    path.append("exact")
    return SequentialResult(gap=metrics["exact"].gap, total_time=total_time, path=path)
