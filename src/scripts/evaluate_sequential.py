"""Sequential escalation policy, cost-accounting check, and the
information-value ablation. See internal/DESIGN.md §10, §12(ablation) and
internal/IMPLEMENTATION.md §8.
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from rungs.sequential import (
    RungMetrics,
    apply_sequential_policy,
    compute_quantile_thresholds,
    fit_sequential_tables,
    quantile_bucket,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"
CACHE_PATH = ROOT / "data" / "cache.parquet"
FIGURES_DIR = ROOT / "figures"

RUNG_NAMES = ["exact", "medium", "greedy"]
MIN_BUCKET_SIZE = 20


def five_fold_assignment(n: int, seed: int, k: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(n)
    fold_of = np.empty(n, dtype=int)
    fold_of[perm] = np.arange(n) % k
    return fold_of


def load_wide_cache() -> pl.DataFrame:
    df = pl.read_parquet(CACHE_PATH)
    metrics = df.pivot(
        on="rung", index="instance_id", values=["quality_gap", "solve_time", "objective", "feasible"]
    ).sort("instance_id")
    feats = (
        df.select(["instance_id", "cheap_lower_bound", "tightness"])
        .unique(subset=["instance_id"])
        .sort("instance_id")
    )
    return metrics.join(feats, on="instance_id")


def build_metrics_list(
    wide: pl.DataFrame,
) -> tuple[list[dict[str, RungMetrics]], list[float], list[float]]:
    metrics_list = []
    for row in wide.iter_rows(named=True):
        m = {
            r: RungMetrics(
                objective=row[f"objective_{r}"],
                feasible=bool(row[f"feasible_{r}"]),
                gap=row[f"quality_gap_{r}"],
                solve_time=row[f"solve_time_{r}"],
            )
            for r in RUNG_NAMES
        }
        metrics_list.append(m)
    return metrics_list, wide["cheap_lower_bound"].to_list(), wide["tightness"].to_list()


def make_full_buckets(train_metrics, train_lb):
    """State includes observed greedy/medium outcomes (the real policy)."""
    ratios = [
        m["greedy"].objective / lb
        for m, lb in zip(train_metrics, train_lb, strict=True)
        if m["greedy"].feasible
    ]
    thresholds = compute_quantile_thresholds(ratios, n_bins=3)

    def bucket_s1(m, lb):
        if not m["greedy"].feasible:
            return "infeasible"
        ratio = m["greedy"].objective / lb
        return f"greedy_{quantile_bucket(ratio, thresholds)}"

    def bucket_s2(m, lb):
        s1 = bucket_s1(m, lb)
        medium_label = "feasible" if m["medium"].feasible else "infeasible"
        return f"{s1}|medium_{medium_label}"

    return bucket_s1, bucket_s2


def make_ablation_buckets(train_metrics, train_tightness):
    """State uses ONLY static pre-solve features (tightness), never the
    realized greedy/medium outcome. Isolates the value of purchased evidence.

    Reuses the bucket_s1/bucket_s2(metrics, lb) signature but repurposes the
    second argument to carry tightness instead of the true lower bound (the
    ablation doesn't need the lower bound at all) — done via the caller
    passing tightness_list in place of lb_list for this variant only.
    train_metrics is unused; kept so both bucket builders share a signature.
    """
    thresholds = compute_quantile_thresholds(train_tightness, n_bins=3)

    def bucket(m, tightness):
        return f"tightness_{quantile_bucket(tightness, thresholds)}"

    return bucket, bucket


def run_cv(metrics_list, context_list, fold_of, cv_folds, lambda_, make_buckets, min_bucket_size):
    n = len(metrics_list)
    gap = np.empty(n)
    time = np.empty(n)
    path_end = [None] * n
    last_tables = None
    last_train_metrics = None
    last_train_context = None

    for fold in range(cv_folds):
        train_idx = np.where(fold_of != fold)[0]
        test_idx = np.where(fold_of == fold)[0]
        train_metrics = [metrics_list[i] for i in train_idx]
        train_context = [context_list[i] for i in train_idx]

        bucket_s1, bucket_s2 = make_buckets(train_metrics, train_context)
        tables = fit_sequential_tables(
            train_metrics, train_context, lambda_, bucket_s1, bucket_s2, min_bucket_size
        )
        last_tables, last_train_metrics, last_train_context = tables, train_metrics, train_context

        for i in test_idx:
            result = apply_sequential_policy(metrics_list[i], context_list[i], tables, lambda_)
            gap[i] = result.gap
            time[i] = result.total_time
            path_end[i] = result.path[-1]

    return gap, time, path_end, last_tables, last_train_metrics, last_train_context


def report_bucket_occupancy(tables, train_metrics, train_context, label):
    s1_counts = Counter(tables.bucket_s1(m, c) for m, c in zip(train_metrics, train_context, strict=True))
    s2_counts = Counter(tables.bucket_s2(m, c) for m, c in zip(train_metrics, train_context, strict=True))
    print(f"  [{label}] s1 bucket occupancy (one representative fold): {dict(s1_counts)}")
    print(f"  [{label}] s2 bucket occupancy (one representative fold): {dict(s2_counts)}")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    seed = config["seed"]
    cv_folds = config["evaluation"]["cv_folds"]
    lambda_grid = config["evaluation"]["lambda_grid"]

    wide = load_wide_cache()
    metrics_list, lb_list, tightness_list = build_metrics_list(wide)
    n = len(metrics_list)
    fold_of = five_fold_assignment(n, seed, cv_folds)

    print(f"n={n} instances, {cv_folds}-fold CV\n")

    print("Diagnostic: full-policy path endpoints across the lambda grid (sanity check")
    print("that the policy isn't degenerate — it should shift from mostly-exact at low")
    print("lambda to mostly-cheap at high lambda, matching the hour-8 gate's oracle shares):")
    diversity_by_lambda = {}
    for lam in lambda_grid:
        g, t, path_end, _, _, _ = run_cv(
            metrics_list, lb_list, fold_of, cv_folds, lam, make_full_buckets, MIN_BUCKET_SIZE
        )
        counts = Counter(path_end)
        shares = {r: counts.get(r, 0) / n for r in ["greedy", "medium", "exact"]}
        diversity_by_lambda[lam] = 1.0 - max(shares.values())  # 0 = degenerate, higher = more mixed
        shares_str = ", ".join(f"{r}={s:.0%}" for r, s in shares.items())
        print(f"  lambda={lam:>6}: gap={g.mean():.4f}  time={t.mean():.5f}s  endpoints: {shares_str}")

    # The mid-of-grid lambda (used for the Phase 3 Pareto plot) can land in a
    # degenerate "always escalate" regime where full and ablation are
    # identical by construction — not informative for isolating the value of
    # purchased evidence. Use the lambda with the most three-way-mixed
    # endpoints instead: that's where escalation decisions actually differ.
    ablation_lambda = max(diversity_by_lambda, key=diversity_by_lambda.get)
    print(
        f"\nMost mixed lambda={ablation_lambda} (endpoint diversity={diversity_by_lambda[ablation_lambda]:.2f}) "
        "chosen for the full-vs-ablation comparison below.\n"
    )

    print(f"Full evaluation at lambda={ablation_lambda}:")
    full_gap, full_time, full_path_end, tables, tm, tc = run_cv(
        metrics_list, lb_list, fold_of, cv_folds, ablation_lambda, make_full_buckets, MIN_BUCKET_SIZE
    )
    report_bucket_occupancy(tables, tm, tc, "full")
    print(f"  full policy: mean gap={full_gap.mean():.4f}  mean time={full_time.mean():.5f}s")
    print(f"  path endpoints: {dict(Counter(full_path_end))}\n")

    ablation_gap, ablation_time, ablation_path_end, ab_tables, ab_tm, ab_tc = run_cv(
        metrics_list, tightness_list, fold_of, cv_folds, ablation_lambda, make_ablation_buckets, MIN_BUCKET_SIZE
    )
    report_bucket_occupancy(ab_tables, ab_tm, ab_tc, "ablation (features-only)")
    print(f"  ablation policy: mean gap={ablation_gap.mean():.4f}  mean time={ablation_time.mean():.5f}s")
    print(f"  path endpoints: {dict(Counter(ablation_path_end))}\n")

    # Cost-accounting sanity check on the actual cache (not just the unit test).
    path_order = ["greedy", "medium", "exact"]
    for i in range(n):
        rung = full_path_end[i]
        expected_time = sum(metrics_list[i][r].solve_time for r in path_order[: path_order.index(rung) + 1])
        assert abs(expected_time - full_time[i]) < 1e-9, f"cost accounting mismatch at instance {i}"
    print("Cost-accounting check passed: total_time matches sum of solve_time over the actual path taken.\n")

    print("Value of purchased information (full - ablation, negative = full is better):")
    print(f"  gap difference:  {full_gap.mean() - ablation_gap.mean():+.4f}")
    print(f"  time difference: {full_time.mean() - ablation_time.mean():+.5f}s")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (label, gap, time) in zip(
        axes, [("full (observed outcomes)", full_gap, full_time), ("ablation (features-only)", ablation_gap, ablation_time)], strict=True
    ):
        ax.scatter(time, gap, alpha=0.3, s=15)
        ax.set_xscale("log")
        ax.set_xlabel("solve time (s, log scale)")
        ax.set_ylabel("quality gap")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Sequential escalation: value of purchased information (lambda={ablation_lambda})")
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "sequential_ablation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
