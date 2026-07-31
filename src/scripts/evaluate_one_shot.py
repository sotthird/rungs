"""One-shot selector + baselines + first Pareto plot. Milestone 4.

See internal/DESIGN.md §9, §11 and internal/IMPLEMENTATION.md §7.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from rungs.evaluate import loss
from rungs.policies import OneShotAllocator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"
CACHE_PATH = ROOT / "data" / "cache.parquet"
FIGURES_DIR = ROOT / "figures"

RUNG_NAMES = ["exact", "medium", "greedy"]
FEATURE_COLUMNS = [
    "tightness",
    "log_tightness_minus_1",
    "demand_mean",
    "demand_std",
    "demand_cv",
    "demand_max_mean_ratio",
    "capacity_mean",
    "capacity_std",
    "capacity_min_mean_ratio",
    "fixed_cost_mean",
    "fixed_transport_ratio",
    "n_sites",
    "n_customers",
    "nearest_site_dist_mean",
    "nearest_site_dist_var",
]


def five_fold_assignment(n: int, seed: int, k: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(n)
    fold_of = np.empty(n, dtype=int)
    fold_of[perm] = np.arange(n) % k
    return fold_of


def load_wide_cache() -> pl.DataFrame:
    df = pl.read_parquet(CACHE_PATH)
    metrics = df.pivot(on="rung", index="instance_id", values=["quality_gap", "solve_time"]).sort(
        "instance_id"
    )
    feats = (
        df.select(["instance_id", *FEATURE_COLUMNS])
        .unique(subset=["instance_id"])
        .sort("instance_id")
    )
    return metrics.join(feats, on="instance_id")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    seed = config["seed"]
    cv_folds = config["evaluation"]["cv_folds"]
    lambda_grid = config["evaluation"]["lambda_grid"]
    mid_lambda = lambda_grid[len(lambda_grid) // 2]

    wide = load_wide_cache()
    n = wide.height
    fold_of = five_fold_assignment(n, seed, cv_folds)

    X = wide.select(FEATURE_COLUMNS).to_numpy()
    gaps = {r: wide[f"quality_gap_{r}"].to_numpy() for r in RUNG_NAMES}
    times = {r: wide[f"solve_time_{r}"].to_numpy() for r in RUNG_NAMES}

    # One-shot policy: 5-fold CV, out-of-fold. Record the ACTUAL cached gap/time
    # of the chosen rung, never the model's predicted values.
    chosen_gap = np.empty(n)
    chosen_time = np.empty(n)
    for fold in range(cv_folds):
        train_mask = fold_of != fold
        test_idx = np.where(fold_of == fold)[0]

        allocator = OneShotAllocator(RUNG_NAMES)
        allocator.fit(
            X[train_mask],
            {r: gaps[r][train_mask] for r in RUNG_NAMES},
            {r: times[r][train_mask] for r in RUNG_NAMES},
        )
        for i in test_idx:
            choice = allocator.choose(X[i], mid_lambda)
            chosen_gap[i] = gaps[choice][i]
            chosen_time[i] = times[choice][i]

    results = {"one_shot": (chosen_gap.mean(), chosen_time.mean())}

    for r in RUNG_NAMES:
        results[f"always_{r}"] = (gaps[r].mean(), times[r].mean())

    # Random: uniform choice among rungs. By linearity of expectation this
    # equals the average of the three always-X means — exact, no sampling.
    results["random"] = (
        np.mean([results[f"always_{r}"][0] for r in RUNG_NAMES]),
        np.mean([results[f"always_{r}"][1] for r in RUNG_NAMES]),
    )

    oracle_gap = np.empty(n)
    oracle_time = np.empty(n)
    for i in range(n):
        losses = {r: loss(gaps[r][i], times[r][i], mid_lambda) for r in RUNG_NAMES}
        best = min(losses, key=lambda r: losses[r])
        oracle_gap[i] = gaps[best][i]
        oracle_time[i] = times[best][i]
    results["oracle"] = (oracle_gap.mean(), oracle_time.mean())

    print(f"n={n} instances, {cv_folds}-fold CV, mid lambda={mid_lambda}")
    print("Pareto summary (mean gap, mean time):")
    for name, (g, t) in results.items():
        print(f"  {name:>14}: gap={g:.4f}  time={t:.5f}s")

    one_shot_gap, one_shot_time = results["one_shot"]
    dominators = [
        name
        for name in (f"always_{r}" for r in RUNG_NAMES)
        if results[name][0] <= one_shot_gap
        and results[name][1] <= one_shot_time
        and (results[name][0] < one_shot_gap or results[name][1] < one_shot_time)
    ]
    if dominators:
        print(
            f"WARNING: one-shot is Pareto-dominated by fixed baseline(s) {dominators} "
            "(worse or equal on both gap and time) — debug before continuing."
        )
    else:
        print("one-shot is not dominated by any fixed baseline — on the frontier as expected.")

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"one_shot": "*", "oracle": "D", "random": "x"}
    for name, (g, t) in results.items():
        ax.scatter(t, g, label=name, marker=markers.get(name, "o"), s=100)
    ax.set_xscale("log")
    ax.set_xlabel("mean solve time (s, log scale)")
    ax.set_ylabel("mean quality gap")
    ax.set_title(f"Pareto frontier at lambda={mid_lambda} (n={n}, {cv_folds}-fold CV)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "pareto_first.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
