"""Phase 5: lambda sweep, gain-fraction table, and all figures.

See internal/DESIGN.md §11 and internal/IMPLEMENTATION.md §9.
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from rungs.evaluate import gain_fraction, loss
from rungs.policies import OneShotAllocator
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
PATH_ORDER = ["greedy", "medium", "exact"]
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
MIN_BUCKET_SIZE = 20

MARKERS = {
    "always_exact": "o",
    "always_medium": "s",
    "always_greedy": "^",
    "random": "x",
    "one_shot": "*",
    "sequential": "P",
    "oracle": "D",
}


def five_fold_assignment(n: int, seed: int, k: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(n)
    fold_of = np.empty(n, dtype=int)
    fold_of[perm] = np.arange(n) % k
    return fold_of


def load_wide_cache() -> pl.DataFrame:
    df = pl.read_parquet(CACHE_PATH)
    metrics = df.pivot(
        on="rung",
        index="instance_id",
        values=["quality_gap", "solve_time", "objective", "feasible"],
    ).sort("instance_id")
    feats = (
        df.select(["instance_id", "cheap_lower_bound", *FEATURE_COLUMNS])
        .unique(subset=["instance_id"])
        .sort("instance_id")
    )
    return metrics.join(feats, on="instance_id")


def build_metrics_list(wide: pl.DataFrame) -> list[dict[str, RungMetrics]]:
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
    return metrics_list


def make_full_buckets(train_metrics, train_lb):
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


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    seed = config["seed"]
    cv_folds = config["evaluation"]["cv_folds"]
    lambda_grid = config["evaluation"]["lambda_grid"]

    wide = load_wide_cache()
    metrics_list = build_metrics_list(wide)
    lb_list = wide["cheap_lower_bound"].to_list()
    X = wide.select(FEATURE_COLUMNS).to_numpy()
    n = len(metrics_list)
    fold_of = five_fold_assignment(n, seed, cv_folds)

    gaps = {r: np.array([m[r].gap for m in metrics_list]) for r in RUNG_NAMES}
    times = {r: np.array([m[r].solve_time for m in metrics_list]) for r in RUNG_NAMES}

    # --- Fit one-shot models once per fold (lambda-independent); re-evaluate
    # choose() cheaply across the whole lambda grid. ---
    one_shot_allocators = {}
    for fold in range(cv_folds):
        train_mask = fold_of != fold
        allocator = OneShotAllocator(RUNG_NAMES)
        allocator.fit(
            X[train_mask],
            {r: gaps[r][train_mask] for r in RUNG_NAMES},
            {r: times[r][train_mask] for r in RUNG_NAMES},
        )
        one_shot_allocators[fold] = allocator

    def one_shot_gap_time(lam: float) -> tuple[np.ndarray, np.ndarray]:
        g = np.empty(n)
        t = np.empty(n)
        for fold in range(cv_folds):
            allocator = one_shot_allocators[fold]
            for i in np.where(fold_of == fold)[0]:
                choice = allocator.choose(X[i], lam)
                g[i] = gaps[choice][i]
                t[i] = times[choice][i]
        return g, t

    def sequential_gap_time(lam: float) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
        g = np.empty(n)
        t = np.empty(n)
        path_end: list[str | None] = [None] * n
        for fold in range(cv_folds):
            train_idx = np.where(fold_of != fold)[0]
            test_idx = np.where(fold_of == fold)[0]
            train_metrics = [metrics_list[i] for i in train_idx]
            train_lb = [lb_list[i] for i in train_idx]
            bucket_s1, bucket_s2 = make_full_buckets(train_metrics, train_lb)
            tables = fit_sequential_tables(
                train_metrics, train_lb, lam, bucket_s1, bucket_s2, MIN_BUCKET_SIZE
            )
            for i in test_idx:
                result = apply_sequential_policy(metrics_list[i], lb_list[i], tables, lam)
                g[i] = result.gap
                t[i] = result.total_time
                path_end[i] = result.path[-1]
        return g, t, path_end

    def best_fixed_and_oracle(lam: float, idx: np.ndarray) -> tuple[float, float]:
        rung_losses = {r: loss(gaps[r][idx], times[r][idx], lam) for r in RUNG_NAMES}
        best_fixed = min(np.mean(rung_losses[r]) for r in RUNG_NAMES)
        oracle = np.mean(np.minimum.reduce([rung_losses[r] for r in RUNG_NAMES]))
        return float(best_fixed), float(oracle)

    print(f"n={n} instances, {cv_folds}-fold CV\n")

    # Gain fraction divides by (best_fixed - oracle) headroom. When headroom
    # is below the hour-8 gate's own 5% exploitability threshold (Phase 2),
    # the fraction is dividing by noise and blows up (a policy loss that
    # differs from best_fixed by a hair, relative to a headroom that's
    # basically zero, can read as +-100s). This is not a bug — it is exactly
    # the boundary condition the hour-8 gate exists to flag. Use the FULL
    # 800-instance headroom (stable) to decide meaningfulness per lambda, and
    # report per-fold gain fraction only where it clears that bar.
    HEADROOM_THRESHOLD = 0.05
    all_idx = np.arange(n)

    print("Gain-fraction table (mean +/- std across CV folds; N/A below the")
    print("hour-8 gate's 5% headroom threshold, where the fraction is not meaningful):")
    print(f"{'lambda':>8}  {'headroom%':>10}  {'one_shot':>18}  {'sequential':>18}")

    gain_table = {}
    diversity_by_lambda = {}
    one_shot_cache = {}
    sequential_cache = {}
    for lam in lambda_grid:
        os_gap, os_time = one_shot_gap_time(lam)
        seq_gap, seq_time, seq_path_end = sequential_gap_time(lam)
        one_shot_cache[lam] = (os_gap, os_time)
        sequential_cache[lam] = (seq_gap, seq_time)

        counts = Counter(seq_path_end)
        shares = {r: counts.get(r, 0) / n for r in PATH_ORDER}
        diversity_by_lambda[lam] = 1.0 - max(shares.values())

        full_bf, full_oc = best_fixed_and_oracle(lam, all_idx)
        headroom_pct = (full_bf - full_oc) / full_bf if full_bf > 0 else 0.0
        meaningful = headroom_pct >= HEADROOM_THRESHOLD

        os_fold_gains = []
        seq_fold_gains = []
        if meaningful:
            for fold in range(cv_folds):
                idx = np.where(fold_of == fold)[0]
                bf, oc = best_fixed_and_oracle(lam, idx)
                os_loss = float(loss(os_gap[idx], os_time[idx], lam).mean())
                seq_loss = float(loss(seq_gap[idx], seq_time[idx], lam).mean())
                os_fold_gains.append(gain_fraction(bf, os_loss, oc))
                seq_fold_gains.append(gain_fraction(bf, seq_loss, oc))

        gain_table[lam] = (os_fold_gains, seq_fold_gains, meaningful)
        if meaningful:
            print(
                f"{lam:>8}  {headroom_pct:>9.1%}  "
                f"{np.mean(os_fold_gains):>7.3f} +/- {np.std(os_fold_gains):<6.3f}   "
                f"{np.mean(seq_fold_gains):>7.3f} +/- {np.std(seq_fold_gains):<6.3f}"
            )
        else:
            print(f"{lam:>8}  {headroom_pct:>9.1%}  {'N/A (no exploitable headroom)':>18}")

    figure_lambda = max(diversity_by_lambda, key=lambda lam: diversity_by_lambda[lam])
    print(f"\nUsing lambda={figure_lambda} (most three-way-mixed) for Figures 1 and 3.\n")

    # --- Figure 1: Pareto frontier, all 7 policies ---
    os_gap, os_time = one_shot_cache[figure_lambda]
    seq_gap, seq_time = sequential_cache[figure_lambda]
    points = {f"always_{r}": (gaps[r].mean(), times[r].mean()) for r in RUNG_NAMES}
    points["random"] = (
        np.mean([points[f"always_{r}"][0] for r in RUNG_NAMES]),
        np.mean([points[f"always_{r}"][1] for r in RUNG_NAMES]),
    )
    points["one_shot"] = (os_gap.mean(), os_time.mean())
    points["sequential"] = (seq_gap.mean(), seq_time.mean())
    oracle_choice = np.argmin(
        np.stack([loss(gaps[r], times[r], figure_lambda) for r in RUNG_NAMES]), axis=0
    )
    oracle_gap = np.array([gaps[RUNG_NAMES[c]][i] for i, c in enumerate(oracle_choice)])
    oracle_time = np.array([times[RUNG_NAMES[c]][i] for i, c in enumerate(oracle_choice)])
    points["oracle"] = (oracle_gap.mean(), oracle_time.mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (g, t) in points.items():
        ax.scatter(t, g, label=name, marker=MARKERS[name], s=100)
    ax.set_xscale("log")
    ax.set_xlabel("mean solve time (s, log scale)")
    ax.set_ylabel("mean quality gap")
    ax.set_title(f"Pareto frontier, all 7 policies (lambda={figure_lambda}, n={n})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig1_pareto_all_policies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_pareto_all_policies.png")

    # --- Figure 2: gain fraction vs lambda ---
    # Only lambdas clearing the hour-8 exploitability threshold are plotted;
    # below it, gain fraction divides by near-zero headroom and is not a
    # meaningful number (see the printed table above).
    meaningful_lambdas = [lam for lam in lambda_grid if gain_table[lam][2]]
    fig, ax = plt.subplots(figsize=(7, 5))
    os_means = [np.mean(gain_table[lam][0]) for lam in meaningful_lambdas]
    os_stds = [np.std(gain_table[lam][0]) for lam in meaningful_lambdas]
    seq_means = [np.mean(gain_table[lam][1]) for lam in meaningful_lambdas]
    seq_stds = [np.std(gain_table[lam][1]) for lam in meaningful_lambdas]
    ax.errorbar(
        meaningful_lambdas,
        os_means,
        yerr=os_stds,
        marker="*",
        linestyle="-",
        label="one_shot",
        capsize=3,
    )
    ax.errorbar(
        meaningful_lambdas,
        seq_means,
        yerr=seq_stds,
        marker="P",
        linestyle="--",
        label="sequential",
        capsize=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("gain fraction (0=best fixed, 1=oracle)")
    ax.set_title(
        "Gain fraction vs lambda, with CV error bars\n(only lambdas with >=5% oracle headroom shown)"
    )
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig2_gain_fraction_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_gain_fraction_vs_lambda.png")

    # --- Figure 3: information-value ablation (bar chart) ---
    tightness_list = wide["tightness"].to_list()

    def make_ablation_buckets(_train_metrics, train_tightness):
        thresholds = compute_quantile_thresholds(train_tightness, n_bins=3)

        def bucket(m, tightness):
            return f"tightness_{quantile_bucket(tightness, thresholds)}"

        return bucket, bucket

    ab_gap = np.empty(n)
    ab_time = np.empty(n)
    for fold in range(cv_folds):
        train_idx = np.where(fold_of != fold)[0]
        test_idx = np.where(fold_of == fold)[0]
        train_metrics = [metrics_list[i] for i in train_idx]
        train_tightness = [tightness_list[i] for i in train_idx]
        bucket_s1, bucket_s2 = make_ablation_buckets(train_metrics, train_tightness)
        tables = fit_sequential_tables(
            train_metrics, train_tightness, figure_lambda, bucket_s1, bucket_s2, MIN_BUCKET_SIZE
        )
        for i in test_idx:
            result = apply_sequential_policy(
                metrics_list[i], tightness_list[i], tables, figure_lambda
            )
            ab_gap[i] = result.gap
            ab_time[i] = result.total_time

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    labels = ["full\n(observed outcomes)", "ablation\n(features-only)"]
    bar_colors = ["#4c72b0", "#c44e52"]
    # #4c72b0 and #c44e52 have near-identical grayscale luminance (~110 both)
    # — add hatching so the two series stay distinguishable without color.
    hatches = ["", "///"]
    for ax, values, ylabel, title in [
        (axes[0], [seq_gap.mean(), ab_gap.mean()], "mean quality gap", "Quality gap"),
        (axes[1], [seq_time.mean(), ab_time.mean()], "mean solve time (s)", "Solve time"),
    ]:
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="black")
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    fig.suptitle(f"Value of purchased information (lambda={figure_lambda})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_information_value_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_information_value_ablation.png")

    # --- Figure 4: feature importance ---
    importances = np.zeros(len(FEATURE_COLUMNS))
    n_models = 0
    for allocator in one_shot_allocators.values():
        for rung in RUNG_NAMES:
            importances += allocator.models[rung].feature_importance(importance_type="gain")
            n_models += 1
    importances /= n_models
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [FEATURE_COLUMNS[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#4c72b0",
        edgecolor="black",
    )
    ax.set_xlabel("mean LightGBM gain importance (across rung-models and folds)")
    ax.set_title("Feature importance, one-shot selector")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_feature_importance.png")
    print(f"  top feature: {FEATURE_COLUMNS[order[0]]}")

    # --- Figure 5: frontier illustration (one flat instance, one steep) ---
    # "Flat" = the best cheap method is already near-optimal (small gap to
    # exact); "steep" = both cheap methods are far off, exact is essential.
    best_cheap_gap = np.minimum(gaps["greedy"], gaps["medium"])
    feasible_cheap = np.array([m["greedy"].feasible or m["medium"].feasible for m in metrics_list])
    candidates = np.where(feasible_cheap)[0]
    flat_idx = candidates[np.argmin(best_cheap_gap[candidates])]
    steep_idx = candidates[np.argmax(best_cheap_gap[candidates])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, idx, label in [
        (axes[0], flat_idx, "flat frontier"),
        (axes[1], steep_idx, "steep frontier"),
    ]:
        pts = [(float(times[r][idx]), float(gaps[r][idx]), r) for r in RUNG_NAMES]
        pts.sort(key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linestyle="-", color="#4c72b0")
        for pt_time, pt_gap, pt_rung in pts:
            ax.annotate(pt_rung, (pt_time, pt_gap), textcoords="offset points", xytext=(5, 5))
        ax.set_xscale("log")
        ax.set_xlabel("solve time (s, log scale)")
        ax.set_ylabel("quality gap")
        ax.set_title(f"{label} (instance {idx})")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Effort-quality frontier: flat vs steep instances")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_frontier_illustration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_frontier_illustration.png")


if __name__ == "__main__":
    main()
