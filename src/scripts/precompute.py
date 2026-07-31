"""Run all rungs on all instances once and cache the results. See internal/IMPLEMENTATION.md §4.

The cache is sacred: everything downstream (selector training, sequential
policy, baselines, the lambda sweep, every plot) reads this file. Nothing
re-solves.

A thin driver over the library (rungs.core.Domain) — see internal/TODO.md
Phase 6. Solving is routed entirely through the domain's declared rungs, not
hardcoded solver imports, so switching domains means editing DOMAIN only.
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from rungs.domains.facility_location import generate_dataset
from rungs.domains.facility_location_domain import FacilityLocationDomain
from rungs.evaluate import quality_gap

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"
CACHE_PATH = ROOT / "data" / "cache.parquet"


def build_domain(config: dict) -> FacilityLocationDomain:
    inst_cfg = config["instance"]
    return FacilityLocationDomain(
        n_sites=inst_cfg["n_sites"],
        n_customers=inst_cfg["n_customers"],
        tightness_range=tuple(inst_cfg["tightness_range"]),
        demand_cv_range=tuple(inst_cfg["demand_cv_range"]),
        fixed_transport_ratio_range=tuple(inst_cfg["fixed_transport_ratio_range"]),
    )


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    rng = np.random.default_rng(config["seed"])
    gap_infeasible = config["quality"]["gap_infeasible"]
    n_instances = config["n_instances"]

    domain = build_domain(config)
    exact_rung = domain.ground_truth_rung()

    # exact_rung.solve() runs here, inside the validity check, so it is never
    # solved twice: an accepted instance's exact Result is captured for reuse.
    exact_results: list = []

    def is_valid_ground_truth(inst) -> bool:
        result = exact_rung.solve(inst)
        if result.feasible:
            exact_results.append(result)
        return result.feasible

    print(f"Generating {n_instances} instances with verified exact ground truth...")
    start = time.perf_counter()
    dataset = generate_dataset(
        rng,
        n_instances=n_instances,
        n_sites=domain.n_sites,
        n_customers=domain.n_customers,
        tightness_range=domain.tightness_range,
        demand_cv_range=domain.demand_cv_range,
        fixed_transport_ratio_range=domain.fixed_transport_ratio_range,
        is_valid=is_valid_ground_truth,
    )
    print(f"  {len(dataset)} instances kept ({time.perf_counter() - start:.1f}s)")

    rows = []
    feature_times = []
    for instance_id, (inst, exact_result) in enumerate(zip(dataset, exact_results, strict=True)):
        if exact_result.solve_time > 0.9 * 5.0:
            print(
                f"  warning: instance {instance_id} exact solve took {exact_result.solve_time:.2f}s, near the time limit"
            )

        feat_start = time.perf_counter()
        feats = domain.features(inst)
        feature_times.append(time.perf_counter() - feat_start)
        lb = domain.cheap_lower_bound(inst)

        for rung in domain.rungs():
            result = exact_result if rung.name == exact_rung.name else rung.solve(inst)
            gap = quality_gap(
                result.objective, result.feasible, exact_result.objective, gap_infeasible
            )
            rows.append(
                {
                    "instance_id": instance_id,
                    "rung": rung.name,
                    "objective": result.objective,
                    "feasible": result.feasible,
                    "solve_time": result.solve_time,
                    "quality_gap": gap,
                    "tightness_true": inst.meta["tightness"],
                    "demand_cv_true": inst.meta["demand_cv"],
                    "fixed_transport_ratio_true": inst.meta["fixed_transport_ratio"],
                    "cheap_lower_bound": lb,
                    **feats,
                }
            )
        if (instance_id + 1) % 100 == 0:
            print(
                f"  solved {instance_id + 1}/{len(dataset)} ({time.perf_counter() - start:.1f}s elapsed)"
            )

    df = pl.DataFrame(rows)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(CACHE_PATH)

    total_time = time.perf_counter() - start
    # polars' Series.max()/median() are typed as a broad literal union covering
    # every possible dtype (not just numeric) — this column is always float.
    max_exact_time = float(df.filter(pl.col("rung") == exact_rung.name)["solve_time"].max())  # type: ignore[arg-type]
    median_greedy_time = float(df.filter(pl.col("rung") == "greedy")["solve_time"].median())  # type: ignore[arg-type]
    median_feature_time = float(np.median(feature_times))
    print(f"Wrote {len(df)} rows to {CACHE_PATH} in {total_time:.1f}s total")
    print(f"Max exact solve time among kept instances: {max_exact_time:.3f}s")
    print(
        f"Median feature-extraction time: {median_feature_time * 1000:.3f}ms "
        f"vs median greedy solve time: {median_greedy_time * 1000:.3f}ms"
    )


if __name__ == "__main__":
    main()
