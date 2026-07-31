from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FLInstance:
    site_xy: np.ndarray  # (n_sites, 2)
    cust_xy: np.ndarray  # (n_cust, 2)
    demand: np.ndarray  # (n_cust,)
    capacity: np.ndarray  # (n_sites,)
    fixed_cost: np.ndarray  # (n_sites,)
    meta: dict = field(default_factory=dict)  # generation params, analysis only — never a feature


def _sample_tightness(rng: np.random.Generator, tightness_range: tuple[float, float]) -> float:
    lo, hi = tightness_range
    u = rng.uniform(0.0, 1.0)
    return lo + (hi - lo) * u**2.5


def _sample_customer_positions(rng: np.random.Generator, n_customers: int) -> np.ndarray:
    if rng.uniform() < 0.5:
        return rng.uniform(0.0, 1.0, size=(n_customers, 2))
    n_clusters = 3
    centers = rng.uniform(0.2, 0.8, size=(n_clusters, 2))
    assignments = rng.integers(0, n_clusters, size=n_customers)
    points = centers[assignments] + rng.normal(0.0, 0.08, size=(n_customers, 2))
    return np.clip(points, 0.0, 1.0)


def generate_instance(
    rng: np.random.Generator,
    n_sites: int,
    n_customers: int,
    tightness_range: tuple[float, float],
    demand_cv_range: tuple[float, float],
    fixed_transport_ratio_range: tuple[float, float],
) -> FLInstance:
    tightness = _sample_tightness(rng, tightness_range)
    demand_cv = rng.uniform(*demand_cv_range)
    fixed_transport_ratio = rng.uniform(*fixed_transport_ratio_range)

    site_xy = rng.uniform(0.0, 1.0, size=(n_sites, 2))
    cust_xy = _sample_customer_positions(rng, n_customers)

    sigma = np.sqrt(np.log(1.0 + demand_cv**2))
    mu = -0.5 * sigma**2
    demand = rng.lognormal(mean=mu, sigma=sigma, size=n_customers)
    demand = demand / demand.sum() * 100.0

    total_capacity = tightness * demand.sum()
    capacity_shares = rng.dirichlet(np.full(n_sites, 5.0))
    capacity = capacity_shares * total_capacity

    site_cust_dist = np.linalg.norm(
        site_xy[:, None, :] - cust_xy[None, :, :], axis=-1
    )
    mean_nearest_dist = site_cust_dist.min(axis=0).mean()
    mean_transport_cost = mean_nearest_dist * demand.mean()
    mean_fixed_cost = fixed_transport_ratio * mean_transport_cost
    fixed_cost = rng.uniform(0.5, 1.5, size=n_sites) * mean_fixed_cost

    meta = {
        "tightness": tightness,
        "demand_cv": demand_cv,
        "fixed_transport_ratio": fixed_transport_ratio,
    }

    return FLInstance(
        site_xy=site_xy,
        cust_xy=cust_xy,
        demand=demand,
        capacity=capacity,
        fixed_cost=fixed_cost,
        meta=meta,
    )


def generate_dataset(
    rng: np.random.Generator,
    n_instances: int,
    n_sites: int,
    n_customers: int,
    tightness_range: tuple[float, float],
    demand_cv_range: tuple[float, float],
    fixed_transport_ratio_range: tuple[float, float],
    is_valid: Callable[[FLInstance], bool],
) -> list[FLInstance]:
    """Draw n_instances instances from rng, skipping any that fail is_valid.

    Used to build a fixed-size dataset where every instance has trustworthy
    exact ground truth (is_valid = "exact rung proved optimality"), without
    silently accepting an instance the exact solver couldn't verify.
    """
    dataset = []
    while len(dataset) < n_instances:
        inst = generate_instance(
            rng,
            n_sites=n_sites,
            n_customers=n_customers,
            tightness_range=tightness_range,
            demand_cv_range=demand_cv_range,
            fixed_transport_ratio_range=fixed_transport_ratio_range,
        )
        if is_valid(inst):
            dataset.append(inst)
    return dataset
