import numpy as np

from rungs.domains.facility_location import FLInstance


def cheap_lower_bound(inst: FLInstance) -> float:
    """A valid, no-solving lower bound on the true optimal objective.

    Serves customers at their nearest site (ignoring capacity — a strict
    relaxation) and assumes only the cheapest site need open (ignoring that
    capacity may force more). Both relaxations can only reduce cost, so the
    result never exceeds the true optimum.

    Used to bucket a solved rung's objective (e.g. greedy's) into a state
    signal for the sequential policy without needing the true optimum, which
    is exactly the number the policy doesn't have without paying for exact.
    """
    site_cust_dist = np.linalg.norm(
        inst.site_xy[:, None, :] - inst.cust_xy[None, :, :], axis=-1
    )
    nearest_site_dist = site_cust_dist.min(axis=0)
    transport_lb = float((nearest_site_dist * inst.demand).sum())
    fixed_lb = float(inst.fixed_cost.min())
    return transport_lb + fixed_lb


def features(inst: FLInstance) -> dict[str, float]:
    """Cheap, no-solving features. Must never read inst.meta (see DESIGN.md §7)."""
    demand = inst.demand
    capacity = inst.capacity
    fixed_cost = inst.fixed_cost

    tightness = capacity.sum() / demand.sum()

    site_cust_dist = np.linalg.norm(
        inst.site_xy[:, None, :] - inst.cust_xy[None, :, :], axis=-1
    )
    nearest_site_dist = site_cust_dist.min(axis=0)
    mean_transport_cost = nearest_site_dist.mean() * demand.mean()

    return {
        "tightness": float(tightness),
        "log_tightness_minus_1": float(np.log(tightness - 1.0)),
        "demand_mean": float(demand.mean()),
        "demand_std": float(demand.std()),
        "demand_cv": float(demand.std() / demand.mean()),
        "demand_max_mean_ratio": float(demand.max() / demand.mean()),
        "capacity_mean": float(capacity.mean()),
        "capacity_std": float(capacity.std()),
        "capacity_min_mean_ratio": float(capacity.min() / capacity.mean()),
        "fixed_cost_mean": float(fixed_cost.mean()),
        "fixed_transport_ratio": float(fixed_cost.mean() / mean_transport_cost),
        "n_sites": float(inst.site_xy.shape[0]),
        "n_customers": float(inst.cust_xy.shape[0]),
        "nearest_site_dist_mean": float(nearest_site_dist.mean()),
        "nearest_site_dist_var": float(nearest_site_dist.var()),
    }
