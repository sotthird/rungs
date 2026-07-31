from time import perf_counter

import numpy as np
import pulp

from rungs.core import Result
from rungs.domains.facility_location import FLInstance

EXACT_TIME_LIMIT_S = 5.0


def _transport_cost_matrix(inst: FLInstance) -> np.ndarray:
    dist = np.linalg.norm(
        inst.site_xy[:, None, :] - inst.cust_xy[None, :, :], axis=-1
    )
    return dist * inst.demand[None, :]


def solve_exact(inst: FLInstance) -> Result:
    n_sites = inst.site_xy.shape[0]
    n_cust = inst.cust_xy.shape[0]
    cost = _transport_cost_matrix(inst)

    prob = pulp.LpProblem("facility_location", pulp.LpMinimize)
    y = prob.add_variable_dicts("y", range(n_sites), cat="Binary")
    x = prob.add_variable_dicts(
        "x", [(i, j) for i in range(n_sites) for j in range(n_cust)], cat="Binary"
    )

    prob += pulp.lpSum(
        inst.fixed_cost[i] * y[i] for i in range(n_sites)
    ) + pulp.lpSum(
        cost[i, j] * x[i, j] for i in range(n_sites) for j in range(n_cust)
    )

    for j in range(n_cust):
        prob += pulp.lpSum(x[i, j] for i in range(n_sites)) == 1

    for i in range(n_sites):
        prob += (
            pulp.lpSum(inst.demand[j] * x[i, j] for j in range(n_cust))
            <= inst.capacity[i] * y[i]
        )

    # Disaggregated linking constraint: tightens the LP relaxation dramatically
    # relative to the capacity constraint alone, which is essential for CBC to
    # close the branch-and-bound gap within the time limit at this instance size.
    for i in range(n_sites):
        for j in range(n_cust):
            prob += x[i, j] <= y[i]

    start = perf_counter()
    prob.solve(pulp.HiGHS(msg=False, timeLimit=EXACT_TIME_LIMIT_S))
    solve_time = perf_counter() - start

    # prob.status alone is unreliable: PuLP maps "stopped on time limit but a
    # feasible incumbent exists" to LpStatusOptimal too. The real optimality
    # signal is sol_status == LpSolutionOptimal (as opposed to
    # LpSolutionIntegerFeasible, i.e. a time-limited incumbent).
    if (
        prob.status != pulp.LpStatusOptimal
        or prob.sol_status != pulp.LpSolutionOptimal
    ):
        return Result(objective=None, feasible=False, solve_time=solve_time)

    y_val = np.array([y[i].value() for i in range(n_sites)])
    x_val = np.array(
        [[x[i, j].value() for j in range(n_cust)] for i in range(n_sites)]
    )
    return Result(
        objective=pulp.value(prob.objective),
        feasible=True,
        solve_time=solve_time,
        artifact=(y_val, x_val),
    )


def _greedy_assign(
    inst: FLInstance, cost: np.ndarray, open_sites: np.ndarray
) -> tuple[float | None, bool, np.ndarray]:
    n_sites = inst.site_xy.shape[0]
    n_cust = inst.cust_xy.shape[0]
    remaining_capacity = inst.capacity * open_sites
    x_val = np.zeros((n_sites, n_cust))
    feasible = True
    total_cost = 0.0

    cust_order = np.argsort(-inst.demand)
    for j in cust_order:
        candidates = [
            i
            for i in range(n_sites)
            if open_sites[i] and remaining_capacity[i] >= inst.demand[j]
        ]
        if not candidates:
            feasible = False
            continue
        i = min(candidates, key=lambda i: cost[i, j])
        x_val[i, j] = 1.0
        remaining_capacity[i] -= inst.demand[j]
        total_cost += cost[i, j]

    if not feasible:
        return None, False, x_val

    total_cost += float((inst.fixed_cost * open_sites).sum())
    return total_cost, True, x_val


def solve_medium(inst: FLInstance) -> Result:
    n_sites = inst.site_xy.shape[0]
    n_cust = inst.cust_xy.shape[0]
    cost = _transport_cost_matrix(inst)

    prob = pulp.LpProblem("facility_location_relax", pulp.LpMinimize)
    y = prob.add_variable_dicts("y", range(n_sites), lowBound=0, upBound=1)
    x = prob.add_variable_dicts(
        "x", [(i, j) for i in range(n_sites) for j in range(n_cust)], lowBound=0, upBound=1
    )

    prob += pulp.lpSum(
        inst.fixed_cost[i] * y[i] for i in range(n_sites)
    ) + pulp.lpSum(
        cost[i, j] * x[i, j] for i in range(n_sites) for j in range(n_cust)
    )

    for j in range(n_cust):
        prob += pulp.lpSum(x[i, j] for i in range(n_sites)) == 1

    for i in range(n_sites):
        prob += (
            pulp.lpSum(inst.demand[j] * x[i, j] for j in range(n_cust))
            <= inst.capacity[i] * y[i]
        )

    start = perf_counter()
    prob.solve(pulp.HiGHS(msg=False))
    y_relaxed = np.array([y[i].value() for i in range(n_sites)])

    open_sites = (y_relaxed >= 0.5).astype(float)
    if open_sites.sum() == 0:
        open_sites[int(np.argmax(y_relaxed))] = 1.0

    objective, feasible, x_val = _greedy_assign(inst, cost, open_sites)
    solve_time = perf_counter() - start

    if not feasible:
        return Result(objective=None, feasible=False, solve_time=solve_time)

    return Result(
        objective=objective,
        feasible=True,
        solve_time=solve_time,
        artifact=(open_sites, x_val),
    )


def solve_greedy(inst: FLInstance) -> Result:
    n_sites = inst.site_xy.shape[0]
    cost = _transport_cost_matrix(inst)
    mean_transport_to_unserved = cost.mean(axis=1)

    start = perf_counter()
    open_sites = np.zeros(n_sites)
    closed = list(range(n_sites))
    while open_sites.sum() == 0 or (inst.capacity * open_sites).sum() < inst.demand.sum():
        if not closed:
            break
        scores = {
            i: inst.capacity[i]
            / (inst.fixed_cost[i] + mean_transport_to_unserved[i])
            for i in closed
        }
        best = max(scores, key=scores.get)
        open_sites[best] = 1.0
        closed.remove(best)

    objective, feasible, x_val = _greedy_assign(inst, cost, open_sites)
    solve_time = perf_counter() - start

    if not feasible:
        return Result(objective=None, feasible=False, solve_time=solve_time)

    return Result(
        objective=objective,
        feasible=True,
        solve_time=solve_time,
        artifact=(open_sites, x_val),
    )
