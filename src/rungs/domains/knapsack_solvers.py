from time import perf_counter

import numpy as np

from rungs.core import Result
from rungs.domains.knapsack import KnapsackInstance


def solve_exact(inst: KnapsackInstance) -> Result:
    """0/1 knapsack via dynamic programming. Exact, pseudo-polynomial in
    capacity — fast at the instance sizes this domain is used at."""
    n = len(inst.values)
    capacity = inst.capacity
    start = perf_counter()

    dp = np.zeros((n + 1, capacity + 1))
    for i in range(1, n + 1):
        w, v = inst.weights[i - 1], inst.values[i - 1]
        dp[i, :w] = dp[i - 1, :w]
        dp[i, w:] = np.maximum(dp[i - 1, w:], dp[i - 1, : capacity + 1 - w] + v)

    chosen = np.zeros(n, dtype=int)
    c = capacity
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]:
            chosen[i - 1] = 1
            c -= inst.weights[i - 1]

    solve_time = perf_counter() - start
    return Result(
        objective=float(dp[n, capacity]), feasible=True, solve_time=solve_time, artifact=chosen
    )


def solve_greedy(inst: KnapsackInstance) -> Result:
    start = perf_counter()
    ratio = inst.values / inst.weights
    order = np.argsort(-ratio)

    chosen = np.zeros(len(inst.values), dtype=int)
    remaining = inst.capacity
    for i in order:
        if inst.weights[i] <= remaining:
            chosen[i] = 1
            remaining -= inst.weights[i]

    objective = float((inst.values * chosen).sum())
    solve_time = perf_counter() - start
    return Result(objective=objective, feasible=True, solve_time=solve_time, artifact=chosen)


def solve_medium(inst: KnapsackInstance) -> Result:
    """Fractional relaxation (greedy-by-ratio fill) + round the boundary
    item at 0.5, analogous to facility location's LP-relax-and-round."""
    start = perf_counter()
    ratio = inst.values / inst.weights
    order = np.argsort(-ratio)

    chosen = np.zeros(len(inst.values), dtype=int)
    remaining = inst.capacity
    for i in order:
        if inst.weights[i] <= remaining:
            chosen[i] = 1
            remaining -= inst.weights[i]
        else:
            fractional_amount = remaining / inst.weights[i]
            if fractional_amount >= 0.5 and inst.weights[i] <= inst.capacity:
                # Round up: make room by dropping the worst already-chosen
                # item(s) (lowest ratio first) until it fits.
                chosen[i] = 1
                needed = inst.weights[i] - remaining
                for j in order[::-1]:
                    if needed <= 0:
                        break
                    if chosen[j] == 1 and j != i:
                        chosen[j] = 0
                        needed -= inst.weights[j]
                remaining = inst.capacity - int((inst.weights * chosen).sum())
            break

    objective = float((inst.values * chosen).sum())
    solve_time = perf_counter() - start
    return Result(objective=objective, feasible=True, solve_time=solve_time, artifact=chosen)
