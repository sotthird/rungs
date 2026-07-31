import itertools

import numpy as np
import pytest

from rungs.domains.knapsack import KnapsackInstance, generate_instance
from rungs.domains.knapsack_solvers import solve_exact, solve_greedy, solve_medium


def _brute_force_optimal(inst: KnapsackInstance) -> float:
    n = len(inst.values)
    best = 0.0
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            weight = sum(inst.weights[i] for i in combo)
            if weight <= inst.capacity:
                value = sum(inst.values[i] for i in combo)
                best = max(best, value)
    return best


def test_solve_exact_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(10):
        inst = generate_instance(rng, n_items=10, capacity_ratio_range=(0.3, 0.7))
        result = solve_exact(inst)
        assert result.feasible
        assert result.objective == pytest.approx(_brute_force_optimal(inst))


def test_solve_exact_solution_respects_capacity():
    rng = np.random.default_rng(1)
    inst = generate_instance(rng, n_items=20, capacity_ratio_range=(0.3, 0.7))
    result = solve_exact(inst)
    chosen = result.artifact
    assert (inst.weights * chosen).sum() <= inst.capacity


def test_solve_greedy_and_medium_always_feasible():
    # DESIGN.md §3: knapsack has no natural infeasibility mode, unlike FL.
    rng = np.random.default_rng(2)
    for _ in range(30):
        inst = generate_instance(rng, n_items=15, capacity_ratio_range=(0.1, 0.9))
        for solver in (solve_greedy, solve_medium):
            result = solver(inst)
            assert result.feasible
            assert (inst.weights * result.artifact).sum() <= inst.capacity


def test_exact_is_at_least_as_good_as_medium_and_greedy():
    rng = np.random.default_rng(3)
    for _ in range(20):
        inst = generate_instance(rng, n_items=15, capacity_ratio_range=(0.2, 0.8))
        exact = solve_exact(inst)
        for solver in (solve_medium, solve_greedy):
            result = solver(inst)
            assert exact.objective >= result.objective - 1e-9
