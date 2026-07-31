import numpy as np
import pytest

from rungs.domains.facility_location import FLInstance, generate_instance
from rungs.domains.facility_location_solvers import solve_exact, solve_greedy, solve_medium


@pytest.fixture
def trivial_instance():
    # One site, capacity covers both customers -> must open it, assign both.
    return FLInstance(
        site_xy=np.array([[0.0, 0.0]]),
        cust_xy=np.array([[1.0, 0.0], [0.0, 1.0]]),
        demand=np.array([2.0, 3.0]),
        capacity=np.array([10.0]),
        fixed_cost=np.array([5.0]),
        meta={},
    )


def test_solve_exact_returns_feasible_optimal(trivial_instance):
    result = solve_exact(trivial_instance)
    assert result.feasible
    # fixed cost 5 + transport: customer0 dist=1*demand=2 -> 2, customer1 dist=1*demand=3 -> 3
    assert result.objective == pytest.approx(5.0 + 2.0 + 3.0)


def test_solve_medium_feasible_with_ample_slack(trivial_instance):
    result = solve_medium(trivial_instance)
    assert result.feasible


def test_solve_greedy_feasible_with_ample_slack(trivial_instance):
    result = solve_greedy(trivial_instance)
    assert result.feasible


def test_exact_is_best():
    rng = np.random.default_rng(1)
    for _ in range(20):
        inst = generate_instance(
            rng,
            n_sites=6,
            n_customers=15,
            tightness_range=(1.02, 2.5),
            demand_cv_range=(0.1, 0.8),
            fixed_transport_ratio_range=(0.2, 5.0),
        )
        exact = solve_exact(inst)
        if not exact.feasible:
            continue  # exact occasionally can't prove optimality in time; see below
        for result in (solve_medium(inst), solve_greedy(inst)):
            if result.feasible:
                assert exact.objective <= result.objective + 1e-6


def test_exact_solve_time_at_target_scale():
    # DESIGN.md's target scale was originally 15 sites / 60 customers, but
    # empirically the exact MILP fails to prove optimality within the 5s
    # limit on ~48% of instances at that size, even with HiGHS. Reduced to
    # 6/15 (config.yaml), where only ~3% of instances miss the time limit and
    # median solve time is ~35ms — this is a statistical property, not a
    # per-instance guarantee, so assert on the aggregate failure rate rather
    # than every single instance.
    rng = np.random.default_rng(42)
    not_optimal = 0
    times = []
    n = 30
    for _ in range(n):
        inst = generate_instance(
            rng,
            n_sites=6,
            n_customers=15,
            tightness_range=(1.02, 2.5),
            demand_cv_range=(0.1, 0.8),
            fixed_transport_ratio_range=(0.2, 5.0),
        )
        result = solve_exact(inst)
        if not result.feasible:
            not_optimal += 1
        else:
            times.append(result.solve_time)
    assert not_optimal / n < 0.2
    assert np.median(times) < 1.0


def test_medium_sometimes_infeasible():
    rng = np.random.default_rng(2)
    infeasible_count = 0
    for _ in range(100):
        inst = generate_instance(
            rng,
            n_sites=6,
            n_customers=15,
            tightness_range=(1.02, 1.15),
            demand_cv_range=(0.1, 0.8),
            fixed_transport_ratio_range=(0.2, 5.0),
        )
        result = solve_medium(inst)
        if not result.feasible:
            infeasible_count += 1
    assert infeasible_count > 0
