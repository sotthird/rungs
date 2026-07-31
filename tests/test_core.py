import numpy as np

from rungs.core import Allocator
from rungs.domains.facility_location_domain import FacilityLocationDomain
from rungs.domains.knapsack_domain import KnapsackDomain


def _fit_instances(n=60):
    domain = FacilityLocationDomain(
        n_sites=6,
        n_customers=15,
        tightness_range=(1.02, 2.5),
        demand_cv_range=(0.1, 0.8),
        fixed_transport_ratio_range=(0.2, 5.0),
    )
    rng = np.random.default_rng(0)
    instances = []
    while len(instances) < n:
        inst = domain.generate(rng)
        if domain.ground_truth_rung().solve(inst).feasible:
            instances.append(inst)
    return domain, instances


def test_domain_satisfies_protocol_shape():
    domain, instances = _fit_instances(n=5)
    assert {r.name for r in domain.rungs()} == {"exact", "medium", "greedy"}
    assert domain.ground_truth_rung().name == "exact"
    feats = domain.features(instances[0])
    assert isinstance(feats, dict)
    assert "tightness" in feats


def test_allocator_one_shot_only_solves_the_chosen_rung():
    domain, instances = _fit_instances(n=60)
    alloc = Allocator(domain, lambda_=10.0, mode="one_shot")
    alloc.fit(instances)

    test_domain, test_instances = _fit_instances(n=5)
    result = alloc.solve(test_instances[0])

    assert result.rung_used in {"exact", "medium", "greedy"}
    assert result.rungs_attempted == [result.rung_used]


def test_allocator_sequential_cost_is_cumulative_over_attempted_rungs():
    domain, instances = _fit_instances(n=60)
    alloc = Allocator(domain, lambda_=3.0, mode="sequential")
    alloc.fit(instances)

    _, test_instances = _fit_instances(n=5)
    result = alloc.solve(test_instances[0])

    assert result.rungs_attempted[0] == "greedy"
    assert result.rung_used == result.rungs_attempted[-1]
    assert result.total_time > 0


def _knapsack_instances(n=60):
    domain = KnapsackDomain(n_items=15, capacity_ratio_range=(0.2, 0.8))
    rng = np.random.default_rng(0)
    instances = [domain.generate(rng) for _ in range(n)]
    return domain, instances


def test_allocator_works_unmodified_with_a_second_domain_one_shot():
    # The whole point of Phase 7: core.py, policies.py, sequential.py are
    # untouched by adding knapsack. Only a new Domain implementation exists.
    domain, instances = _knapsack_instances(n=60)
    alloc = Allocator(domain, lambda_=0.001, mode="one_shot")
    alloc.fit(instances)

    _, test_instances = _knapsack_instances(n=5)
    result = alloc.solve(test_instances[0])
    assert result.rung_used in {"exact", "medium", "greedy"}
    assert result.feasible


def test_allocator_works_unmodified_with_a_second_domain_sequential():
    domain, instances = _knapsack_instances(n=60)
    alloc = Allocator(domain, lambda_=0.001, mode="sequential")
    alloc.fit(instances)

    _, test_instances = _knapsack_instances(n=5)
    result = alloc.solve(test_instances[0])
    assert result.rungs_attempted[0] == "greedy"
    assert result.feasible
