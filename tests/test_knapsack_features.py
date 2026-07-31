from dataclasses import replace

import numpy as np
import pytest

from rungs.domains.knapsack import generate_instance
from rungs.domains.knapsack_features import cheap_lower_bound, features
from rungs.domains.knapsack_solvers import solve_exact


@pytest.fixture
def instance():
    rng = np.random.default_rng(0)
    return generate_instance(rng, n_items=15, capacity_ratio_range=(0.3, 0.7))


def test_features_ignore_meta(instance):
    stripped = replace(instance, meta={})
    assert features(stripped) == features(instance)


def test_capacity_ratio_feature_matches_capacity_over_total_weight(instance):
    expected = instance.capacity / instance.weights.sum()
    assert features(instance)["capacity_ratio"] == pytest.approx(expected)


def test_cheap_lower_bound_never_exceeds_true_optimum():
    rng = np.random.default_rng(1)
    for _ in range(10):
        inst = generate_instance(rng, n_items=15, capacity_ratio_range=(0.2, 0.8))
        exact = solve_exact(inst)
        assert cheap_lower_bound(inst) <= exact.objective + 1e-6
