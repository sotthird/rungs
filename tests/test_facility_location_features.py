from dataclasses import replace

import numpy as np
import pytest

from rungs.domains.facility_location import generate_instance
from rungs.domains.facility_location_features import cheap_lower_bound, features


@pytest.fixture
def instance():
    rng = np.random.default_rng(0)
    return generate_instance(
        rng,
        n_sites=6,
        n_customers=15,
        tightness_range=(1.02, 2.5),
        demand_cv_range=(0.1, 0.8),
        fixed_transport_ratio_range=(0.2, 5.0),
    )


def test_features_ignore_meta(instance):
    stripped = replace(instance, meta={})
    assert features(stripped) == features(instance)


def test_features_contains_expected_keys(instance):
    keys = set(features(instance))
    assert keys == {
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
    }


def test_tightness_feature_matches_capacity_over_demand(instance):
    expected = instance.capacity.sum() / instance.demand.sum()
    assert features(instance)["tightness"] == pytest.approx(expected)


def test_cheap_lower_bound_never_exceeds_true_optimum():
    from rungs.domains.facility_location_solvers import solve_exact

    rng = np.random.default_rng(3)
    for _ in range(10):
        inst = generate_instance(
            rng,
            n_sites=6,
            n_customers=15,
            tightness_range=(1.02, 2.5),
            demand_cv_range=(0.1, 0.8),
            fixed_transport_ratio_range=(0.2, 5.0),
        )
        exact = solve_exact(inst)
        assert exact.feasible
        assert cheap_lower_bound(inst) <= exact.objective + 1e-6


def test_cheap_lower_bound_ignores_meta(instance):
    stripped = replace(instance, meta={})
    assert cheap_lower_bound(stripped) == cheap_lower_bound(instance)
