import numpy as np

from rungs.domains.facility_location import generate_dataset


def test_generate_dataset_returns_requested_count_skipping_invalid():
    calls = []

    def is_valid(inst):
        calls.append(inst)
        # Reject the first two draws deterministically by a cheap property.
        return len(calls) > 2

    rng = np.random.default_rng(0)
    dataset = generate_dataset(
        rng,
        n_instances=5,
        n_sites=6,
        n_customers=15,
        tightness_range=(1.02, 2.5),
        demand_cv_range=(0.1, 0.8),
        fixed_transport_ratio_range=(0.2, 5.0),
        is_valid=is_valid,
    )
    assert len(dataset) == 5
    assert len(calls) == 7  # 2 rejected + 5 accepted


def test_generate_dataset_is_reproducible_given_same_seed():
    def is_valid(inst):
        return True

    dataset_a = generate_dataset(
        np.random.default_rng(42),
        n_instances=3,
        n_sites=6,
        n_customers=15,
        tightness_range=(1.02, 2.5),
        demand_cv_range=(0.1, 0.8),
        fixed_transport_ratio_range=(0.2, 5.0),
        is_valid=is_valid,
    )
    dataset_b = generate_dataset(
        np.random.default_rng(42),
        n_instances=3,
        n_sites=6,
        n_customers=15,
        tightness_range=(1.02, 2.5),
        demand_cv_range=(0.1, 0.8),
        fixed_transport_ratio_range=(0.2, 5.0),
        is_valid=is_valid,
    )
    for a, b in zip(dataset_a, dataset_b, strict=True):
        assert a.meta == b.meta
        assert np.array_equal(a.demand, b.demand)
