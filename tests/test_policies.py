import numpy as np

from rungs.policies import OneShotAllocator


def test_choose_prefers_rung_with_lower_predicted_gap():
    rng = np.random.default_rng(0)
    n = 60
    X = rng.uniform(0, 1, size=(n, 1))
    # "good" rung has near-zero gap everywhere; "bad" rung has gap ~1 everywhere.
    gaps_per_rung = {
        "good": np.zeros(n) + rng.normal(0, 0.001, size=n),
        "bad": np.ones(n) + rng.normal(0, 0.001, size=n),
    }
    times_per_rung = {
        "good": np.full(n, 0.01),
        "bad": np.full(n, 0.01),
    }

    allocator = OneShotAllocator(
        rung_names=["good", "bad"], lgbm_params={"min_data_in_leaf": 1}, num_boost_round=20
    )
    allocator.fit(X, gaps_per_rung, times_per_rung)

    choice = allocator.choose(np.array([0.5]), lambda_=1.0)
    assert choice == "good"


def test_choose_accounts_for_time_when_gaps_are_similar():
    rng = np.random.default_rng(1)
    n = 60
    X = rng.uniform(0, 1, size=(n, 1))
    gaps_per_rung = {
        "fast": np.zeros(n) + rng.normal(0, 0.001, size=n),
        "slow": np.zeros(n) + rng.normal(0, 0.001, size=n),
    }
    times_per_rung = {
        "fast": np.full(n, 0.001),
        "slow": np.full(n, 10.0),
    }

    allocator = OneShotAllocator(
        rung_names=["fast", "slow"], lgbm_params={"min_data_in_leaf": 1}, num_boost_round=20
    )
    allocator.fit(X, gaps_per_rung, times_per_rung)

    # With gaps tied, a large lambda must break the tie toward the faster rung.
    choice = allocator.choose(np.array([0.5]), lambda_=1.0)
    assert choice == "fast"
