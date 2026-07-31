from rungs.sequential import (
    RungMetrics,
    SequentialTables,
    apply_sequential_policy,
    compute_quantile_thresholds,
    fit_sequential_tables,
    quantile_bucket,
)


def test_escalation_cost_is_cumulative_across_all_three_rungs():
    # A policy that escalates greedy -> medium -> exact must report total time
    # as the SUM of all three rungs' solve times, not just the last one run.
    # This is risk #1 in DESIGN.md §17: it fails silently and still looks plausible.
    metrics = {
        "greedy": RungMetrics(objective=100.0, feasible=True, gap=0.5, solve_time=0.001),
        "medium": RungMetrics(objective=80.0, feasible=True, gap=0.2, solve_time=0.01),
        "exact": RungMetrics(objective=70.0, feasible=True, gap=0.0, solve_time=1.5),
    }

    # A table that always escalates, regardless of bucket.
    tables = SequentialTables(
        s1_decisions={"any": "escalate"},
        s2_decisions={"any": "escalate"},
        bucket_s1=lambda m, lb: "any",
        bucket_s2=lambda m, lb: "any",
    )

    result = apply_sequential_policy(metrics, lb=1.0, tables=tables, lambda_=1.0)

    assert result.total_time == 0.001 + 0.01 + 1.5
    assert result.gap == 0.0
    assert result.path == ["greedy", "medium", "exact"]


def test_accepting_at_greedy_only_pays_greedy_time():
    metrics = {
        "greedy": RungMetrics(objective=100.0, feasible=True, gap=0.5, solve_time=0.001),
        "medium": RungMetrics(objective=80.0, feasible=True, gap=0.2, solve_time=0.01),
        "exact": RungMetrics(objective=70.0, feasible=True, gap=0.0, solve_time=1.5),
    }
    tables = SequentialTables(
        s1_decisions={"any": "accept"},
        s2_decisions={"any": "accept"},
        bucket_s1=lambda m, lb: "any",
        bucket_s2=lambda m, lb: "any",
    )

    result = apply_sequential_policy(metrics, lb=1.0, tables=tables, lambda_=1.0)

    assert result.total_time == 0.001
    assert result.gap == 0.5
    assert result.path == ["greedy"]


def test_quantile_bucket_assigns_low_mid_high():
    values = list(range(1, 101))  # 1..100
    thresholds = compute_quantile_thresholds(values, n_bins=3)
    assert quantile_bucket(1, thresholds) == "q0"
    assert quantile_bucket(50, thresholds) == "q1"
    assert quantile_bucket(100, thresholds) == "q2"


def _make_metrics(greedy_gap, medium_gap, medium_time, exact_time):
    return {
        "greedy": RungMetrics(objective=1.0, feasible=True, gap=greedy_gap, solve_time=0.001),
        "medium": RungMetrics(objective=1.0, feasible=True, gap=medium_gap, solve_time=medium_time),
        "exact": RungMetrics(objective=1.0, feasible=True, gap=0.0, solve_time=exact_time),
    }


def test_fit_learns_to_accept_at_s2_when_escalating_to_exact_is_not_worth_it():
    # Medium is already near-perfect and exact is comparatively slow: at a
    # moderate lambda, backward induction should learn "accept" at s2.
    metrics_list = [_make_metrics(0.5, 0.01, 0.01, 1.0) for _ in range(30)]
    lb_list = [1.0] * 30

    tables = fit_sequential_tables(
        metrics_list,
        lb_list,
        lambda_=1.0,
        bucket_s1=lambda m, lb: "any",
        bucket_s2=lambda m, lb: "any",
        min_bucket_size=5,
    )

    assert tables.s2_decisions[tables.bucket_s2(metrics_list[0], 1.0)] == "accept"


def test_fit_learns_to_escalate_at_s2_when_exact_is_cheap_and_medium_is_bad():
    # Medium is bad and exact is fast: escalating clearly pays off.
    metrics_list = [_make_metrics(0.5, 0.9, 0.001, 0.001) for _ in range(30)]
    lb_list = [1.0] * 30

    tables = fit_sequential_tables(
        metrics_list,
        lb_list,
        lambda_=1.0,
        bucket_s1=lambda m, lb: "any",
        bucket_s2=lambda m, lb: "any",
        min_bucket_size=5,
    )

    assert tables.s2_decisions[tables.bucket_s2(metrics_list[0], 1.0)] == "escalate"


def test_fit_merges_sparse_buckets_instead_of_leaving_them_undecided():
    # 1 "rare" instance is below min_bucket_size=5 and must be folded into a
    # merged bucket rather than left with no computed decision.
    common = [_make_metrics(0.5, 0.01, 0.01, 1.0) for _ in range(20)]
    rare = [_make_metrics(0.5, 0.01, 0.01, 1.0)]
    metrics_list = common + rare
    lb_list = [1.0] * len(metrics_list)

    def bucket_s1(m, lb):
        return "rare" if m is rare[0] else "common"

    tables = fit_sequential_tables(
        metrics_list,
        lb_list,
        lambda_=1.0,
        bucket_s1=bucket_s1,
        bucket_s2=lambda m, lb: "any",
        min_bucket_size=5,
    )

    # The rare instance's resolved bucket must have a real, computed decision.
    resolved = tables.bucket_s1(rare[0], 1.0)
    assert resolved in tables.s1_decisions


def test_accepting_at_medium_pays_greedy_plus_medium_time():
    metrics = {
        "greedy": RungMetrics(objective=100.0, feasible=True, gap=0.5, solve_time=0.001),
        "medium": RungMetrics(objective=80.0, feasible=True, gap=0.2, solve_time=0.01),
        "exact": RungMetrics(objective=70.0, feasible=True, gap=0.0, solve_time=1.5),
    }
    tables = SequentialTables(
        s1_decisions={"any": "escalate"},
        s2_decisions={"any": "accept"},
        bucket_s1=lambda m, lb: "any",
        bucket_s2=lambda m, lb: "any",
    )

    result = apply_sequential_policy(metrics, lb=1.0, tables=tables, lambda_=1.0)

    assert result.total_time == 0.001 + 0.01
    assert result.gap == 0.2
    assert result.path == ["greedy", "medium"]
