import pytest

from rungs.evaluate import gain_fraction, loss, quality_gap


def test_quality_gap_zero_when_matches_exact():
    assert quality_gap(objective=10.0, feasible=True, obj_exact=10.0, gap_infeasible=1.0) == 0.0


def test_quality_gap_relative_to_exact():
    assert quality_gap(
        objective=12.0, feasible=True, obj_exact=10.0, gap_infeasible=1.0
    ) == pytest.approx(0.2)


def test_quality_gap_infeasible_gets_fixed_penalty_not_partial_credit():
    # Even a near-feasible solution gets the full fixed penalty, no partial credit.
    assert quality_gap(objective=10.01, feasible=False, obj_exact=10.0, gap_infeasible=1.0) == 1.0


def test_loss_combines_gap_and_time():
    assert loss(gap=0.2, solve_time=0.5, lambda_=2.0) == pytest.approx(0.2 + 2.0 * 0.5)


def test_gain_fraction_is_zero_at_best_fixed_and_one_at_oracle():
    assert gain_fraction(loss_best_fixed=1.0, loss_policy=1.0, loss_oracle=0.0) == 0.0
    assert gain_fraction(loss_best_fixed=1.0, loss_policy=0.0, loss_oracle=0.0) == 1.0


def test_gain_fraction_is_zero_when_best_fixed_equals_oracle():
    # No exploitable variation exists; the fraction is undefined but must not
    # divide by zero — define it as 0 (no headroom to capture).
    assert gain_fraction(loss_best_fixed=0.5, loss_policy=0.5, loss_oracle=0.5) == 0.0
