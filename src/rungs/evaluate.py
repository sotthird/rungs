def quality_gap(
    objective: float | None, feasible: bool, obj_exact: float, gap_infeasible: float
) -> float:
    if not feasible or objective is None:
        return gap_infeasible
    return (objective - obj_exact) / obj_exact


def loss(gap: float, solve_time: float, lambda_: float) -> float:
    return gap + lambda_ * solve_time


def gain_fraction(loss_best_fixed: float, loss_policy: float, loss_oracle: float) -> float:
    """Fraction of the oracle-achievable improvement over best_fixed that
    policy captures. Bounded in [0, 1] by construction (DESIGN.md §11)."""
    headroom = loss_best_fixed - loss_oracle
    if headroom <= 0:
        return 0.0
    return (loss_best_fixed - loss_policy) / headroom
