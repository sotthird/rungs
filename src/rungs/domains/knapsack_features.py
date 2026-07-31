from rungs.domains.knapsack import KnapsackInstance


def cheap_lower_bound(inst: KnapsackInstance) -> float:
    """A valid, no-solving lower bound: the single best item that fits alone.

    Deliberately NOT the greedy rung's own value — reusing a rung's own
    output as its bucketing denominator would make the ratio trivially 1.0
    and destroy the bucket's discriminative power (see facility_location's
    cheap_lower_bound for the same reasoning).
    """
    fits = inst.weights <= inst.capacity
    if not fits.any():
        return 0.0
    return float(inst.values[fits].max())


def features(inst: KnapsackInstance) -> dict[str, float]:
    ratio = inst.values / inst.weights
    return {
        "capacity_ratio": float(inst.capacity / inst.weights.sum()),
        "value_mean": float(inst.values.mean()),
        "value_std": float(inst.values.std()),
        "weight_mean": float(inst.weights.mean()),
        "weight_std": float(inst.weights.std()),
        "value_weight_ratio_mean": float(ratio.mean()),
        "value_weight_ratio_std": float(ratio.std()),
        "n_items": float(len(inst.values)),
    }
