from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class KnapsackInstance:
    values: np.ndarray  # (n_items,)
    weights: np.ndarray  # (n_items,) integers
    capacity: int
    meta: dict = field(default_factory=dict)  # generation params, analysis only


def generate_instance(
    rng: np.random.Generator,
    n_items: int,
    capacity_ratio_range: tuple[float, float],
) -> KnapsackInstance:
    """capacity_ratio = capacity / total_weight, the knapsack analogue of
    facility location's tightness — the primary hardness driver."""
    weights = rng.integers(1, 50, size=n_items)
    values = rng.integers(1, 100, size=n_items)
    capacity_ratio = rng.uniform(*capacity_ratio_range)
    capacity = max(1, int(round(capacity_ratio * weights.sum())))

    meta = {"capacity_ratio": capacity_ratio}
    return KnapsackInstance(values=values, weights=weights, capacity=capacity, meta=meta)


def generate_dataset(
    rng: np.random.Generator,
    n_instances: int,
    n_items: int,
    capacity_ratio_range: tuple[float, float],
    is_valid,
) -> list[KnapsackInstance]:
    dataset = []
    while len(dataset) < n_instances:
        inst = generate_instance(rng, n_items=n_items, capacity_ratio_range=capacity_ratio_range)
        if is_valid(inst):
            dataset.append(inst)
    return dataset
