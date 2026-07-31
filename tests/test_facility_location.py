import numpy as np

from rungs.domains.facility_location import generate_instance


def test_instance_is_feasible():
    rng = np.random.default_rng(0)
    inst = generate_instance(
        rng,
        n_sites=15,
        n_customers=60,
        tightness_range=(1.02, 2.5),
        demand_cv_range=(0.1, 0.8),
        fixed_transport_ratio_range=(0.2, 5.0),
    )
    assert inst.capacity.sum() >= inst.demand.sum()
